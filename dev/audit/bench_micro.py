"""Interleaved A/B microbenchmarks of the targeted inner functions.

Each comparison alternates baseline/variant within the same process and takes
the median of N reps, to defeat the episodic system-load drift seen in
sequential whole-run timings.
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_speed import _qc_vec_impl, _gptq_np, load_variant, load_group  # noqa: E402

REP = int(os.environ.get("REP", "5"))


def ab(label, fn_a, fn_b, rep=REP, warm=1):
    for _ in range(warm):
        fn_a()
        fn_b()
    ta, tb = [], []
    for _ in range(rep):
        t0 = time.perf_counter(); fn_a(); ta.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); fn_b(); tb.append(time.perf_counter() - t0)
    ma, mb = statistics.median(ta), statistics.median(tb)
    print(f"{label:<58s} base {ma:7.3f}s  var {mb:7.3f}s  save {ma - mb:+7.3f}s "
          f"({(ma - mb) / ma * 100 if ma else 0:+5.1f}%)")
    return ma - mb


def main():
    torch.set_num_threads(torch.get_num_threads())
    base = load_variant()

    # ---- 1. weight quant: _quantize_weighted (16-cand grid) ----
    for name in ("c2048_n8192", "c4096_n8192", "c8192_n8192"):
        g = load_group(name)
        w = base.dequantize_nvfp4(g["weight"][0], g["weight"][1]).float()
        ones = torch.ones(1, w.shape[1])
        out = {}

        def f_base():
            out["b"] = base._quantize_weighted(w, ones)

        for KB in (2, 4, 8):
            def f_vec(KB=KB):
                base._quant_chunk, orig = (lambda a, b, g_, KB=KB: _qc_vec_impl(a, b, g_, KB)), base._quant_chunk
                try:
                    out["v"] = base._quantize_weighted(w, ones)
                finally:
                    base._quant_chunk = orig
            ab(f"wquant {name} KB{KB}", f_base, f_vec)
        ident = all(torch.equal(out["b"][k], out["v"][k]) for k in out["b"])
        print(f"    last outputs bit-identical: {ident}")

    # ---- 2. dyn-side act quant: _quantize_weighted on (T, C) 6-cand ----
    for C, T in ((2048, 512), (8192, 1024), (4096, 1024)):
        x = torch.randn(T, C)
        ones = torch.ones(1, C)
        out = {}

        def f_base():
            out["b"] = base._quantize_weighted(x, ones)

        def f_vec():
            orig = base._quant_chunk
            base._quant_chunk = lambda a, b, g_: _qc_vec_impl(a, b, g_, 6)
            try:
                out["v"] = base._quantize_weighted(x, ones)
            finally:
                base._quant_chunk = orig
        ab(f"acquanT T={T} C={C} KB6", f_base, f_vec)
        ident = all(torch.equal(out["b"][k], out["v"][k]) for k in out["b"])
        print(f"    bit-identical: {ident}")

    # ---- 3. GPTQ values: torch vs numpy by (R, C) ----
    for C in (2048, 4096, 8192):
        A = torch.randn(2 * C, C)
        H = A.T @ A + torch.eye(C) * 0.5
        U = base._upper_cholesky_inv(H)
        for R in (10, 128, 1024, 8192):
            x = torch.randn(R, C)
            u = torch.rand(R, C) + 0.5
            out = {}

            def f_t():
                out["t"] = base._gptq_quantize_values(x, u, U)

            def f_n():
                out["n"] = _gptq_np(x, u, U, 128)
            ab(f"gptq R={R:5d} C={C}", f_t, f_n, rep=3)
            print(f"    bit-identical: {torch.equal(out['t'], out['n'])}")

    # ---- 4. GPTQ_BLOCK sweep on weight GPTQ (R=8192) ----
    g8 = load_group("c8192_n8192")
    w8 = base.dequantize_nvfp4(g8["weight"][0], g8["weight"][1]).float()
    acts = [base.dequantize_nvfp4(aq, as_).float() for aq, as_ in g8["calib_activation_list"]]
    Hs = torch.zeros(8192, 8192)
    for a in acts[:-1]:
        Hs += a.T @ a
    U8 = base._upper_cholesky_inv(Hs)
    pw = base._quantize_weighted(w8, torch.ones(1, 8192))
    unit8 = base._params_unit_flat(pw)
    for gb in (64, 128, 256, 512):
        base.GPTQ_BLOCK = gb
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            q = base._gptq_quantize_values(w8, unit8, U8)
            ts.append(time.perf_counter() - t0)
        print(f"gptq BLOCK={gb:4d} R=8192 C=8192: median {statistics.median(ts):7.3f}s "
              f"(all {' '.join(f'{t:.2f}' for t in ts)})")
    base.GPTQ_BLOCK = 128

    # ---- 5. chol cost by C (f3 skips one _upper_cholesky_inv(Ha)) ----
    for C in (2048, 4096, 8192):
        A = torch.randn(2 * C, C)
        Ha = A.T @ A
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            base._upper_cholesky_inv(Ha)
            ts.append(time.perf_counter() - t0)
        print(f"upper_chol_inv C={C}: median {statistics.median(ts):.3f}s "
              f"(f3 saves one of these when act-order chol succeeds)")

    # ---- 6. ROW_CHUNK effect (weight quant already in base=2048) ----
    for rc in (512, 4096):
        base.ROW_CHUNK = rc
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            base._quantize_weighted(w8, torch.ones(1, 8192))
            ts.append(time.perf_counter() - t0)
        print(f"wquant ROW_CHUNK={rc} c8192: median {statistics.median(ts):.3f}s")
    base.ROW_CHUNK = 2048


if __name__ == "__main__":
    main()
