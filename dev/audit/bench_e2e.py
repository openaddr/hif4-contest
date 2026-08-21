"""End-to-end interleaved A/B of the tuned savings bundle vs baseline.

Bundle (all measured bit-identical individually):
  f1  _quant_chunk -> KB2-batched vectorized version (weights always; act
      quant only when T*C >= 4M where it measured positive)
  f1b ROW_CHUNK 2048 -> 512 (weight quant only; bit-identical)
  f2  _gptq_quantize_values -> numpy impl when R <= 2048 (dynamic + proxy +
      act-search GPTQ), torch when R > 2048 (weight GPTQ at N=8192)
  f3  act-GPTQ double-Cholesky skip (textual patch)

Interleaved A,B,A,B,A... reps, medians reported; bit-identity asserted on
calibration params, state, and all dynamic-call params.
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_speed import (_qc_vec_impl, _gptq_np, build_f3, load_variant,  # noqa: E402
                       load_group, eq_params, eq_state)


def make_combo(base):
    import inspect
    import exp_speed
    src = build_f3()
    qc_src = inspect.getsource(_qc_vec_impl)
    np_src = inspect.getsource(_gptq_np)

    prelude = f'''
import numpy as np
SF_MIN_V, SF_MAX_V = SF_MIN, SF_MAX
{qc_src}
{np_src}


def _gptq_dispatch(x, unit, hinv):
    if x.shape[0] > 2048:
        return _ORIG_GPTQ(x, unit, hinv)
    return _gptq_np(x, unit, hinv, GPTQ_BLOCK)


def _quantize_weighted_tuned(x2d, wgt, grid=CAND_GRID):
    R, C = x2d.shape
    nb = C // 64
    out = {{k: [] for k in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant")}}
    if not USE_WEIGHTS:
        wgt = torch.ones(1, C, dtype=torch.float32)
    else:
        wgt = wgt / wgt.mean().clamp_min(1e-30)
        wgt = wgt.clamp(0.25, 4.0)
    w2d = wgt if wgt.shape == (R, C) else wgt.expand(R, C)
    fn = (_qc_vec2 if R * C >= 4_000_000 else _ORIG_QUANT_CHUNK)
    for s0 in range(0, R, ROW_CHUNK):
        x_chunk = x2d[s0:s0 + ROW_CHUNK]
        p = fn(x_chunk.reshape(-1, nb, 8, 2, 4),
               w2d[s0:s0 + ROW_CHUNK].reshape(-1, nb, 8, 2, 4), grid)
        for k in out:
            out[k].append(p[k])
    return {{k: torch.cat(v, dim=0) for k, v in out.items()}}

_ORIG_GPTQ = _gptq_quantize_values
_ORIG_QUANT_CHUNK = _quant_chunk
_qc_vec2 = lambda a, b, g: _qc_vec_impl(a, b, g, 2)
_quant_chunk = _qc_vec2
_gptq_quantize_values = _gptq_dispatch
_quantize_weighted = _quantize_weighted_tuned
'''
    inject_at = "def hif4_calibration_and_quantize_weight("
    assert src.count(inject_at) == 1
    src = src.replace(inject_at, prelude + "\n\n" + inject_at)
    mod = load_variant(patch_src=src)
    return mod


def run_once(sol, g):
    torch.manual_seed(0)
    t0 = time.perf_counter()
    out = sol.hif4_calibration_and_quantize_weight(
        g["weight"][0], g["weight"][1], g["calib_activation_list"])
    t_cal = time.perf_counter() - t0
    st = out["activation_state"]
    t_dyn = 0.0
    ps = []
    for pair in g["test_activation_list"]:
        t0 = time.perf_counter()
        p = sol.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
        t_dyn += time.perf_counter() - t0
        ps.append(p)
    return out, ps, t_cal, t_dyn


def main():
    names = sys.argv[1:] or ["c4096_n4096"]
    base = load_variant()
    combo = make_combo(base)
    reps = 3
    for name in names:
        g = load_group(name)
        cals_b, cals_c, dyns_b, dyns_c = [], [], [], []
        ok = None
        for r in range(reps):
            ob, pb, tcb, tdb = run_once(base, g)
            oc, pc, tcc, tdc = run_once(combo, g)
            cals_b.append(tcb); cals_c.append(tcc)
            dyns_b.append(tdb); dyns_c.append(tdc)
            if r == 0:
                ok = (eq_params(ob["weight_params"], oc["weight_params"])
                      and eq_state(ob["activation_state"], oc["activation_state"])
                      and all(eq_params(a, b) for a, b in zip(pb, pc)))
        mc_b, mc_c = statistics.median(cals_b), statistics.median(cals_c)
        md_b, md_c = statistics.median(dyns_b), statistics.median(dyns_c)
        print(f"{name}: calib {mc_b:6.2f} -> {mc_c:6.2f} (save {mc_b - mc_c:+6.2f}) | "
              f"dyn {md_b:6.2f} -> {md_c:6.2f} (save {md_b - md_c:+6.2f}) | "
              f"total save {mc_b - mc_c + md_b - md_c:+6.2f}s | "
              f"bit-identical: {ok}")
        print(f"    reps calib base {' '.join(f'{t:.2f}' for t in cals_b)} | "
              f"combo {' '.join(f'{t:.2f}' for t in cals_c)}")
        print(f"    reps dyn   base {' '.join(f'{t:.2f}' for t in dyns_b)} | "
              f"combo {' '.join(f'{t:.2f}' for t in dyns_c)}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
