"""roundopt/verify_e2e: bit-identity gate for the PATCHED solution module.

A. function level: patched._refine_act_values vs the ship module's, on all
   captured realistic refine inputs (160 calls: 32 groups x 5 test T).
B. end-to-end: calibrate ship vs patched on synthetic groups (both seeded
   identically) -> weight_params / q_used-derived state must be
   torch.equal; then dynamic quantize every test activation -> returned
   HiF4 params must be torch.equal, with per-call timing.
C. mini_sample/linear.pt: same end-to-end flow on the real example data.

Usage: python dev/roundopt/verify_e2e.py [--groups 8] [--skip-fn]
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.dirname(HERE)
ROOT = os.path.dirname(DEV)
sys.path.insert(0, DEV)
import synth  # noqa: E402

SOL = os.path.join(ROOT, "example", "solution", "solution.py")
PATCHED = os.path.join(HERE, "patched", "solution.py")
CAP = os.path.join(HERE, "capture")
MINI = os.path.join(ROOT, "example", "mini_sample", "linear.pt")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tensors_equal(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    return True     # non-tensor leaf: compare below


def deep_equal(x, y, path=""):
    ok = True
    if isinstance(x, dict) and isinstance(y, dict):
        if set(x.keys()) != set(y.keys()):
            print(f"  KEY MISMATCH at {path}")
            return False
        for k in x:
            ok = deep_equal(x[k], y[k], f"{path}.{k}") and ok
        return ok
    if isinstance(x, torch.Tensor):
        if not tensors_equal(x, y):
            print(f"  TENSOR MISMATCH at {path}: {tuple(x.shape)} vs "
                  f"{tuple(y.shape) if isinstance(y, torch.Tensor) else y}")
            return False
        return True
    if x != y:
        print(f"  VALUE MISMATCH at {path}: {x!r} vs {y!r}")
        return False
    return True


def e2e_group(S, P, g, name):
    ok = True
    torch.manual_seed(0)
    t0 = time.perf_counter()
    cal_s = S.hif4_calibration_and_quantize_weight(
        g["weight"][0], g["weight"][1], g["calib_activation_list"])
    ts = time.perf_counter() - t0
    torch.manual_seed(0)
    t0 = time.perf_counter()
    cal_p = P.hif4_calibration_and_quantize_weight(
        g["weight"][0], g["weight"][1], g["calib_activation_list"])
    tp = time.perf_counter() - t0
    ok = deep_equal(cal_s, cal_p, f"{name}.cal") and ok
    for i, pair in enumerate(g["test_activation_list"]):
        t0 = time.perf_counter()
        p_s = S.hif4_dynamic_quantize_activation(
            pair[0], pair[1], cal_s["activation_state"])
        t_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        p_p = P.hif4_dynamic_quantize_activation(
            pair[0], pair[1], cal_p["activation_state"])
        t_p = time.perf_counter() - t0
        ok = deep_equal(p_s, p_p, f"{name}.test{i}") and ok
        T = pair[0].shape[0]
        print(f"  {name} T={T:5d}: ship {t_s*1e3:8.1f}ms patched "
              f"{t_p*1e3:8.1f}ms  ({t_s/t_p:4.2f}x)")
    print(f"  {name} calib: ship {ts:.1f}s patched {tp:.1f}s")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--skip-fn", action="store_true")
    args = ap.parse_args()
    S = load(SOL, "_ve_ship")
    P = load(PATCHED, "_ve_patched")

    if not args.skip_fn:
        n = 0
        for f in sorted(glob.glob(os.path.join(CAP, "*.pt"))):
            d = torch.load(f, weights_only=False)
            gw = d["gw"].float()
            gwf = d["gwf"].float()
            for i, rec in enumerate(d["recs"]):
                a = S._refine_act_values(rec["x"], rec["v0"], rec["unit"],
                                         gw, gwf)
                b = P._refine_act_values(rec["x"], rec["v0"], rec["unit"],
                                         gw, gwf)
                assert torch.equal(a, b), \
                    f"FN MISMATCH {os.path.basename(f)}#{i}"
                n += 1
        print(f"A. function gate: {n}/160 patched==ship refine calls "
              f"torch.equal OK")

    from capture import CALIB_T, TEST_T, iter_grid
    ok = True
    print("B. synthetic groups (calibrate both, compare everything):")
    for name, seed, C, N, spread, outp in iter_grid()[:args.groups]:
        tokens = CALIB_T + TEST_T
        g = synth.make_linear_group(seed, N, C, tokens=tokens,
                                    spread=spread, outlier_p=outp)
        nc = len(CALIB_T)
        g = {"weight": g["weight"],
             "calib_activation_list": g["calib_activation_list"][:nc],
             "test_activation_list": g["calib_activation_list"][nc:]}
        ok = e2e_group(S, P, g, name) and ok

    print("C. mini_sample/linear.pt:")
    mini = torch.load(MINI, weights_only=False)[0]
    ok = e2e_group(S, P, mini, "mini_linear") and ok
    print("E2E", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
