"""roundopt/capture: record REALISTIC _refine_act_values inputs end-to-end.

Never modifies solution.py.  Loads it as a module, calibrates synthetic
groups (decomp-suite conventions: C{1024,2048} x spread{0.5,0.9} x
outlier{0,0.002}, N{1024,8192}, calib (10,128,512,1024), test
T=(10,128,512,1024,1024)), then wraps _refine_act_values while running
hif4_dynamic_quantize_activation on each test activation to snapshot
(x, values, unit, gw, gwf).  Two seeds per config (decomp enumeration seed
and seed+100000) -> 32 groups / 160 calls.  gw/gwf are stored once per
group (shared objects across calls).

Usage: python dev/roundopt/capture.py [--groups N] [--only name,...]
Output: dev/roundopt/capture/<name>.pt  {args: [(x,v0,unit,gwid), ...],
gw: (C,C), gwf: (C,C), T list}
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.dirname(HERE)
ROOT = os.path.dirname(DEV)
sys.path.insert(0, DEV)
import synth  # noqa: E402

SOL_PATH = os.path.join(ROOT, "example", "solution", "solution.py")
OUT = os.path.join(HERE, "capture")
CALIB_T = (10, 128, 512, 1024)
TEST_T = (10, 128, 512, 1024, 1024)
CS = (1024, 2048)
NS = (1024, 8192)
SPREADS = (0.5, 0.9)
OUTLIERS = (0.0, 0.002)


def load_sol():
    spec = importlib.util.spec_from_file_location("_ro_sol", SOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def iter_grid(limit=None):
    out = []
    i = 0
    for C in (512, 1024, 2048, 4096, 8192):
        for N in (1024, 8192):
            for spread in (0.5, 0.9):
                for outp in (0.0, 0.002):
                    if C in CS:
                        out.append((f"c{C}_n{N}_s{spread}_o{outp}",
                                    4200 + 13 * i, C, N, spread, outp))
                    i += 1
    # second seed per config for coverage (same enumeration offset)
    out2 = [(n + "_b", s + 100000, C, N, sp, op)
            for (n, s, C, N, sp, op) in out]
    all_g = out + out2
    return all_g[:limit] if limit else all_g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=None)
    ap.add_argument("--only", type=str, default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    S = load_sol()

    grid = iter_grid(args.groups)
    if args.only:
        keep = set(args.only.split(","))
        grid = [g for g in grid if g[0] in keep]

    for gi, (name, seed, C, N, spread, outp) in enumerate(grid):
        path = os.path.join(OUT, f"{name}.pt")
        if os.path.exists(path):
            print(f"[{gi + 1}/{len(grid)}] {name}: cached")
            continue
        tokens = CALIB_T + TEST_T
        g = synth.make_linear_group(seed, N, C, tokens=tokens,
                                    spread=spread, outlier_p=outp)
        nc = len(CALIB_T)
        calib = g["calib_activation_list"][:nc]
        test = g["calib_activation_list"][nc:]
        torch.manual_seed(0)
        cal = S.hif4_calibration_and_quantize_weight(
            g["weight"][0], g["weight"][1], calib)

        recs = []
        first_gram = []      # (gw, gwf) fp32 from the first call

        orig = S._refine_act_values

        def wrapped(x, values, unit, gw, gwf):
            if first_gram:
                assert torch.equal(gw, first_gram[0][0])
                assert torch.equal(gwf, first_gram[0][1])
            else:
                first_gram.append((gw.clone(), gwf.clone()))
            recs.append({"x": x.clone(), "v0": values.clone(),
                         "unit": unit.clone(), "T": int(x.shape[0])})
            out = orig(x, values, unit, gw, gwf)
            recs[-1]["v1_ref"] = out.clone()
            return out

        S._refine_act_values = wrapped
        try:
            for pair in test:
                S.hif4_dynamic_quantize_activation(
                    pair[0], pair[1], cal["activation_state"])
        finally:
            S._refine_act_values = orig
        # store shared grams once, in bf16 (the fp32 the calls see is an
        # exact bf16->fp32 widening; verify then reconstruct at load)
        gw32, gwf32 = first_gram[0]
        gwb = cal["activation_state"].get("gw")
        gwfb = cal["activation_state"].get("gwf")
        assert gwb is not None and torch.equal(gwb.float(), gw32)
        assert gwfb is not None and torch.equal(gwfb.float(), gwf32)
        torch.save({"recs": recs, "gw": gwb.clone(), "gwf": gwfb.clone(),
                    "C": C, "N": N, "spread": spread, "outp": outp,
                    "seed": seed}, path)
        Ts = ",".join(str(r["T"]) for r in recs)
        print(f"[{gi + 1}/{len(grid)}] {name}: captured T=[{Ts}]")


if __name__ == "__main__":
    main()
