"""roundopt/bench: per-call timing of _refine_act_values variants.

Measures the ship module function vs core.refine_active variants on the
captured realistic inputs (each capture = one (T,C,sweep-tier) cell).
Reports median wall time over repeats.

Usage: python dev/roundopt/bench.py [--reps 3] [--variants ship,v2n]
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core  # noqa: E402

CAP = os.path.join(HERE, "capture")


def time_call(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--variants", type=str,
                    default="ship,v1,v2,v2n")
    args = ap.parse_args()
    S = core.load_sol()
    variants = args.variants.split(",")
    cells = {}      # (T, C) -> per-variant times
    counts = {}     # (T, C) -> n
    for f in sorted(glob.glob(os.path.join(CAP, "*.pt"))):
        d = torch.load(f, weights_only=False)
        gw = d["gw"].float()
        gwf = d["gwf"].float()
        C = d["C"]
        for rec in d["recs"]:
            T = rec["T"]
            key = (T, C)
            counts[key] = counts.get(key, 0) + 1
            res = cells.setdefault(key, {})
            for v in variants:
                if v == "ship":
                    fn = lambda: S._refine_act_values(
                        rec["x"], rec["v0"], rec["unit"], gw, gwf)
                elif v == "v1":
                    fn = lambda: core.refine_active(
                        rec["x"], rec["v0"], rec["unit"], gw, gwf, S=S)
                elif v == "v2":
                    fn = lambda: core.refine_active(
                        rec["x"], rec["v0"], rec["unit"], gw, gwf, S=S,
                        v2=True)
                elif v == "v2n":
                    fn = lambda: core.refine_active(
                        rec["x"], rec["v0"], rec["unit"], gw, gwf, S=S,
                        v2=True, np_thresh=32)
                else:
                    raise ValueError(v)
                res.setdefault(v, []).append(time_call(fn, args.reps))
    print(f"median per-call seconds by (T,C); n calls each, reps="
          f"{args.reps} (captures are the realistic dynamic inputs)")
    hdr = f"{'T':>5} {'C':>5} {'n':>3} " + " ".join(
        f"{v:>9}" for v in variants) + "   speedup(best var)"
    print(hdr)
    for key in sorted(cells):
        T, C = key
        row = f"{T:>5} {C:>5} {counts[key]:>3} "
        best = None
        for v in variants:
            m = statistics.median(cells[key][v])
            row += f" {m:8.4f}"
            if v != "ship":
                if best is None or m < best[1]:
                    best = (v, m)
        ship = statistics.median(cells[key]["ship"]) if "ship" in cells[key] \
            else None
        sp = f"{ship / best[1]:5.2f}x ({best[0]})" if ship and best else ""
        print(row + "   " + sp)


if __name__ == "__main__":
    main()
