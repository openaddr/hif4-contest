"""Gate 4: interleaved A/B timing (medians of 3 alternating reps in one
process) of the v19 baseline vs the patched live solution.
Groups: real mini linear, synthetic c2048_n8192, synthetic c8192_n8192
(calib (10,128,512,1024), test (10,128,512,1024,1024), spread 0.5).
"""
from __future__ import annotations

import importlib.util
import os
import statistics
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "dev"))
import synth  # noqa: E402

AUDIT = os.path.join(ROOT, "dev", "audit")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(sol, g):
    torch.manual_seed(0)
    t0 = time.perf_counter()
    out = sol.hif4_calibration_and_quantize_weight(
        g["weight"][0], g["weight"][1], g["calib_activation_list"])
    t_cal = time.perf_counter() - t0
    st = out["activation_state"]
    t_dyn = 0.0
    for pair in g["test_activation_list"]:
        t0 = time.perf_counter()
        sol.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
        t_dyn += time.perf_counter() - t0
    return t_cal, t_dyn


def main():
    base = load(os.path.join(AUDIT, "solution_v19_baseline.py"), "_t_base")
    live = load(os.path.join(ROOT, "example", "solution", "solution.py"), "_t_live")
    groups = []
    lin = torch.load(os.path.join(ROOT, "example", "mini_sample", "linear.pt"),
                     weights_only=True)[0]
    groups.append(("mini (real, 2048x8192)", lin))

    def make(seed, N, C):
        g = synth.make_linear_group(seed, N, C, tokens=(10, 128, 512, 1024),
                                    spread=0.5, outlier_p=0.0, w_spread=0.3)
        g2 = synth.make_linear_group(seed + 7777, N, C,
                                     tokens=(10, 128, 512, 1024, 1024),
                                     spread=0.5, outlier_p=0.0, w_spread=0.3)
        g["test_activation_list"] = g2["test_activation_list"]
        return g
    groups.append(("c2048_n8192", make(3100 + 2048, 8192, 2048)))
    groups.append(("c8192_n8192", make(3100 + 8192, 8192, 8192)))

    print(f"{'group':<24s} {'calib base->live':>22s} {'dyn base->live':>20s} "
          f"{'saved local':>12s} {'saved online':>13s}")
    tot_b = tot_l = 0.0
    for name, g in groups:
        cb, cl, db, dl = [], [], [], []
        for _ in range(3):
            a = run(base, g)
            b = run(live, g)
            cb.append(a[0]); db.append(a[1]); cl.append(b[0]); dl.append(b[1])
        mcb, mcl, mdb, mdl = (statistics.median(cb), statistics.median(cl),
                              statistics.median(db), statistics.median(dl))
        sv, svo = (mcb - mcl + mdb - mdl), (mcb - mcl + mdb - mdl) / 4.8
        tot_b += mcb + mdb
        tot_l += mcl + mdl
        print(f"{name:<24s} {mcb:8.2f}s -> {mcl:6.2f}s    {mdb:7.2f}s -> {mdl:5.2f}s "
              f"{sv:+9.2f}s   {svo:+8.2f}s")
    print(f"{'3-group total':<24s} {tot_b:8.2f}s -> {tot_l:6.2f}s "
          f"({(tot_b - tot_l) / tot_b * 100:+.1f}% faster)")


if __name__ == "__main__":
    main()
