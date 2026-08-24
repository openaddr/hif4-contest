"""r1024v: (1) clean interleaved overhead timing for mechanism B;
(2) output-legality validation for the A/B switch-ON paths.

Usage: python dev/r1024v/timing.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.dirname(HERE)
ROOT = os.path.dirname(DEV)
sys.path.insert(0, DEV)
sys.path.insert(0, os.path.join(ROOT, "example"))
import hif4 as H          # noqa: E402
import measure as M       # noqa: E402
import self_check as SC   # noqa: E402


def time_b():
    print("=== B overhead: interleaved min-of-5 per call (s, local) ===")
    for name, T in (("c2048_n8192_s0.5_o0.0", 10),
                    ("c2048_n8192_s0.5_o0.0", 1024),
                    ("c1024_n8192_s0.9_o0.002", 1024),
                    ("c2048_n8192_s0.5_o0.0", 2048)):
        _, seed, C, N, spread, outp = M.grid_by_name(name)
        group = M.make_group(name)
        cc = M.calibrate(name, group)
        st = cc["cal"]["activation_state"]
        wp = cc["cal"]["weight_params"]
        if T == 2048:
            pair = M.make_extra_act(seed, C, spread, outp, 2048)
        else:
            pair = next(p for p in group["test_activation_list"]
                        if p[0].shape[0] == T)
        best = {t: None for t in ("B0", "B2", "B3")}
        for rep in range(5):
            for tag, cfg in M.B_CFGS:
                mod = M.apply_cfg(cfg)
                t0 = time.perf_counter()
                mod.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
                el = time.perf_counter() - t0
                if best[tag] is None or el < best[tag]:
                    best[tag] = el
        b0 = best["B0"]
        print(f"{name} T={T}: B0 {b0:.3f} | B2 {best['B2']:.3f} "
              f"(+{(best['B2']-b0)*1e3:.0f}ms) | B3 {best['B3']:.3f} "
              f"(+{(best['B3']-b0)*1e3:.0f}ms)")
        sys.stdout.flush()


def legality():
    print("=== legality: A/B switch-ON paths (validate_hif4_params) ===")
    lin = torch.load(os.path.join(ROOT, "example", "mini_sample", "linear.pt"),
                     weights_only=True, map_location="cpu")[0]
    group = {"weight": lin["weight"],
             "calib_activation_list": lin["calib_activation_list"],
             "test_activation_list": lin["test_activation_list"]}
    cc = M.calibrate("mini", group)
    st = cc["cal"]["activation_state"]
    wp = cc["cal"]["weight_params"]
    errs = SC.validate_hif4_params(wp, tuple(group["weight"][0].shape), "w")
    errs += SC.validate_frozen_state(st, "state")
    mod = M.apply_cfg({"PREFIX_REFINE": True, "PREFIX_SWEEPS": 20,
                       "DYN_SELECT": 3})
    for i, pair in enumerate(group["test_activation_list"]):
        p = mod.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
        errs += SC.validate_hif4_params(p, tuple(pair[0].shape), f"a{i}")
    print(f"mini (B3+prefix-ready): errors = {errs or 'NONE'}")

    for name, T in (("c2048_n8192_s0.5_o0.0", 2048),
                    ("c4096_n8192_s0.5_o0.0", 4096)):
        _, seed, C, N, spread, outp = M.grid_by_name(name)
        g = M.make_group(name)
        cc = M.calibrate(name, g)
        st2 = cc["cal"]["activation_state"]
        pair = M.make_extra_act(seed, C, spread, outp, T)
        for tag, cfg in (("A8", {"PREFIX_REFINE": True, "PREFIX_SWEEPS": 0,
                                 "DYN_SELECT": 3}),
                         ("A20", {"PREFIX_REFINE": True, "PREFIX_SWEEPS": 20,
                                  "DYN_SELECT": 3}),
                         ("Full", {"DYN_TMAX_OVERRIDE": 8192,
                                   "DYN_SELECT": 3})):
            mod = M.apply_cfg(cfg)
            p = mod.hif4_dynamic_quantize_activation(pair[0], pair[1], st2)
            errs = SC.validate_hif4_params(p, tuple(pair[0].shape),
                                           f"{name}-T{T}-{tag}")
            print(f"{name} T={T} {tag}: errors = {errs or 'NONE'}")
        sys.stdout.flush()


def time_a():
    print("=== A per-call cost: interleaved min-of-3 (s, local) ===")
    for name in ("c2048_n8192_s0.5_o0.0", "c4096_n8192_s0.5_o0.0"):
        _, seed, C, N, spread, outp = M.grid_by_name(name)
        group = M.make_group(name)
        cc = M.calibrate(name, group)
        st = cc["cal"]["activation_state"]
        for T in (2048, 4096):
            pair = M.make_extra_act(seed, C, spread, outp, T)
            best = {t: None for t, _ in M.A_CFGS}
            for rep in range(3):
                for tag, cfg in M.A_CFGS:
                    mod = M.apply_cfg(cfg)
                    t0 = time.perf_counter()
                    mod.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
                    el = time.perf_counter() - t0
                    if best[tag] is None or el < best[tag]:
                        best[tag] = el
            b0 = best["A0"]
            print(f"{name} T={T}: " + " | ".join(
                f"{t} {best[t]:.2f}s(+{(best[t]-b0):.2f})" for t, _ in M.A_CFGS))
            sys.stdout.flush()


if __name__ == "__main__":
    time_b()
    time_a()
    legality()
