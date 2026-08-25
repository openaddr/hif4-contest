"""msel extension: CLOSE variant arm (resample).

Two-regime completeness: step1 tested variants at rms log-ratio(s_v/s_ship) in
[0.043, 0.62] -- all lose (mismatch penalty).  The remaining hole: a variant
fit on a DIFFERENT random subsample of the same calib rows sits much closer to
s_ship.  Prediction: its J gap collapses to noise level -- nothing to select.
Uses the cached calibrations from step1; writes results/step1b.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.dirname(HERE)
ROOT = os.path.dirname(DEV)
sys.path.insert(0, DEV)
import hif4 as H          # noqa: E402
import variants as V      # noqa: E402

import measure as ME      # noqa: E402  (same directory)

M = ME.M


def fit_resamp(group):
    """_bal_search on a DIFFERENT random subsample of all calib[:-1] rows."""
    acts_raw = [H.dequantize_nvfp4(*p).float()
                for p in group["calib_activation_list"]]
    w = H.dequantize_nvfp4(*group["weight"]).float()
    R, C = w.shape
    allrows = torch.cat([a for a in acts_raw[:-1]], dim=0)
    gen = torch.Generator().manual_seed(555 + R)
    perm = torch.randperm(allrows.shape[0], generator=gen)[: M.SMOOTH_FIT_ROWS]
    xf = allrows[perm].contiguous()
    gw_col = (w * w).sum(dim=0) + 1e-30
    gx_col = (xf * xf).sum(dim=0) + 1e-30
    return M._bal_search(xf, None, None, gw_col, gx_col), xf.shape[0]


def main():
    out_path = os.path.join(ME.RES, "step1b.json")
    out = ME.jload(out_path)
    jobs = []
    jobs.append(("mini", ME.mini_group()))
    for cn, N, C, sp, op, wsp, sh in ME.CONFIGS:
        for seed in ME.SEEDS:
            g = ME.make_group(seed, N, C, spread=sp, outlier_p=op,
                              w_spread=wsp, share=sh)
            jobs.append((f"{cn}_s{seed}", g))
    for name, group in jobs:
        if name in out:
            print(f"[b] {name}: cached, skip")
            continue
        cc = ME.calibrate(name, group)
        cal = cc["cal"]
        st = cal["activation_state"]
        s_v, nrows = fit_resamp(group)
        s_ship = st["s"].float()
        mode = st.get("mode") or 0
        lr = (s_v / s_ship).log()
        gw32 = st["gw"].float()
        gwf32 = st["gwf"].float()
        calls = []
        for pair in group["test_activation_list"]:
            x_raw = H.dequantize_nvfp4(*pair).float()
            x_ship = ME.transformed(x_raw, s_ship, mode)
            p_ship = M.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
            v_ship = M._deq_params(p_ship)
            j_def = ME.j_value(v_ship, x_ship, gw32, gwf32)
            v_v, x_v = ME.dyn_variant_values(x_raw, s_v, st)
            e = {"j_true": ME.j_value(v_v, x_ship, gw32, gwf32),
                 "j_perf": ME.j_value(x_v, x_ship, gw32, gwf32)}
            e["rel_j_true"] = (e["j_true"] - j_def) / abs(j_def)
            e["rel_j_perf"] = (e["j_perf"] - j_def) / abs(j_def)
            calls.append({"T": int(x_raw.shape[0]), "j_def": j_def,
                          "resamp": e})
        out[name] = {"rms_log_ratio": float(lr.pow(2).mean().sqrt()),
                     "max_abs_log_ratio": float(lr.abs().max()),
                     "n_fit_rows": int(nrows), "calls": calls}
        rels = [f"T{c['T']}:{c['resamp']['rel_j_true']:+.2e}" for c in calls]
        print(f"[b] {name}: rms_lr {out[name]['rms_log_ratio']:.4f} {' '.join(rels)}")
        sys.stdout.flush()
        ME.jsave(out_path, out)
    # summary
    rels = []
    perfs = []
    wins = 0
    for name in out:
        for c in out[name]["calls"]:
            r = c["resamp"]["rel_j_true"]
            rels.append(r)
            perfs.append(c["resamp"]["rel_j_perf"])
            if r < -1e-4:
                wins += 1
    rels.sort()
    perfs.sort()
    n = len(rels)
    print(f"\nresamp arm: n={n} wins={wins} median={rels[n//2]:+.3e} "
          f"p10={rels[max(0,n//10-1)]:+.3e} best={rels[0]:+.3e} worst={rels[-1]:+.3e}")
    print(f"resamp floor (j_perf): median={perfs[n//2]:+.3e} best={perfs[0]:+.3e} "
          f"worse-than-default={sum(1 for p in perfs if p > 1e-4)}/{n}")


if __name__ == "__main__":
    main()
