"""Exp2: activation-side internals on mini real test (v40).

Stage ladder per dynamic call (output MSE vs exact ref, pp per case):
  S0 table   : staged table quantizer only (6-cand sf, exact-refined ranking)
  S1 gptq    : + activation GPTQ (u_act, act-ordered) [state g==1]
  S2 ship    : + lattice refinement (ship tiers) == play
  S3 deep    : refinement continued to convergence (sweep x8) -- oracle
Also: rows with improving flips left after ship tier; grid-lock distance
distribution (|xt - nearest legal grid point| / grid step, in step units),
edge-clamp fraction (|xt|/unit > 7).
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import hif4  # noqa: E402

SOL = C.load_sol()
grp, _ = C.load_mini()
W, CAL, TST = grp["weight"], grp["calib_activation_list"], grp["test_activation_list"]
cal = torch.load(os.path.join(C.HERE, "cache", "mini_linear_cal_v40.pt"),
                 weights_only=True)
st = cal["activation_state"]
s = st["s"].float()
mode = st["mode"]
assert st["g"] == 1 and st["gw"] is not None


def tf(t):
    return SOL._rot_blocks(t) if mode == 1 else t


w_ref = hif4.dequantize_nvfp4(*W).float()
wt = tf(w_ref / s)
wq = hif4.hif4_dequantize(cal["weight_params"]).float()
ref_w = wt.T
gw, gwf = st["gw"].float(), st["gwf"].float()
u_act, order = st["u_act"], st["order"]

recs = []
for pair in TST:
    t0 = time.perf_counter()
    T_, C_ = pair[0].shape
    x = SOL.dequantize_nvfp4(pair[0], pair[1]).float()
    xs = x * s
    if mode == 1:
        xs = SOL._rot_blocks(xs)
    p = SOL._quantize_weighted(xs, torch.ones(1, C_, dtype=torch.float32))
    unit = SOL._params_unit_flat(p)
    v_table = SOL._deq_params(p)
    # GPTQ stage (mirror ship dynamic path incl. act-order)
    ol = order.long() if order is not None else None
    if ol is not None:
        q = SOL._gptq_quantize_values(xs[:, ol], unit[:, ol], u_act.float())
        q0 = torch.empty_like(q)
        q0[:, ol] = q
        v_gptq = q0
    else:
        v_gptq = SOL._gptq_quantize_values(xs, unit, u_act.float())
    # ship refinement
    v_ship = SOL._refine_act_values(xs, v_gptq, unit, gw, gwf)
    # deep refinement (8x ship sweeps) from the SAME gptq start
    T = v_gptq.shape[0]
    deep_sweeps = (44 if T <= 256 else 20 if T <= 512 else 8) * 8
    v_deep = SOL._refine_act_values(xs, v_gptq, unit, gw, gwf,
                                    sweep_override=deep_sweeps)
    # convergence probe: improving flips remaining on ship result
    v4 = torch.round(v_ship / unit * 4.0)
    d = 0.25 * unit
    col2 = gw.diagonal()
    M = (v4 * d) @ gw - xs @ gwf
    gflip, _ = SOL._flip_sel(d, M, col2, v4)
    rows_left = int((gflip < 0).any(dim=1).sum().item())
    # output MSE per stage vs the EXACT output; std = alg1 both sides
    ref_out = xs @ wt.T
    x_std_t = tf(C.V.deq(C.V.quant_alg1(x)).float() * s)
    w_std_t = tf(C.V.deq(C.V.quant_alg1(w_ref)).float() / s)
    mse_std = ((x_std_t @ w_std_t.T - ref_out) ** 2).mean().item()
    mses = {
        "S0_table": ((v_table @ wq.T - ref_out) ** 2).mean().item(),
        "S1_gptq": ((v_gptq @ wq.T - ref_out) ** 2).mean().item(),
        "S2_ship": ((v_ship @ wq.T - ref_out) ** 2).mean().item(),
        "S3_deep": ((v_deep @ wq.T - ref_out) ** 2).mean().item(),
    }
    # grid-lock: distance of exact values to the nearest legal grid point
    step = 0.25 * unit
    nearest = torch.round(xs / step) * step
    near_legal = torch.round(torch.clamp(xs / unit, -7.0, 7.0) / 0.25) * 0.25 * unit
    gd = (xs - near_legal).abs() / step          # in grid steps
    edge = (xs.abs() / unit > 7.0).float().mean().item()
    rec = {"T": T_, "mse_std": mse_std, **mses,
           "rows_left_flips": rows_left, "rows_total": T_,
           "edge_frac": edge,
           "gd_mean": float(gd.mean()), "gd_p50": float(gd.quantile(0.5)),
           "gd_p90": float(gd.quantile(0.9)), "gd_rms": float((gd ** 2).mean().sqrt()),
           "rt_rms_theory": 0.2887,
           "t": round(time.perf_counter() - t0, 1)}
    for k in list(mses):
        rec["pp_" + k] = (mse_std - mses[k]) / mse_std * 100.0
    recs.append(rec)
    print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rec.items()},
          flush=True)

mean = {k: sum(r[k] for r in recs) / len(recs) for k in recs[0]
        if isinstance(recs[0][k], (float, int))}
print("MEAN", json.dumps({k: round(v, 5) for k, v in mean.items()}))
with open(os.path.join(C.HERE, "results_exp2.json"), "w", encoding="utf-8") as fh:
    json.dump({"recs": recs, "mean": mean}, fh, indent=1)
print("DONE")
