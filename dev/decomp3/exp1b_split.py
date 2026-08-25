"""Exp1b: linear side decomposition, CORRECTED (all quantities in the deployed
transformed space: xt = tf(x_ref), wt = tf(w_ref); transform is an exact
matmul invariant so MSEs are unchanged).

  mse_play   = ||xq@wq.T - xt@wt.T||^2        (ship)
  act_term   = ||(xq-xt)@wt.T||^2             (residual if weight were exact)
  w_term     = ||xt@(wq-wt).T||^2             (residual if act were exact)
  cross      = mse_play - act_term - w_term
  pp_Wexact  = pp of output xq@wt.T  -> weight-side pool = pp_Wexact - pp_play
  pp_Xexact  = pp of output xt@wq.T  -> act-side pool    = pp_Xexact - pp_play
"""
from __future__ import annotations

import json
import os
import sys

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


def tf(t):
    if mode == 1:
        return SOL._rot_blocks(t)
    return t


w_ref = hif4.dequantize_nvfp4(*W).float()
w_std_t = C.V.deq(C.V.quant_alg1(w_ref)).float()
wt = tf(w_ref / s)                      # exact weight in deployed space
wq = hif4.hif4_dequantize(cal["weight_params"]).float()

recs = []
for pair in TST:
    x_ref = hif4.dequantize_nvfp4(*pair)
    xf = x_ref.float()
    xt = tf(xf * s)                     # exact activation in deployed space
    ref = xt @ wt.T
    mse_std = ((tf(C.V.deq(C.V.quant_alg1(xf)) * s) @ tf(C.V.deq(
        C.V.quant_alg1(w_ref)) / s).T - ref) ** 2).mean().item()
    p = SOL.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
    xq = hif4.hif4_dequantize(p).float()
    d_act = (xq - xt) @ wt.T
    d_w = xt @ (wq - wt).T
    m = {
        "T": pair[0].shape[0], "mse_std": mse_std,
        "mse_play": ((xq @ wq.T - ref) ** 2).mean().item(),
        "mse_act_term": (d_act ** 2).mean().item(),
        "mse_w_term": (d_w ** 2).mean().item(),
        "mse_Wexact_out": (d_act ** 2).mean().item(),
        "mse_Xexact_out": (d_w ** 2).mean().item(),
    }
    m["mse_cross"] = m["mse_play"] - m["mse_act_term"] - m["mse_w_term"]
    m["pp_play"] = (mse_std - m["mse_play"]) / mse_std * 100.0
    m["pp_Wexact"] = (mse_std - m["mse_act_term"]) / mse_std * 100.0
    m["pp_Xexact"] = (mse_std - m["mse_w_term"]) / mse_std * 100.0
    m["relerr_act"] = m["mse_act_term"] / m["mse_play"]
    m["relerr_w"] = m["mse_w_term"] / m["mse_play"]
    recs.append(m)
    print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in m.items()},
          flush=True)

mean = {k: sum(r[k] for r in recs) / len(recs) for k in recs[0] if k != "T"}
print("MEAN", json.dumps({k: round(v, 6) for k, v in mean.items()}))
print(f"weight-side pool (pp_Xexact - pp_play) = {mean['pp_Xexact']-mean['pp_play']:+.3f}")
print(f"act-side    pool (pp_Wexact - pp_play) = {mean['pp_Wexact']-mean['pp_play']:+.3f}")

with open(os.path.join(C.HERE, "results_exp1b.json"), "w", encoding="utf-8") as fh:
    json.dump({"recs": recs, "mean": mean}, fh, indent=1)
print("DONE")
