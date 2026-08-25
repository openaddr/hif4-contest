"""Exp2c: fp32-gram pool extensions.
(a) T=10 tiny path (numpy rounds + deepen loop) with bf16 vs fp32 grams.
(b) variant value at SHIP sweep tiers with fp32 grams (no extra time):
    bf16 (ship) / fp32-gw-only / fp32-both, at ship tier and deep tier.
Output = play-MSE per call -> pp.
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
tf = (lambda t: SOL._rot_blocks(t)) if mode == 1 else (lambda t: t)
w_ref = hif4.dequantize_nvfp4(*W).float()
wt = tf(w_ref / s)
wq = hif4.hif4_dequantize(cal["weight_params"]).float()
gw32 = (wq.T @ wq)
gwf32 = (wt.T @ wq)
gw16, gwf16 = st["gw"].float(), st["gwf"].float()
u_act, order = st["u_act"], st["order"]

w_std_t = tf(C.V.deq(C.V.quant_alg1(w_ref)).float() / s)


def staged(pair):
    T_, C_ = pair[0].shape
    x = SOL.dequantize_nvfp4(pair[0], pair[1]).float()
    xs = x * s
    if mode == 1:
        xs = SOL._rot_blocks(xs)
    p = SOL._quantize_weighted(xs, torch.ones(1, C_, dtype=torch.float32))
    unit = SOL._params_unit_flat(p)
    ol = order.long() if order is not None else None
    if ol is not None:
        q = SOL._gptq_quantize_values(xs[:, ol], unit[:, ol], u_act.float())
        q0 = torch.empty_like(q)
        q0[:, ol] = q
        v_gptq = q0
    else:
        v_gptq = SOL._gptq_quantize_values(xs, unit, u_act.float())
    return xs, v_gptq, unit


recs = []
for pair in TST:
    T_, C_ = pair[0].shape
    xs, v_gptq, unit = staged(pair)
    x_ref = hif4.dequantize_nvfp4(*pair).float()
    ref_out = xs @ wt.T
    mse_std = ((tf(C.V.deq(C.V.quant_alg1(x_ref)).float() * s) @ w_std_t.T
                - ref_out) ** 2).mean().item()
    ship_sweeps = (44 if T_ <= 256 else 20 if T_ <= 512 else 8)
    arms = {}
    for tag, gw, gwf in (("bf16", gw16, gwf16),
                         ("fp32_gw", gw32, gwf16),
                         ("fp32", gw32, gwf32)):
        v_ship = SOL._refine_act_values(xs, v_gptq, unit, gw, gwf)
        arms[tag + "_ship"] = ((v_ship @ wq.T - ref_out) ** 2).mean().item()
        if T_ > 32:
            v_deep = SOL._refine_act_values(xs, v_gptq, unit, gw, gwf,
                                            sweep_override=ship_sweeps * 5)
            arms[tag + "_deep5x"] = ((v_deep @ wq.T - ref_out) ** 2).mean().item()
    rec = {"T": T_, "mse_std": mse_std, **arms}
    for k in list(arms):
        rec["pp_" + k] = (mse_std - arms[k]) / mse_std * 100.0
    recs.append(rec)
    print({k: (round(v, 7) if isinstance(v, float) else v) for k, v in rec.items()},
          flush=True)

mean = {k: sum(r[k] for r in recs) / len(recs) for k in recs[0]
        if isinstance(recs[0][k], (float, int))}
print("MEAN", json.dumps({k: round(v, 4) for k, v in mean.items()}))
with open(os.path.join(C.HERE, "results_exp2c.json"), "w", encoding="utf-8") as fh:
    json.dump({"recs": recs, "mean": mean}, fh, indent=1)
print("DONE")
