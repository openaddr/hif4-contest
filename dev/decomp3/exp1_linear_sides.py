"""Exp1: linear side decomposition on mini real test (v40).

Pools measured (end-to-end pp per case):
  weight pool = pp(w_exact) - pp(play)     [activation quantized, weight exact]
  act pool    = pp(x_exact) - pp(play)     [weight quantized, activation exact]
plus the additive MSE split (act term, weight term, cross) for reference.
Caches calibration to cache/ so later experiments reuse it.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

SOL = C.load_sol()
grp, _ = C.load_mini()
W, CAL, TST = grp["weight"], grp["calib_activation_list"], grp["test_activation_list"]

t0 = time.perf_counter()
torch.manual_seed(0)
cal = SOL.hif4_calibration_and_quantize_weight(*W, CAL)
t_cal = time.perf_counter() - t0
st = cal["activation_state"]
print(f"t_cal={t_cal:.1f}s mode={st['mode']} g={st['g']} tmax={st['tmax']} "
      f"grams={'yes' if st['gw'] is not None else 'no'} "
      f"smooth_acc={SOL.SMOOTH_DEBUG.get('accepted')} "
      f"j={SOL.SMOOTH_DEBUG.get('j_base')}->{SOL.SMOOTH_DEBUG.get('j_cand')} "
      f"s_logstd={float(st['s'].log().std()):.3f}", flush=True)

os.makedirs(os.path.join(C.HERE, "cache"), exist_ok=True)
torch.save(cal, os.path.join(C.HERE, "cache", "mini_linear_cal_v40.pt"))

# ---- side swap + additive split (needs transformed-space x for the split) ----
import hif4  # noqa: E402

w_ref = hif4.dequantize_nvfp4(*W)
w_play = hif4.hif4_dequantize(cal["weight_params"])
w_std_t = C.V.deq(C.V.quant_alg1(w_ref.float()))

recs = []
t0 = time.perf_counter()
for pair in TST:
    x_ref = hif4.dequantize_nvfp4(*pair)
    ref = hif4.linear_ref(x_ref, w_ref)
    mse_std = ((hif4.linear_ref(C.V.deq(C.V.quant_alg1(x_ref.float())), w_std_t)
                - ref) ** 2).mean().item()
    p = SOL.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
    x_play = hif4.hif4_dequantize(p)
    # exact tensors in float for additive split
    xf, wf = x_ref.float(), w_ref.float()
    xq, wq = x_play.float(), w_play.float()
    dw = (xq @ wq.T - xf @ wf.T)                       # total error tensor
    d_act = ((xq - xf) @ wf.T)                          # act-side term
    d_w = (xf @ (wq - wf).T)                            # weight-side term
    m = {
        "T": pair[0].shape[0], "mse_std": mse_std,
        "mse_play": (dw ** 2).mean().item(),
        "mse_act_term": (d_act ** 2).mean().item(),
        "mse_w_term": (d_w ** 2).mean().item(),
        "mse_cross": ((dw ** 2).mean() - (d_act ** 2).mean()
                      - (d_w ** 2).mean()).item(),
        "mse_w_exact": ((xq @ wf.T - xf @ wf.T) ** 2).mean().item(),
        "mse_x_exact": ((xf @ wq.T - xf @ wf.T) ** 2).mean().item(),
    }
    for k in ("play", "w_exact", "x_exact"):
        m["pp_" + k] = (mse_std - m["mse_" + k]) / mse_std * 100.0
    recs.append(m)
print(f"t_dyn={time.perf_counter()-t0:.1f}s", flush=True)

for r in recs:
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()},
          flush=True)

mean = {k: sum(r[k] for r in recs) / len(recs) for k in recs[0] if k != "T"}
print("MEAN", json.dumps({k: round(v, 5) for k, v in mean.items()}), flush=True)

out = {"t_cal": t_cal, "mode": st["mode"], "g": st["g"],
       "smooth_debug": dict(SOL.SMOOTH_DEBUG), "recs": recs, "mean": mean}
with open(os.path.join(C.HERE, "results_exp1.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("DONE")
