"""Exp4: decision-quality pools + attention refine T-gate extension.

Linear:
  mode0 : force mode=0 (patch _rot_blocks -> identity) full recal; eval tests
          -> mode-decision pool = pp(ship mode1) - pp(mode0)
Attention:
  rot0  : force rot=0 (patch _make_R -> None) full recal
  Tgate : ship states, ATTN_REFINE_MAX_T 128 -> 4096 (dynamic-only change)
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
grp, mini = C.load_mini()
W, CAL, TST = grp["weight"], grp["calib_activation_list"], grp["test_activation_list"]


def eval_lin(cal, tag):
    w_ref = hif4.dequantize_nvfp4(*W)
    w_std = C.V.deq(C.V.quant_alg1(w_ref.float()))
    w_play = hif4.hif4_dequantize(cal["weight_params"])
    st = cal["activation_state"]
    scores = []
    for pair in TST:
        x_ref = hif4.dequantize_nvfp4(*pair)
        ref = hif4.linear_ref(x_ref, w_ref)
        mse_std = ((hif4.linear_ref(C.V.deq(C.V.quant_alg1(x_ref.float())), w_std)
                    - ref) ** 2).mean().item()
        p = SOL.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
        mse_play = ((hif4.linear_ref(hif4.hif4_dequantize(p), w_play)
                     - ref) ** 2).mean().item()
        scores.append((mse_std - mse_play) / mse_std * 100.0)
    rec = {"tag": tag, "mean": round(sum(scores) / len(scores), 3),
           "scores": [round(v, 2) for v in scores], "mode": st["mode"],
           "g": st["g"], "smooth_acc": SOL.SMOOTH_DEBUG.get("accepted")}
    print(json.dumps(rec), flush=True)
    return rec


out = []
# ---- linear mode0 (rotation disabled everywhere) ----
_orig_rot = SOL._rot_blocks
SOL._rot_blocks = lambda x: x
torch.manual_seed(0)
cal = SOL.hif4_calibration_and_quantize_weight(*W, CAL)
out.append(eval_lin(cal, "lin_mode0"))
SOL._rot_blocks = _orig_rot

# ---- attention rot0 ----
QH, KVH, DH = mini["q_num_heads"], mini["kv_num_heads"], mini["head_dim"]
ACAL, ATST = mini["calib"], mini["test"]

SOL.QKS_MODE = "pre"
_orig_makeR = SOL._make_R
SOL._make_R = lambda dh: None
torch.manual_seed(0)
cal = SOL.hif4_calibration_attention(ACAL, QH, KVH, DH)
SOL._make_R = _orig_makeR
recs = C.run_attn_tests(SOL, cal, ATST, QH, KVH, DH)
mean0 = round(sum(r["pp_play"] for r in recs) / len(recs), 3)
out.append({"tag": "attn_rot0", "mean": mean0,
            "scores": [round(r["pp_play"], 2) for r in recs],
            "dbg": {k: v for k, v in SOL.QKS_DEBUG.items()}})
print(json.dumps(out[-1]), flush=True)

# ---- attention T-gate extension (ship states, dynamic only) ----
torch.manual_seed(0)
SOL._make_R = _orig_makeR
torch.manual_seed(0)
cal = SOL.hif4_calibration_attention(ACAL, QH, KVH, DH)
SOL.ATTN_REFINE_MAX_T = 4096
recs = C.run_attn_tests(SOL, cal, ATST, QH, KVH, DH)
SOL.ATTN_REFINE_MAX_T = 128
meanT = round(sum(r["pp_play"] for r in recs) / len(recs), 3)
out.append({"tag": "attn_Tgate4096", "mean": meanT,
            "scores": [round(r["pp_play"], 2) for r in recs]})
print(json.dumps(out[-1]), flush=True)

with open(os.path.join(C.HERE, "results_exp4.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("DONE")
