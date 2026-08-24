"""Mini single-holdout measurement for free-form smoothing.

Splits (pipeline convention: fit = list[:-1], guard = list[-1]):
  A) fit {10,128,512} guard calib[3](T=1024), eval calib[4](T=1024)
  B) fit {10,128,512,1024} guard calib[4](T=1024), eval calib[3](T=1024)
  T) shipped form: fit first 4, guard calib[4], eval the 5 REAL test samples
Arms: base, ff_icm, ff_bal, mag_scan.  Report per-case pp deltas.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "dev"))
import hif4  # noqa: E402
import variants as V  # noqa: E402
import exp_smooth as E  # noqa: E402

SOL = E.load_sol()

mini = torch.load(os.path.join(ROOT, "example", "mini_sample", "linear.pt"),
                  weights_only=True, map_location="cpu")[0]
W, CAL, TST = mini["weight"], mini["calib_activation_list"], mini["test_activation_list"]


def run(arm, fit_list, eval_pairs, tag):
    SOL.SMOOTH_MODE = "base" if arm == "base" else arm
    SOL.SMOOTH_GUARD = True
    SOL.SMOOTH_DEBUG.clear()
    torch.manual_seed(0)
    cal = SOL.hif4_calibration_and_quantize_weight(*W, fit_list)
    w_ref = hif4.dequantize_nvfp4(*W)
    w_std = V.deq(V.quant_alg1(w_ref.float()))
    w_play = hif4.hif4_dequantize(cal["weight_params"])
    rows = []
    for i, pair in enumerate(eval_pairs):
        x_ref = hif4.dequantize_nvfp4(*pair)
        ref = hif4.linear_ref(x_ref, w_ref)
        x_std = V.deq(V.quant_alg1(x_ref.float()))
        mse_std = ((hif4.linear_ref(x_std, w_std) - ref) ** 2).mean().item()
        p = SOL.hif4_dynamic_quantize_activation(pair[0], pair[1],
                                                 cal["activation_state"])
        mse_play = ((hif4.linear_ref(hif4.hif4_dequantize(p), w_play) - ref) ** 2).mean().item()
        rows.append((mse_std - mse_play) / mse_std * 100.0)
    rec = {"tag": tag, "arm": arm, "scores": [round(r, 3) for r in rows],
           "mean": round(sum(rows) / len(rows), 3),
           "acc": SOL.SMOOTH_DEBUG.get("accepted"),
           "j": [SOL.SMOOTH_DEBUG.get("j_base"), SOL.SMOOTH_DEBUG.get("j_cand")],
           "mode": cal["activation_state"]["mode"]}
    print(json.dumps(rec), flush=True)
    return rec


SPLITS = [
    ("A", CAL[:4], [CAL[4]]),          # fit 3, guard calib[3], eval calib[4]
    ("B", CAL[:5], [CAL[3]]),          # fit 4, guard calib[4], eval calib[3]
    ("T", CAL[:5], TST),               # shipped form, real test
]

out = []
for tag, fit_list, evals in SPLITS:
    for arm in ("base", "ff_icm", "ff_bal", "mag_scan"):
        out.append(run(arm, fit_list, evals, tag))

with open(os.path.join(HERE, "mini_results.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("DONE")
