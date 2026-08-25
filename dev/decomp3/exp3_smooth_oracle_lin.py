"""Exp3: linear smoothing oracle gap on mini real test.

Arms (full recalibration each, guard bypassed for oracle arms):
  ship      : v40 ff_bal (fit calib[:-1] 160 rows, guarded)
  allcal    : ff_bal energies fit on ALL calib rows (incl. guard sample)
  orac_en   : energies fit on the 5 TEST samples (perfect test knowledge)
  orac_icm  : orac_en + deploy-aware coordinate descent (_icm_search) on TEST rows
  s1        : baseline alpha search (s=1 de facto)
Also: rms log-ratio of each arm's s vs ship s (msel noise floor ~0.05).
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

# precompute fit-row pools (mirror _freeform_s row selection)


def rows_from(samples, budget=160):
    rows = []
    per = max(1, budget // len(samples))
    for a in samples:
        T = a.shape[0]
        if T > per:
            stride = T // per
            idx = torch.arange(0, T, stride)[:per]
        else:
            idx = torch.arange(T)
        rows.append(a[idx])
        budget -= idx.shape[0]
        if budget <= 0:
            break
    return torch.cat(rows, dim=0)[:160].contiguous()


cal_raw = [hif4.dequantize_nvfp4(*p).float() for p in CAL]
tst_raw = [hif4.dequantize_nvfp4(*p).float() for p in TST]
xf_cal = rows_from(cal_raw[:-1])
xf_allcal = rows_from(cal_raw)
xf_test = rows_from(tst_raw)

w_full = hif4.dequantize_nvfp4(*W).float()
gw_col = (w_full * w_full).sum(dim=0) + 1e-30


def bal_from(xf):
    gx = (xf * xf).sum(dim=0) + 1e-30
    ls = 0.25 * (gw_col / gx).clamp_min(1e-30).log()
    ls = ls - ls.mean()
    ls = ls.clamp(-SOL.SMOOTH_LOGS_CLIP, SOL.SMOOTH_LOGS_CLIP)
    return ls.exp()


s_orac = bal_from(xf_test)
s_allcal = bal_from(xf_allcal)


def eval_cal(cal, tag, s_ref=None):
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
    s = st["s"]
    rlr = (float((s.log() - s_ref.log()).pow(2).mean().sqrt())
           if s_ref is not None else None)
    rec = {"arm": tag, "mean": round(sum(scores) / len(scores), 3),
           "scores": [round(v, 3) for v in scores], "mode": st["mode"],
           "g": st["g"], "rms_log_ratio_vs_ship": round(rlr, 4) if rlr else None,
           "s_logstd": round(float(s.log().std()), 4)}
    print(json.dumps(rec), flush=True)
    return rec


out = []
# ---- ship ----
torch.manual_seed(0)
SOL.SMOOTH_MODE = "ff_bal"
SOL._freeform_s.__wrapped__ if False else None
cal = SOL.hif4_calibration_and_quantize_weight(*W, CAL)
s_ship = cal["activation_state"]["s"].clone()
out.append(eval_cal(cal, "ship"))
_orig_freeform = SOL._freeform_s

# ---- oracle arms via monkey-patch (guard bypassed: return s directly) ----


def make_patch(s_cand):
    def _patch(acts_raw, w, s_base, logm):
        return s_cand.contiguous()
    return _patch


for tag, s_cand in (("orac_en", s_orac), ("allcal", s_allcal)):
    SOL._freeform_s = make_patch(s_cand)
    torch.manual_seed(0)
    cal = SOL.hif4_calibration_and_quantize_weight(*W, CAL)
    out.append(eval_cal(cal, tag, s_ref=s_ship))

# orac_icm: init at s_orac, coordinate descent on test rows (deploy-aware)
gen = torch.Generator().manual_seed(7717 + W[0].shape[0])
wsub = w_full[torch.randperm(W[0].shape[0], generator=gen)[:SOL.SMOOTH_W_ROWS]].contiguous()
gx_t = (xf_test * xf_test).sum(dim=0) + 1e-30
s_icm = SOL._icm_search(xf_test, wsub, s_orac, gw_col, gx_t)
s_icm = s_icm / torch.exp(s_icm.log().mean())
SOL._freeform_s = make_patch(s_icm)
torch.manual_seed(0)
cal = SOL.hif4_calibration_and_quantize_weight(*W, CAL)
out.append(eval_cal(cal, "orac_icm", s_ref=s_ship))

# ---- s=1 baseline ----
SOL._freeform_s = _orig_freeform
SOL.SMOOTH_MODE = "base"
torch.manual_seed(0)
cal = SOL.hif4_calibration_and_quantize_weight(*W, CAL)
out.append(eval_cal(cal, "s1_base", s_ref=s_ship))
SOL.SMOOTH_MODE = "ff_bal"

with open(os.path.join(C.HERE, "results_exp3_lin.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("DONE")
