"""Exp5 (+exp3-attn): attention-side anatomy on mini real test (v40).

Arms (full recalibration each):
  off   : QKS_MODE=off (pre-QKS baseline)
  ship  : QKS_MODE=pre (v40)
  orac  : QKS s fit on TEST q/k energies (perfect knowledge; guard bypassed)
Side decomposition on the ship calibration (exact swap per side):
  pools: qe / ke / ve / qke (each = pp(side exact) - pp(play))
plus exact-P oracle re-measure at v40: V lattice-refined against exact
attention probabilities (transpose trick: rows = dims, gw = sum_h P_h^T P_h).
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
_, mini = C.load_mini()
QH, KVH, DH = mini["q_num_heads"], mini["kv_num_heads"], mini["head_dim"]
CAL, TST = mini["calib"], mini["test"]
REP = QH // KVH


def eval_cal(cal, tag):
    recs = C.run_attn_tests(SOL, cal, TST, QH, KVH, DH)
    keys = ["pp_play", "pp_qe", "pp_ke", "pp_ve", "pp_qke"]
    mean = {k: round(sum(r[k] for r in recs) / len(recs), 3) for k in keys}
    print(f"[{tag}] mean:", json.dumps(mean), flush=True)
    for r in recs:
        print("   ", {k: round(v, 2) for k, v in r.items()
                      if k.startswith("pp") or k == "T"}, flush=True)
    return {"tag": tag, "mean": mean, "recs": [
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
        for r in recs]}


def dyn_qkv(cal, smp):
    pq = SOL.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], QH, DH,
                                     C.clone_state(cal["q_state"]))
    pk = SOL.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], KVH, DH,
                                     C.clone_state(cal["k_state"]))
    pv = SOL.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], KVH, DH,
                                     C.clone_state(cal["v_state"]))
    return (hif4.hif4_dequantize(pq), hif4.hif4_dequantize(pk),
            hif4.hif4_dequantize(pv), pv)


out = []
t0 = time.perf_counter()
SOL.QKS_MODE = "off"
torch.manual_seed(0)
cal = SOL.hif4_calibration_attention(CAL, QH, KVH, DH)
out.append(eval_cal(cal, "off"))

SOL.QKS_MODE = "pre"
torch.manual_seed(0)
cal = SOL.hif4_calibration_attention(CAL, QH, KVH, DH)
ship_dbg = dict(SOL.QKS_DEBUG)
cal_ship = cal
out.append(eval_cal(cal, "ship"))
print(f"t_attn_cals={time.perf_counter()-t0:.1f}s  qks_dbg={ship_dbg}", flush=True)

# ---- oracle QKS s from TEST energies ----
A = torch.zeros(KVH, DH)
B = torch.zeros(KVH, DH)
n = 0.0
for smp in TST:
    qd = hif4.dequantize_nvfp4(*smp["q"]).float()
    kd = hif4.dequantize_nvfp4(*smp["k"]).float()
    Tt = min(qd.shape[0], kd.shape[0])
    qe = (qd[:Tt].view(Tt, QH, DH) ** 2).view(Tt, KVH, REP, DH).sum(dim=(0, 2))
    ke = (kd[:Tt].view(Tt, KVH, DH) ** 2).sum(dim=0)
    A += qe
    B += ke
    n += Tt
A /= n
B /= n
ls = 0.25 * (B / A.clamp_min(1e-30)).log()
ls = ls - ls.mean(dim=1, keepdim=True)
ls = ls.clamp(-SOL.QKS_LOGS_CLIP, SOL.QKS_LOGS_CLIP)
s_orac = ls.exp().to(torch.bfloat16).float().contiguous()

_orig_maybe = SOL._qks_maybe
SOL._qks_maybe = lambda *a, **k: s_orac
torch.manual_seed(0)
cal = SOL.hif4_calibration_attention(CAL, QH, KVH, DH)
SOL._qks_maybe = _orig_maybe
out.append(eval_cal(cal, "orac_qks"))

s_ship = cal_ship["q_state"].get("qs")
rlr = (float((s_orac.log() - s_ship.float().log()).pow(2).mean().sqrt())
       if s_ship is not None else None)
print(f"rms log-ratio orac vs ship QKS s: {rlr:.4f}", flush=True)
out.append({"tag": "qks_rms_log_ratio", "value": round(rlr, 4)})

# ---- exact-P oracle: V lattice-refined against exact probabilities ----
recs_p = []
for smp in TST:
    q_ref = hif4.dequantize_nvfp4(*smp["q"])
    k_ref = hif4.dequantize_nvfp4(*smp["k"])
    v_ref = hif4.dequantize_nvfp4(*smp["v"])
    ref = hif4.attn_ref(q_ref, k_ref, v_ref, QH, KVH, DH)
    mse_std = ((hif4.attn_ref(C.V.deq(C.V.quant_alg1(q_ref.float())),
                              C.V.deq(C.V.quant_alg1(k_ref.float())),
                              C.V.deq(C.V.quant_alg1(v_ref.float())), QH, KVH, DH)
                - ref) ** 2).mean().item()
    q_play, k_play, v_play, pv = dyn_qkv(cal_ship, smp)
    mse_play = ((hif4.attn_ref(q_play, k_play, v_play, QH, KVH, DH)
                 - ref) ** 2).mean().item()
    # exact P per q-head
    T = q_ref.shape[0]
    qf = q_ref.float().view(T, QH, DH).transpose(0, 1)
    kf = k_ref.float().view(T, KVH, DH).transpose(0, 1).repeat_interleave(REP, 0)
    prob = torch.softmax(torch.bmm(qf, kf.transpose(1, 2)) / (DH ** 0.5), -1)
    # H_p per kv head = sum over group q-heads of P^T P  (T, T)
    PP = torch.bmm(prob.transpose(1, 2), prob)          # (QH, T, T)
    H_p = PP.view(KVH, REP, T, T).sum(dim=1)            # (KVH, T, T)
    unit = SOL._params_unit_flat(pv)                    # (T, KVH*DH)
    v4 = torch.round(v_play.float() / unit * 4.0)
    # transpose formulation: rows = (kv-head, dim) -> use per-head slices
    v_or = torch.empty_like(v_play.float())
    for h in range(KVH):
        sl = slice(h * DH, (h + 1) * DH)
        vh = v_play.float()[:, sl]                       # (T, DH)
        v4h = torch.round(vh / unit[:, sl] * 4.0).T.contiguous()   # (DH, T)
        dh_ = (0.25 * unit[:, sl]).T.contiguous()
        vhT = v_ref.float()[:, sl].T.contiguous()        # target (DH, T)
        Hh = H_p[h].contiguous()
        M = (v4h * dh_ - vhT) @ Hh                       # (DH, T)
        col2 = Hh.diagonal()
        neg2d = -2.0 * dh_
        d2col = (dh_ * dh_) * col2
        SOL._rounds_active(M, v4h, dh_, neg2d, d2col, Hh, 44 * 20)
        v_or[:, sl] = (v4h * dh_).T
    mse_p = ((hif4.attn_ref(q_play, k_play, v_or, QH, KVH, DH)
              - ref) ** 2).mean().item()
    recs_p.append({"T": T,
                   "pp_play": round((mse_std - mse_play) / mse_std * 100, 3),
                   "pp_exactP": round((mse_std - mse_p) / mse_std * 100, 3)})
    print("   exactP", recs_p[-1], flush=True)
out.append({"tag": "exactP", "recs": recs_p,
            "mean_gain": round(sum(r["pp_exactP"] - r["pp_play"]
                                   for r in recs_p) / len(recs_p), 3)})

with open(os.path.join(C.HERE, "results_exp5.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("DONE")
