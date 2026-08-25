"""Exp5b: attention single-side pools, transform-preserving.

The exp5 single-side swaps (qe/ke) broke the QKS scaling + rotation joint
invariance (one side transformed, the other not -> logits garbage).  Redo
with the exact side mapped into the deployed space:
  qe : q_exact = rot(q_ref * s)   [s = ship QKS scale, rot = ship mode]
  ke : k_exact = rot(k_ref / s)
Also record ship attention state flags (rot, gq, rf) and per-T refine gate.
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
_, mini = C.load_mini()
QH, KVH, DH = mini["q_num_heads"], mini["kv_num_heads"], mini["head_dim"]
CAL, TST = mini["calib"], mini["test"]
REP = QH // KVH

SOL.QKS_MODE = "pre"
torch.manual_seed(0)
cal = SOL.hif4_calibration_attention(CAL, QH, KVH, DH)
qs = cal["q_state"]
ks = cal["k_state"]
print("q_state flags:", {k: (v.shape if isinstance(v, torch.Tensor) else v)
                         for k, v in qs.items()}, flush=True)
s = qs.get("qs")
s = s.float() if s is not None else None
rot = qs.get("rot")
R = SOL._make_R(DH) if rot == 1 else None

recs = []
for smp in TST:
    q_ref = hif4.dequantize_nvfp4(*smp["q"])
    k_ref = hif4.dequantize_nvfp4(*smp["k"])
    v_ref = hif4.dequantize_nvfp4(*smp["v"])
    ref = hif4.attn_ref(q_ref, k_ref, v_ref, QH, KVH, DH)
    mse_std = ((hif4.attn_ref(C.V.deq(C.V.quant_alg1(q_ref.float())),
                              C.V.deq(C.V.quant_alg1(k_ref.float())),
                              C.V.deq(C.V.quant_alg1(v_ref.float())), QH, KVH, DH)
                - ref) ** 2).mean().item()
    pq = SOL.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], QH, DH,
                                     C.clone_state(qs))
    pk = SOL.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], KVH, DH,
                                     C.clone_state(ks))
    pv = SOL.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], KVH, DH,
                                     C.clone_state(cal["v_state"]))
    q_play, k_play, v_play = (hif4.hif4_dequantize(pq),
                              hif4.hif4_dequantize(pk),
                              hif4.hif4_dequantize(pv))
    # exact sides in deployed space
    T = q_ref.shape[0]
    qe = q_ref.float()
    ke = k_ref.float()
    if s is not None:
        qe = SOL._qks_apply_q(qe, s, QH, KVH, DH, inv=False)
        ke = SOL._qks_apply_q(ke, s, QH, KVH, DH, inv=True)
    if R is not None:
        qe = (qe.view(T, QH, DH) @ R).reshape(T, -1)
        ke = (ke.view(T, KVH, DH) @ R).reshape(T, -1)
    outs = {
        "mse_play": hif4.attn_ref(q_play, k_play, v_play, QH, KVH, DH),
        "mse_qe": hif4.attn_ref(qe, k_play, v_play, QH, KVH, DH),
        "mse_ke": hif4.attn_ref(q_play, ke, v_play, QH, KVH, DH),
        "mse_ve": hif4.attn_ref(q_play, k_play, v_ref.float(), QH, KVH, DH),
        "mse_qke": hif4.attn_ref(qe, ke, v_play, QH, KVH, DH),
    }
    rec = {"T": T, "mse_std": mse_std}
    for k, o in outs.items():
        rec[k] = ((o - ref) ** 2).mean().item()
        rec["pp_" + k[4:]] = (mse_std - rec[k]) / mse_std * 100.0
    recs.append(rec)
    print({k: round(v, 2) for k, v in rec.items()
           if k.startswith("pp") or k == "T"}, flush=True)

mean = {k: round(sum(r[k] for r in recs) / len(recs), 3)
        for k in recs[0] if k.startswith("pp_") or k == "mse_play"}
print("MEAN", json.dumps(mean))
print(f"q-side pool = pp_qe - pp_play = {mean['pp_qe']-mean['pp_play']:+.2f}")
print(f"k-side pool = pp_ke - pp_play = {mean['pp_ke']-mean['pp_play']:+.2f}")
print(f"v-side pool = pp_ve - pp_play = {mean['pp_ve']-mean['pp_play']:+.2f}")
print(f"qk-side pool = pp_qke - pp_play = {mean['pp_qke']-mean['pp_play']:+.2f}")
with open(os.path.join(C.HERE, "results_exp5b.json"), "w", encoding="utf-8") as fh:
    json.dump({"recs": recs, "mean": mean,
               "state": {"rot": rot, "gq": qs.get("gq"),
                         "rf_q": qs.get("rf"), "rf_k": ks.get("rf")}},
              fh, indent=1)
print("DONE")
