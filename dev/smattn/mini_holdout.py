"""Mini single-holdout for Q/K joint channel balancing (attention).

Splits (pipeline convention: fit = list[:-1], guard = list[-1]):
  A) fit calib[0:3], guard calib[3], eval calib[4]
  B) fit calib[0:4], guard calib[4], eval calib[3]
  T) shipped form: fit calib[0:4], guard calib[4], eval the 5 REAL tests
Arms: off, pre.  Report per-case pp deltas + guard debug.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "dev"))
import hif4  # noqa: E402
import variants as V  # noqa: E402

spec = importlib.util.spec_from_file_location("_qks_sol", os.path.join(HERE, "solution.py"))
SOL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SOL)

mini = torch.load(os.path.join(ROOT, "example", "mini_sample", "attn.pt"),
                  weights_only=True, map_location="cpu")[0]
QH, KVH, DH = mini["q_num_heads"], mini["kv_num_heads"], mini["head_dim"]
CAL, TST = mini["calib"], mini["test"]


def clone_state(st):
    if isinstance(st, torch.Tensor):
        return st.clone()
    if isinstance(st, dict):
        return {k: clone_state(v) for k, v in st.items()}
    return st


def run(arm, fit_list, eval_samples, tag):
    SOL.QKS_MODE = "off" if arm == "off" else "pre"
    SOL.QKS_DEBUG.clear()
    torch.manual_seed(0)
    t0 = time.perf_counter()
    cal = SOL.hif4_calibration_attention(fit_list, QH, KVH, DH)
    t_cal = time.perf_counter() - t0
    dbg = dict(SOL.QKS_DEBUG)
    rows = []
    for smp in eval_samples:
        q_ref = hif4.dequantize_nvfp4(*smp["q"])
        k_ref = hif4.dequantize_nvfp4(*smp["k"])
        v_ref = hif4.dequantize_nvfp4(*smp["v"])
        ref = hif4.attn_ref(q_ref, k_ref, v_ref, QH, KVH, DH)
        mse_std = ((hif4.attn_ref(V.deq(V.quant_alg1(q_ref.float())),
                                  V.deq(V.quant_alg1(k_ref.float())),
                                  V.deq(V.quant_alg1(v_ref.float())), QH, KVH, DH)
                    - ref) ** 2).mean().item()
        pq = SOL.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], QH, DH,
                                         clone_state(cal["q_state"]))
        pk = SOL.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], KVH, DH,
                                         clone_state(cal["k_state"]))
        pv = SOL.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], KVH, DH,
                                         clone_state(cal["v_state"]))
        out = hif4.attn_ref(hif4.hif4_dequantize(pq), hif4.hif4_dequantize(pk),
                            hif4.hif4_dequantize(pv), QH, KVH, DH)
        mse_play = ((out - ref) ** 2).mean().item()
        rows.append((mse_std - mse_play) / mse_std * 100.0)
    rec = {"tag": tag, "arm": arm, "scores": [round(r, 3) for r in rows],
           "mean": round(sum(rows) / len(rows), 3), "t_cal": round(t_cal, 2),
           "acc": dbg.get("accepted"), "stab": dbg.get("stab"),
           "j": [round(dbg.get("j_base") or 0, 6), round(dbg.get("j_cand") or 0, 6)]}
    print(json.dumps(rec), flush=True)
    return rec


SPLITS = [
    ("A", CAL[:4], [CAL[4]]),
    ("B", CAL[:5], [CAL[3]]),
    ("T", CAL[:5], TST),
]

out = []
for tag, fit_list, evals in SPLITS:
    base = run("off", fit_list, evals, tag)
    pre = run("pre", fit_list, evals, tag)
    print(f"[{tag}] delta = {pre['mean'] - base['mean']:+.3f}pp/case", flush=True)
    out.extend([base, pre])

with open(os.path.join(HERE, "mini_results.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("DONE")
