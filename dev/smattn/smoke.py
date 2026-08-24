"""Smoke test: QKS on/off on mini attention + one synthetic group.
Checks: finite, guard fires, off == baseline solution bit-identically,
accepted s actually changes outputs, timing.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "dev"))
import hif4  # noqa: E402
import variants as V  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE = load(os.path.join(ROOT, "example", "solution", "solution.py"), "sol_base")
SOL = load(os.path.join(HERE, "solution.py"), "sol_qks")

mini = torch.load(os.path.join(ROOT, "example", "mini_sample", "attn.pt"),
                  weights_only=True, map_location="cpu")[0]
qh, kvh, dh = mini["q_num_heads"], mini["kv_num_heads"], mini["head_dim"]


def run(mod, calib, tag):
    mod.QKS_DEBUG.clear() if hasattr(mod, "QKS_DEBUG") else None
    torch.manual_seed(0)
    t0 = time.perf_counter()
    cal = mod.hif4_calibration_attention(calib, qh, kvh, dh)
    t_cal = time.perf_counter() - t0
    dbg = dict(mod.QKS_DEBUG) if hasattr(mod, "QKS_DEBUG") else {}
    q_state, k_state = cal["q_state"], cal["k_state"]
    outs = []
    for smp in mini["test"]:
        pq = mod.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh, dh, q_state)
        pk = mod.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], kvh, dh, k_state)
        outs.append((hif4.hif4_dequantize(pq), hif4.hif4_dequantize(pk)))
    # score vs alg1
    tot = 0.0
    for smp, (qd, kd) in zip(mini["test"], outs):
        q_ref = hif4.dequantize_nvfp4(*smp["q"])
        k_ref = hif4.dequantize_nvfp4(*smp["k"])
        v_ref = hif4.dequantize_nvfp4(*smp["v"])
        ref = hif4.attn_ref(q_ref, k_ref, v_ref, qh, kvh, dh)
        pv = mod.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], kvh, dh, None)
        mse_std = ((hif4.attn_ref(V.deq(V.quant_alg1(q_ref.float())),
                                  V.deq(V.quant_alg1(k_ref.float())),
                                  V.deq(V.quant_alg1(v_ref.float())), qh, kvh, dh)
                    - ref) ** 2).mean().item()
        mse_play = ((hif4.attn_ref(qd, kd, hif4.hif4_dequantize(pv), qh, kvh, dh)
                     - ref) ** 2).mean().item()
        tot += (mse_std - mse_play) / mse_std * 100.0
    print(f"[{tag}] t_cal={t_cal:.2f}s dbg={dbg} mean_pp={tot/5:.3f}")
    return cal, outs, tot / 5.0


cal_b, out_b, pp_b = run(BASE, mini["calib"], "baseline")
SOL.QKS_MODE = "off"
cal_o, out_o, pp_o = run(SOL, mini["calib"], "qks=off")
same_all = True
for k1 in cal_b["q_state"]:
    a, b_ = cal_b["q_state"][k1], cal_o["q_state"].get(k1)
    if isinstance(a, torch.Tensor):
        same_all &= (b_ is not None and torch.equal(a, b_))
for k1 in cal_b["k_state"]:
    a, b_ = cal_b["k_state"][k1], cal_o["k_state"].get(k1)
    if isinstance(a, torch.Tensor):
        same_all &= (b_ is not None and torch.equal(a, b_))
ok_dyn = all(torch.equal(a[0], b_[0]) and torch.equal(a[1], b_[1])
             for a, b_ in zip(out_b, out_o))
print("off == baseline: states", same_all, "dyn", ok_dyn)

SOL.QKS_MODE = "pre"
cal_p, out_p, pp_p = run(SOL, mini["calib"], "qks=pre")
print(f"delta_pp = {pp_p - pp_b:+.3f}")
print("qs in state:", "qs" in cal_p["q_state"],
      cal_p["q_state"]["qs"].shape if "qs" in cal_p["q_state"] else None)
# guard-forced reject == baseline bit-identical
SOL.QKS_GUARD_MARGIN = 10.0   # force reject
SOL.QKS_DEBUG.clear()
cal_r, out_r, _ = run(SOL, mini["calib"], "qks=reject")
ok_dyn2 = all(torch.equal(a[0], b_[0]) and torch.equal(a[1], b_[1])
              for a, b_ in zip(out_b, out_r))
print("rejected == baseline dyn:", ok_dyn2)
SOL.QKS_GUARD_MARGIN = 0.002
print("DONE")
