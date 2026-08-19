"""Quick check: quant_v2 (amax/7-anchored search) vs norm7 / usearch_lv / v1."""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hif4  # noqa: E402
import variants as V  # noqa: E402


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V1 = load_mod(os.path.join(os.path.dirname(__file__), "..", "ref", "v1_solution.py"), "v1sol")
root = os.path.join(os.path.dirname(__file__), "..", "example", "mini_sample")

lin = torch.load(os.path.join(root, "linear.pt"), weights_only=True, map_location="cpu")[0]
w_ref = hif4.dequantize_nvfp4(*lin["weight"]).float()
x = hif4.dequantize_nvfp4(*lin["test_activation_list"][-1]).float()
ref_out = x @ w_ref.T

wq = {
    "norm7": V.deq(V.quant_norm7(w_ref)),
    "usearch_lv": V.deq(V.quant_usearch_lv(w_ref)),
    "v2_plain": V.deq(V.quant_v2(w_ref)),
}
cal = V1.hif4_calibration_and_quantize_weight(*lin["weight"], lin["calib_activation_list"])
wq["v1"] = hif4.hif4_dequantize(cal["weight_params"])

print("=== Linear output MSE (lower better) ===")
base = ((wq["norm7"] - w_ref) ** 2).mean().item()
for k, v in wq.items():
    m = ((v - w_ref) ** 2).mean().item()
    mo = ((x @ v.T - ref_out) ** 2).mean().item()
    mo7 = ((x @ wq["norm7"].T - ref_out) ** 2).mean().item()
    print(f"{k:<11} elemMSE={m:.3e}  outMSE={mo:.3e}  score_vs_norm7={(mo7 - mo) / mo7 * 100:+6.1f}%")

att = torch.load(os.path.join(root, "attn.pt"), weights_only=True, map_location="cpu")[0]
qh, kvh, dh = att["q_num_heads"], att["kv_num_heads"], att["head_dim"]
smp = att["test"][-1]
q_ref = hif4.dequantize_nvfp4(*smp["q"]).float()
k_ref = hif4.dequantize_nvfp4(*smp["k"]).float()
v_ref = hif4.dequantize_nvfp4(*smp["v"]).float()
ref = hif4.attn_ref(q_ref, k_ref, v_ref, qh, kvh, dh)


def out_of(fns):
    qs = [V.deq(f(t)) for f, t in zip(fns, (q_ref, k_ref, v_ref))]
    return hif4.attn_ref(*qs, qh, kvh, dh)


o_norm7 = out_of([V.quant_norm7] * 3)
o_v2 = out_of([V.quant_v2] * 3)
m7 = ((o_norm7 - ref) ** 2).mean().item()
m2 = ((o_v2 - ref) ** 2).mean().item()
print("\n=== Attention output MSE ===")
print(f"norm7   {m7:.3e}")
print(f"v2plain {m2:.3e}   score_vs_norm7={(m7 - m2) / m7 * 100:+.1f}%")
