"""Diagnosis: compare v1 against baseline variants and near-oracle quantizers,
on element MSE and TRUE output MSE, for both linear and attention paths.
Also stress-test on synthetic distributions.
"""
from __future__ import annotations

import importlib.util
import math
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

# ------------------- Linear diagnosis -------------------
lin = torch.load(os.path.join(root, "linear.pt"), weights_only=True, map_location="cpu")[0]
w_ref = hif4.dequantize_nvfp4(*lin["weight"]).float()
acts = [(hif4.dequantize_nvfp4(*p)).float() for p in lin["test_activation_list"]]

cal = V1.hif4_calibration_and_quantize_weight(*lin["weight"], lin["calib_activation_list"])
w_v1 = hif4.hif4_dequantize(cal["weight_params"])

w_variants = {
    "greedy": V.deq(V.quant_greedy(w_ref)),
    "norm7": V.deq(V.quant_norm7(w_ref)),
    "usearch": V.deq(V.quant_usearch(w_ref)),
    "usearch_lv": V.deq(V.quant_usearch_lv(w_ref)),
}

print("=== Linear: weight-side element MSE ===")
base = ((w_variants["greedy"] - w_ref) ** 2).mean().item()
print(f"greedy     {base:.6e}  (x1.00)")
for k, v in w_variants.items():
    if k != "greedy":
        print(f"{k:<10} {((v - w_ref) ** 2).mean().item():.6e}  (x{base / ((v - w_ref) ** 2).mean().item():.3f})")
print(f"v1         {((w_v1 - w_ref) ** 2).mean().item():.6e}  (x{base / ((w_v1 - w_ref) ** 2).mean().item():.3f})")

print("\n=== Linear: true output MSE (largest test act) ===")
x = acts[-1]
ref_out = x @ w_ref.T
rows = []
for k, wq in [("v1", w_v1)] + list(w_variants.items()):
    mse = ((x @ wq.T - ref_out) ** 2).mean().item()
    rows.append((k, mse))
base = dict(rows)["greedy"]
for k, mse in rows:
    print(f"{k:<10} {mse:.6e}  (x{base / mse:.3f})  score_vs_greedy={1 - mse / base:+.3f}")

# score if the judge's baseline were each variant
print("\n=== hypothetical online score of v1 (weight side only) ===")
for k, mse in rows[1:]:
    v1m = dict(rows)["v1"]
    print(f"if judge baseline == {k:<10}: v1 score = {(mse - v1m) / mse * 100:+.1f}%")

# ------------------- Attention diagnosis -------------------
att = torch.load(os.path.join(root, "attn.pt"), weights_only=True, map_location="cpu")[0]
qh, kvh, dh = att["q_num_heads"], att["kv_num_heads"], att["head_dim"]


def attn_out(q, k, v):
    return hif4.attn_ref(q.bfloat16().float(), k.bfloat16().float(), v.bfloat16().float(), qh, kvh, dh)


smp = att["test"][-1]
q_ref = hif4.dequantize_nvfp4(*smp["q"]).float()
k_ref = hif4.dequantize_nvfp4(*smp["k"]).float()
v_ref = hif4.dequantize_nvfp4(*smp["v"]).float()
ref = hif4.attn_ref(q_ref, k_ref, v_ref, qh, kvh, dh)

acal = V1.hif4_calibration_attention(att["calib"], qh, kvh, dh)
pq = V1.hif4_dynamic_quantize_q(*smp["q"], qh, dh, acal["q_state"])
pk = V1.hif4_dynamic_quantize_k(*smp["k"], kvh, dh, acal["k_state"])
pv = V1.hif4_dynamic_quantize_v(*smp["v"], kvh, dh, acal["v_state"])
out_v1 = hif4.attn_ref(hif4.hif4_dequantize(pq), hif4.hif4_dequantize(pk), hif4.hif4_dequantize(pv), qh, kvh, dh)
mse_v1 = ((out_v1 - ref) ** 2).mean().item()

outs = {}
for k, fn in [("greedy", V.quant_greedy), ("norm7", V.quant_norm7),
              ("usearch", V.quant_usearch), ("usearch_lv", V.quant_usearch_lv)]:
    qq, kk, vv = V.deq(fn(q_ref)), V.deq(fn(k_ref)), V.deq(fn(v_ref))
    outs[k] = hif4.attn_ref(qq, kk, vv, qh, kvh, dh)

print("\n=== Attention: true output MSE (last test) ===")
base = ((outs["greedy"] - ref) ** 2).mean().item()
print(f"greedy     {base:.6e}  (x1.000)")
for k, o in outs.items():
    if k != "greedy":
        m = ((o - ref) ** 2).mean().item()
        print(f"{k:<10} {m:.6e}  (x{base / m:.3f})  score_vs_greedy={1 - m / base:+.3f}")
print(f"v1         {mse_v1:.6e}  (x{base / mse_v1:.3f})  score_vs_greedy={1 - mse_v1 / base:+.3f}")
print("\n=== hypothetical online score of v1 (attn, if judge baseline == variant) ===")
for k, o in outs.items():
    if k == "greedy":
        continue
    m = ((o - ref) ** 2).mean().item()
    print(f"if judge baseline == {k:<10}: v1 score = {(m - mse_v1) / m * 100:+.1f}%")

# ------------------- Stress: element MSE ratios on synthetic data -------------------
print("\n=== stress: v1 vs variants on synthetic distributions (element MSE, x = vs greedy) ===")
torch.manual_seed(0)
cases = {
    "uniform": torch.rand(256, 1024) * 2 - 1,
    "gauss": torch.randn(256, 1024),
    "outlier1e3": torch.randn(256, 1024) * (torch.rand(256, 1024) < 0.001).float() * 1e3 + torch.randn(256, 1024) * 0.01,
    "lognorm": torch.exp(torch.randn(256, 1024) * 3) * (torch.rand(256, 1024) - 0.5).sign(),
    "mixed_scale": torch.randn(256, 1024) * torch.logspace(-4, 2, 1024),
    "sparse": (torch.randn(256, 1024) * (torch.rand(256, 1024) < 0.05).float()),
    "tiny": torch.randn(256, 1024) * 1e-6,
    "huge": torch.randn(256, 1024) * 1e4,
}
for name, x in cases.items():
    g = ((V.deq(V.quant_greedy(x)) - x) ** 2).mean().item()
    res = [f"{name:<10}"]
    for k, fn in [("norm7", V.quant_norm7), ("usearch", V.quant_usearch),
                  ("usearch_lv", V.quant_usearch_lv)]:
        m = ((V.deq(fn(x)) - x) ** 2).mean().item()
        res.append(f"{k}=x{g / m:5.2f}")
    # v1 with uniform weights == usearch; with wild weights test worst case
    print("  ".join(res))
