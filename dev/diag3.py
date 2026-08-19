"""Score the current solution against the norm7 baseline (best guess of the
judge's standard quantizer) instead of the greedy reconstruction."""
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


SOL = load_mod(os.path.join(os.path.dirname(__file__), "..", "example", "solution", "solution.py"), "sol")
root = os.path.join(os.path.dirname(__file__), "..", "example", "mini_sample")
total = 0.0
n = 0

lin = torch.load(os.path.join(root, "linear.pt"), weights_only=True, map_location="cpu")[0]
w_ref = hif4.dequantize_nvfp4(*lin["weight"])
cal = SOL.hif4_calibration_and_quantize_weight(*lin["weight"], lin["calib_activation_list"])
w_play = hif4.hif4_dequantize(cal["weight_params"])
w_std = V.deq(V.quant_alg1(w_ref.float()))

for ti, pair in enumerate(lin["test_activation_list"]):
    x_ref = hif4.dequantize_nvfp4(*pair)
    ref = hif4.linear_ref(x_ref, w_ref)
    x_std = V.deq(V.quant_alg1(x_ref.float()))
    mse_std = ((hif4.linear_ref(x_std, w_std) - ref) ** 2).mean().item()
    p = SOL.hif4_dynamic_quantize_activation(pair[0], pair[1], cal["activation_state"])
    mse_play = ((hif4.linear_ref(x_ref, w_play) - ref) ** 2).mean().item()
    s = (mse_std - mse_play) / mse_std
    total += s
    n += 1
    print(f"[linear t{ti}] std={mse_std:.4e} play={mse_play:.4e} score={s:+.4f}")

att = torch.load(os.path.join(root, "attn.pt"), weights_only=True, map_location="cpu")[0]
qh, kvh, dh = att["q_num_heads"], att["kv_num_heads"], att["head_dim"]
acal = SOL.hif4_calibration_attention(att["calib"], qh, kvh, dh)

for ti, smp in enumerate(att["test"]):
    q_ref = hif4.dequantize_nvfp4(*smp["q"])
    k_ref = hif4.dequantize_nvfp4(*smp["k"])
    v_ref = hif4.dequantize_nvfp4(*smp["v"])
    ref = hif4.attn_ref(q_ref, k_ref, v_ref, qh, kvh, dh)
    qs = V.deq(V.quant_alg1(q_ref.float()))
    ks = V.deq(V.quant_alg1(k_ref.float()))
    vs = V.deq(V.quant_alg1(v_ref.float()))
    mse_std = ((hif4.attn_ref(qs, ks, vs, qh, kvh, dh) - ref) ** 2).mean().item()

    pq = SOL.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh, dh, acal["q_state"])
    pk = SOL.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], kvh, dh, acal["k_state"])
    pv = SOL.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], kvh, dh, acal["v_state"])
    out = hif4.attn_ref(hif4.hif4_dequantize(pq), hif4.hif4_dequantize(pk), hif4.hif4_dequantize(pv), qh, kvh, dh)
    mse_play = ((out - ref) ** 2).mean().item()
    s = (mse_std - mse_play) / mse_std
    total += s
    n += 1
    print(f"[attn   t{ti}] std={mse_std:.4e} play={mse_play:.4e} score={s:+.4f}")

print(f"\nTOTAL vs exact-alg1 baseline: {total:+.4f} over {n} cases "
      f"(=> x100 x50groups est online ≈ {total / n * 100 * 500:+.0f})")
