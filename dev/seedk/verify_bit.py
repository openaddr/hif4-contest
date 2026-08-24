"""Bit-level verification of the seed no-op on mini groups.

Runs seeds 777 (shipped) vs 1001 (alt) on the FULL stack and compares, before
any reduction:
  - mse maps ((x_play @ w_play.T - ref)**2) elementwise, torch.equal
  - |x_play| and |w_play| elementwise (abs-based parts must be bit-identical)
  - x_play sign flips consistent with |x_play| equality + deq outputs nonzero
  - attention: per-case out tensors and the score-relevant maps
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "dev"))
import hif4  # noqa: E402
import variants as V  # noqa: E402

MINI = os.path.join(ROOT, "example", "mini_sample")
COPY_SOL = os.path.join(HERE, "solution.py")


def load_mod(base):
    spec = importlib.util.spec_from_file_location(f"_v_{base}", COPY_SOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._ROT_LIN_SEED_BASE = base
    mod._ROT_ATTN_SEED_BASE = base
    return mod


def linear_check():
    g = torch.load(os.path.join(MINI, "linear.pt"), weights_only=True,
                   map_location="cpu")[0]
    out = {}
    for base in (777, 1001):
        mod = load_mod(base)
        cal = mod.hif4_calibration_and_quantize_weight(
            g["weight"][0], g["weight"][1], g["calib_activation_list"])
        w_play = hif4.hif4_dequantize(cal["weight_params"])
        w_ref = hif4.dequantize_nvfp4(*g["weight"])
        st = cal["activation_state"]
        per = []
        for pair in g["test_activation_list"]:
            x_ref = hif4.dequantize_nvfp4(*pair)
            ref = hif4.linear_ref(x_ref, w_ref)
            p = mod.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
            x_play = hif4.hif4_dequantize(p)
            mse_map = (hif4.linear_ref(x_play, w_play) - ref) ** 2
            per.append((x_play, w_play, mse_map))
        out[base] = (per, st)
    (p0, st0), (p1, st1) = out[777], out[1001]
    print("[linear] flags:", st0.get("mode"), st0.get("g"), st0.get("gw") is not None)
    for i, ((x0, w0, m0), (x1, w1, m1)) in enumerate(zip(p0, p1)):
        print(f"  test{i}: mse_map torch.equal={torch.equal(m0, m1)}  "
              f"|x| equal={torch.equal(x0.abs(), x1.abs())}  "
              f"|w| equal={torch.equal(w0.abs(), w1.abs())}")
    gw0, gw1 = st0["gw"], st1["gw"]
    gw0f = gw0.float() if gw0 is not None else None
    gw1f = gw1.float() if gw1 is not None else None
    print(f"  gw diag equal={torch.equal(gw0f.diagonal(), gw1f.diagonal())}  "
          f"u_act diag equal={torch.equal(st0['u_act'].diagonal(), st1['u_act'].diagonal())}  "
          f"order equal={torch.equal(st0['order'], st1['order'])}")


def attn_check():
    g = torch.load(os.path.join(MINI, "attn.pt"), weights_only=True,
                   map_location="cpu")[0]
    qh, kvh, dh = g["q_num_heads"], g["kv_num_heads"], g["head_dim"]
    out = {}
    for base in (0xA5A5, 2001):
        mod = load_mod(base)
        acal = mod.hif4_calibration_attention(g["calib"], qh, kvh, dh)
        outs = []
        for smp in g["test"]:
            q_ref = hif4.dequantize_nvfp4(*smp["q"])
            k_ref = hif4.dequantize_nvfp4(*smp["k"])
            v_ref = hif4.dequantize_nvfp4(*smp["v"])
            ref = hif4.attn_ref(q_ref, k_ref, v_ref, qh, kvh, dh)
            pq = mod.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh, dh,
                                             acal["q_state"])
            pk = mod.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], kvh, dh,
                                             acal["k_state"])
            pv = mod.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], kvh, dh,
                                             acal["v_state"])
            o = hif4.attn_ref(hif4.hif4_dequantize(pq),
                              hif4.hif4_dequantize(pk),
                              hif4.hif4_dequantize(pv), qh, kvh, dh)
            outs.append((o - ref) ** 2)
        out[base] = outs
    print("[attn] rot flag:", acal["q_state"].get("rot"))
    for i, (m0, m1) in enumerate(zip(out[0xA5A5], out[2001])):
        print(f"  test{i}: out-mse map torch.equal={torch.equal(m0, m1)}")


if __name__ == "__main__":
    linear_check()
    attn_check()
