"""Decompose attention output error: Q/K-side vs V-side + P concentration."""
import sys, os
import importlib.util

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "example", "solution"))
sys.path.insert(0, ROOT)

S = load_mod = None


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load_mod(os.path.join(ROOT, "..", "example", "solution", "solution.py"), "sol")
import hif4  # noqa: E402
import synth  # noqa: E402
import variants as V  # noqa: E402


def run(group):
    qh, kvh, dh = group["q_num_heads"], group["kv_num_heads"], group["head_dim"]
    calib, tests = group["calib"], group["test"]
    st = S.hif4_calibration_attention(calib, qh, kvh, dh)

    def out_of(q, k, v):
        return hif4.attn_ref(q, k, v, qh, kvh, dh)

    tot = qk_side = v_side = 0.0
    pmax_ratio = []
    for smp in tests:
        q_ref = hif4.dequantize_nvfp4(*smp["q"]).float()
        k_ref = hif4.dequantize_nvfp4(*smp["k"]).float()
        v_ref = hif4.dequantize_nvfp4(*smp["v"]).float()
        ref = out_of(q_ref, k_ref, v_ref)
        pq = S.hif4_dynamic_quantize_q(*smp["q"], qh, dh, st["q_state"])
        pk = S.hif4_dynamic_quantize_k(*smp["k"], kvh, dh, st["k_state"])
        pv = S.hif4_dynamic_quantize_v(*smp["v"], kvh, dh, st["v_state"])
        vq = hif4.hif4_dequantize(pq)
        vk = hif4.hif4_dequantize(pk)
        vvv = hif4.hif4_dequantize(pv)
        tot += ((out_of(vq, vk, vvv) - ref) ** 2).mean().item()
        qk_side += ((out_of(vq, vk, v_ref) - ref) ** 2).mean().item()
        v_side += ((out_of(q_ref, k_ref, vvv) - ref) ** 2).mean().item()
        # P concentration for the reference attention
        seq = q_ref.shape[0]
        qf = q_ref.view(seq, qh, dh).transpose(0, 1)
        kf = k_ref.view(seq, kvh, dh).transpose(0, 1)
        rep = qh // kvh
        sc = torch.bmm(qf, kf.repeat_interleave(rep, 0).transpose(1, 2)) / (dh ** 0.5)
        P = torch.softmax(sc, dim=-1)
        pmax_ratio.append((P.amax(dim=-1) / (1.0 / seq)).mean().item())
    n = len(tests)
    print(f"  total {tot/n:.3e} | Q/K-side {100*qk_side/tot:.0f}% | V-side {100*v_side/tot:.0f}% "
          f"| P max/mean-uniform ratio: {sum(pmax_ratio)/n:.1f}x")


torch.manual_seed(0)
for name, group in [
    ("gqa_256_r0.4", synth.make_attn_group(21, 16, 2, 256, spread=0.4)),
    ("mha_128_r0.3", synth.make_attn_group(22, 8, 8, 128, spread=0.3)),
    ("flat_256_r0.1", synth.make_attn_group(23, 16, 2, 256, spread=0.1)),
]:
    print(name)
    run(group)
