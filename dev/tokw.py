"""Attention-mass-aware row weighting for K/V/Q quantization.

Sensitivity of the output to K_c (token c) scales with its attention mass
(sum over rows of P^2-ish); to V_c with mass^2-ish; to Q_r with row
concentration. Current search treats all rows uniformly. Weights are
estimated from CALIBRATION attention probabilities only (transferable?).
"""
import sys, os
import importlib.util

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "example", "solution"))
sys.path.insert(0, ROOT)


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load_mod(os.path.join(ROOT, "..", "example", "solution", "solution.py"), "sol")
import hif4  # noqa: E402
import synth  # noqa: E402
import variants as V  # noqa: E402


def probs(q, k, qh, kvh, dh):
    seq = q.shape[0]
    qf = q.view(seq, qh, dh).transpose(0, 1)
    kf = k.view(seq, kvh, dh).transpose(0, 1)
    rep = qh // kvh
    sc = torch.bmm(qf, kf.repeat_interleave(rep, 0).transpose(1, 2)) / (dh ** 0.5)
    return torch.softmax(sc, dim=-1)  # (qh, seq, seq)


def run(group):
    qh, kvh, dh = group["q_num_heads"], group["kv_num_heads"], group["head_dim"]
    calib, tests = group["calib"], group["test"]
    st = S.hif4_calibration_attention(calib, qh, kvh, dh)

    # ---- calib-derived token masses (largest sample) ----
    big = max(calib, key=lambda s: s["q"][0].shape[0])
    qd = hif4.dequantize_nvfp4(*big["q"]).float()
    kd = hif4.dequantize_nvfp4(*big["k"]).float()
    P = probs(qd, kd, qh, kvh, dh)
    mass = P.square().sum(dim=(0, 1))     # (seq,) token mass
    rowc = P.square().sum(dim=-1).mean(dim=0)  # (seq,) row concentration
    mass = mass / mass.mean()
    rowc = rowc / rowc.mean()
    wk = mass.clamp(0.25, 4.0)   # K/V token weights
    wq = rowc.clamp(0.25, 4.0)   # Q row weights

    def out_of(q, k, v):
        return hif4.attn_ref(q, k, v, qh, kvh, dh)

    tot = tot_w = 0.0
    for smp in tests:
        q_ref = hif4.dequantize_nvfp4(*smp["q"]).float()
        k_ref = hif4.dequantize_nvfp4(*smp["k"]).float()
        v_ref = hif4.dequantize_nvfp4(*smp["v"]).float()
        ref = out_of(q_ref, k_ref, v_ref)

        # --- baseline pipeline ---
        pq = S.hif4_dynamic_quantize_q(*smp["q"], qh, dh, st["q_state"])
        pk = S.hif4_dynamic_quantize_k(*smp["k"], kvh, dh, st["k_state"])
        pv = S.hif4_dynamic_quantize_v(*smp["v"], kvh, dh, st["v_state"])
        tot += ((out_of(hif4.hif4_dequantize(pq), hif4.hif4_dequantize(pk),
                        hif4.hif4_dequantize(pv)) - ref) ** 2).mean().item()

        # --- weighted search variant (same transforms, row-weighted search) ---
        rot = st["q_state"].get("rot") == 1
        R = S._make_R(dh) if rot else None
        T = q_ref.shape[0]
        Cq = q_ref.shape[1]

        def prep(x, num_heads, wrow):
            if R is not None:
                x = (x.view(T, num_heads, dh) @ R).reshape(T, -1)
            w2d = wrow[:T].unsqueeze(1).expand(T, x.shape[1]).contiguous()
            return S._quantize_weighted(x.contiguous(), w2d)

        pq2 = prep(q_ref, qh, wq)
        pk2 = prep(k_ref, kvh, wk)
        # V is never rotated (its output is consumed directly)
        wv = wk[:T].unsqueeze(1).expand(T, v_ref.shape[1]).contiguous()
        pv2 = S._quantize_weighted(v_ref.contiguous(), wv)
        tot_w += ((out_of(hif4.hif4_dequantize(pq2), hif4.hif4_dequantize(pk2),
                          hif4.hif4_dequantize(pv2)) - ref) ** 2).mean().item()
    n = len(tests)
    return tot / n, tot_w / n


torch.manual_seed(0)
for name, group in [
    ("gqa_256_r0.4", synth.make_attn_group(21, 16, 2, 256, spread=0.4)),
    ("mha_128_r0.3", synth.make_attn_group(22, 8, 8, 128, spread=0.3)),
    ("flat_256_r0.1", synth.make_attn_group(23, 16, 2, 256, spread=0.1)),
    ("gqa_128_r0.5", synth.make_attn_group(24, 32, 4, 128, spread=0.5)),
]:
    b, w = run(group)
    print(f"{name:16s} play MSE {b:.3e} -> {w:.3e}  ({100*(1-w/b):+.1f}%)")
