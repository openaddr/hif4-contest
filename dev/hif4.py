"""HiF4 primitives + standard (greedy) baseline quantizer + task references.

Shared by the local eval harness. The baseline mirrors the task book's
Fig.2 "standard HiF4 quantization": power-of-2 E6M2 scale_factor chosen as
2^floor(log2(absmax)), greedy lv2/lv3 in {1,2}, mantissa rounding at 0.25.
"""
from __future__ import annotations

import math

import torch

SF_MIN = 2.0 ** -48
SF_MAX = 49152.0  # 1.5 * 2^15, per self_check.py public range


def dequantize_nvfp4(quant, scale, blk_size=16):
    C = quant.shape[-1]
    x = quant.unflatten(-1, (-1, blk_size)) * scale.unsqueeze(-1)
    return x.flatten(-2, -1).to(torch.bfloat16)


def _params_from_scales(sf, lv2, lv3, xb):
    """sf: (...,nb,1,1,1)  lv2: (...,nb,8,1,1)  lv3: (...,nb,8,2,1)  xb: (...,nb,8,2,4)"""
    ab = xb.abs()
    unit = sf * lv2 * lv3
    mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
    return {
        "scale_factor": sf,
        "scale_lv2": lv2,
        "scale_lv3": lv3,
        "sign": torch.sign(xb),
        "mant": mant,
    }


def hif4_dequantize(p):
    return (p["sign"] * p["mant"] * p["scale_lv3"] * p["scale_lv2"] * p["scale_factor"]).flatten(-4, -1)


def hif4_quantize_standard(x):
    """Greedy standard HiF4 baseline. x: fp tensor, last dim % 64 == 0."""
    shape = x.shape
    xb = x.unflatten(-1, (shape[-1] // 64, 8, 2, 4)).float()
    amax = xb.abs().amax(dim=(2, 3, 4), keepdim=True)
    sf = torch.exp2(torch.floor(torch.log2(amax.clamp_min(1e-38))))
    sf = sf.clamp(SF_MIN, SF_MAX)
    r2 = xb.abs().amax(dim=(3, 4), keepdim=True) / sf
    lv2 = torch.where(r2 > 1.75, 2.0, 1.0)
    r3 = xb.abs().amax(dim=4, keepdim=True) / (sf * lv2)
    lv3 = torch.where(r3 > 1.75, 2.0, 1.0)
    return _params_from_scales(sf, lv2, lv3, xb)


def linear_ref(x_bf16, w_bf16):
    return x_bf16.float() @ w_bf16.float().T


def attn_ref(q, k, v, qh, kvh, dh):
    """GQA attention on dequantized tensors, FP32, non-causal."""
    seq = q.shape[0]
    qf = q.float().view(seq, qh, dh).transpose(0, 1)          # (qh, seq, dh)
    kf = k.float().view(seq, kvh, dh).transpose(0, 1)
    vf = v.float().view(seq, kvh, dh).transpose(0, 1)
    rep = qh // kvh
    kf = kf.repeat_interleave(rep, dim=0)
    vf = vf.repeat_interleave(rep, dim=0)
    scores = torch.bmm(qf, kf.transpose(1, 2)) / math.sqrt(dh)
    prob = torch.softmax(scores, dim=-1)
    out = torch.bmm(prob, vf)                                  # (qh, seq, dh)
    return out.transpose(0, 1).reshape(seq, qh * dh)
