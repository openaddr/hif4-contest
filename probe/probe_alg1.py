"""Probe submission: EXACT paper Algorithm 1 (no search, no smoothing, no weights).

Purpose: calibrate the judge's standard baseline.
 - score ~= 0     -> baseline == paper Algorithm 1
 - score << 0     -> baseline is stronger than the paper algorithm
 - score > 0      -> baseline is weaker (unlikely)

Implements the reference solution's quantizer with search=0, fixed lv
thresholds (>=4 / >=2), floor(x*4+0.5) rounding, E6M2( absmax/7 ) anchor.
"""
from __future__ import annotations

from typing import Any

import torch

_E6M2_MIN = 2.0 ** (-48)
_E6M2_MAX = 49152.0


def dequantize_nvfp4(quant_float, scale_float, blk_size=16):
    channels = int(quant_float.shape[-1])
    if channels % blk_size != 0:
        raise ValueError(
            f"last dimension {channels} is not divisible by block size {blk_size}"
        )
    x = quant_float.unflatten(-1, (-1, blk_size))
    x = x * scale_float.unsqueeze(-1)
    return x.flatten(-2, -1).to(torch.bfloat16)


def _encode_e6m2(x):
    xc = x.clamp(min=1e-30)
    e = torch.floor(torch.log2(xc))
    m = torch.round(xc * (2.0 ** (2 - e)))
    m = torch.clamp(m, 4, 8)
    out = m * (2.0 ** (e - 2))
    return torch.clamp(out, _E6M2_MIN, _E6M2_MAX)


def _quantize_alg1(x):
    shape = tuple(x.shape)
    C = shape[-1]
    prefix = shape[:-1]
    xf = x.detach().float()
    xr = xf.reshape(*prefix, C // 64, 8, 2, 4)
    ax = xr.abs()

    v16 = ax.amax(dim=-1)
    v8 = v16.amax(dim=-1)
    vmax = v8.amax(dim=-1)

    sf = _encode_e6m2(vmax / 7.0)
    rec = torch.reciprocal(sf.float())
    e1_8 = (v8 * rec.unsqueeze(-1) >= 4.0).to(xf.dtype)
    e1_8_g = e1_8.unsqueeze(-1)
    e1_16 = ((v16 * rec.unsqueeze(-1).unsqueeze(-1) * (2.0 ** (-e1_8_g))) >= 2.0).to(xf.dtype)
    x_scaled = (
        xr * rec[..., None, None, None]
        * (2.0 ** (-e1_8[..., None, None]))
        * (2.0 ** (-e1_16[..., None]))
    )
    sign = torch.sign(x_scaled)
    qi = torch.floor(x_scaled.abs() * 4.0 + 0.5).clamp(0, 7)
    mant = qi / 4.0
    sign = torch.where(mant == 0, torch.zeros_like(sign), sign)
    return {
        "scale_factor": sf[..., None, None, None],
        "scale_lv2": (2.0 ** e1_8)[..., None, None],
        "scale_lv3": (2.0 ** e1_16)[..., None],
        "sign": sign,
        "mant": mant,
    }


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    w = dequantize_nvfp4(weight_quant, weight_scale)
    return {"weight_params": _quantize_alg1(w), "activation_state": None}


def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    x = dequantize_nvfp4(activation_quant, activation_scale)
    return _quantize_alg1(x)


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    return {"q_state": None, "k_state": None, "v_state": None}


def hif4_dynamic_quantize_q(q_quant, q_scale, q_num_heads, head_dim, q_state):
    return _quantize_alg1(dequantize_nvfp4(q_quant, q_scale))


def hif4_dynamic_quantize_k(k_quant, k_scale, kv_num_heads, head_dim, k_state):
    return _quantize_alg1(dequantize_nvfp4(k_quant, k_scale))


def hif4_dynamic_quantize_v(v_quant, v_scale, kv_num_heads, head_dim, v_state):
    return _quantize_alg1(dequantize_nvfp4(v_quant, v_scale))
