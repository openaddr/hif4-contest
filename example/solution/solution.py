"""HiF4 solution: NVFP4 -> HiF4 conversion for Linear and Attention.

Core idea: hierarchical greedy layout identical to the standard quantizer,
but the per-block E6M2 scale_factor is searched over the exact E6M2 grid
(significands 1.0/1.25/1.5/1.75 around 2^floor(log2(absmax))) minimizing the
*weighted* block MSE. Weights reflect how each channel's quantization error
propagates into the final output:

  - Weight channel j    -> mean_t act[t, j]^2      (output err ~= X @ dW^T)
  - Activation channel j-> sum_out W[:, j]^2       (output err ~= dX @ W^T)
  - Q channel (h, d)    -> mean K[kv(h), d]^2      (logit err, diagonal approx)
  - K channel (kv, d)   -> mean over group of Q^2
  - V                   -> uniform (softmax mixing, no cheap diagonal)
"""
from __future__ import annotations

from typing import Any

import torch

SF_MIN = 2.0 ** -48
SF_MAX = 49152.0
# E6M2-exact significands; kept as one exponent-below to also probe smaller scales.
CANDS = (0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75)
CANDS_T = torch.tensor(CANDS, dtype=torch.float32)


def dequantize_nvfp4(quant_float, scale_float, blk_size=16):
    channels = int(quant_float.shape[-1])
    if channels % blk_size != 0:
        raise ValueError(
            f"last dimension {channels} is not divisible by block size {blk_size}"
        )
    x = quant_float.unflatten(-1, (-1, blk_size))
    x = x * scale_float.unsqueeze(-1)
    return x.flatten(-2, -1).to(torch.bfloat16)


def _quantize_weighted(x2d: torch.Tensor, wgt: torch.Tensor) -> dict[str, torch.Tensor]:
    """Search scale_factor over E6M2 candidates; greedy lv2/lv3; weighted MSE.

    x2d: (R, C) fp32, C % 64 == 0. wgt: (C,) fp32 >= 0 (broadcast over rows).
    """
    R, C = x2d.shape
    nb = C // 64
    xb = x2d.reshape(R, nb, 8, 2, 4)
    ab = xb.abs()
    wblk = wgt.reshape(nb, 8, 2, 4)  # broadcast (R, nb, 8, 2, 4) x (nb,8,2,4)

    amax = ab.amax(dim=(2, 3, 4), keepdim=True)          # (R, nb, 1, 1, 1)
    amax8 = ab.amax(dim=(3, 4), keepdim=True)            # (R, nb, 8, 1, 1)
    amax4 = ab.amax(dim=4, keepdim=True)                 # (R, nb, 8, 2, 1)
    pe = torch.exp2(torch.floor(torch.log2(amax.clamp_min(1e-38))))  # 2^e

    err_best = None
    idx_best = None
    for i in range(len(CANDS)):
        sf = (pe * CANDS_T[i]).clamp(SF_MIN, SF_MAX)
        lv2 = torch.where(amax8 / sf > 1.75, 2.0, 1.0)
        lv3 = torch.where(amax4 / (sf * lv2) > 1.75, 2.0, 1.0)
        unit = sf * lv2 * lv3
        mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
        err = ((mant * unit - ab) ** 2 * wblk).sum(dim=(2, 3, 4))      # (R, nb)
        if err_best is None:
            err_best, idx_best = err, torch.zeros_like(err, dtype=torch.int64)
        else:
            better = err < err_best
            err_best = torch.where(better, err, err_best)
            idx_best = torch.where(better, i, idx_best)

    sf = (pe * CANDS_T[idx_best.reshape(R, nb, 1, 1, 1)]).clamp(SF_MIN, SF_MAX)
    lv2 = torch.where(amax8 / sf > 1.75, 2.0, 1.0)
    lv3 = torch.where(amax4 / (sf * lv2) > 1.75, 2.0, 1.0)
    unit = sf * lv2 * lv3
    mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
    sign = torch.sign(xb)
    return {
        "scale_factor": sf,
        "scale_lv2": lv2,
        "scale_lv3": lv3,
        "sign": sign,
        "mant": mant,
    }


def _uniform_weights(C: int) -> torch.Tensor:
    return torch.ones(C, dtype=torch.float32)


# =============================================================================
# 1. Linear calibration + Weight quantization
# =============================================================================

def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    w = dequantize_nvfp4(weight_quant, weight_scale).float()

    # activation energy per input channel: diagonal of the calib Gram matrix
    sq_sum = torch.zeros(w.shape[1], dtype=torch.float32)
    n_tok = 0
    for act_quant, act_scale in calib_activation_list:
        a = dequantize_nvfp4(act_quant, act_scale).float()
        sq_sum += (a * a).sum(dim=0)
        n_tok += a.shape[0]
    act_energy = sq_sum / max(n_tok, 1)

    weight_params = _quantize_weighted(w, act_energy)

    # for online activation quantization: per-channel weight column energy
    w_col = (w * w).sum(dim=0)
    activation_state = {"w_col": w_col}
    return {"weight_params": weight_params, "activation_state": activation_state}


# =============================================================================
# 2. Dynamic Activation quantization
# =============================================================================

def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    x = dequantize_nvfp4(activation_quant, activation_scale).float()
    wgt = activation_state["w_col"] if isinstance(activation_state, dict) else None
    if not isinstance(wgt, torch.Tensor) or wgt.numel() != x.shape[-1]:
        wgt = _uniform_weights(x.shape[-1])
    return _quantize_weighted(x, wgt)


# =============================================================================
# 3. Attention calibration
# =============================================================================

def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    qh, kvh, dh = q_num_heads, kv_num_heads, head_dim
    rep = qh // kvh

    q_sq = torch.zeros(qh * dh, dtype=torch.float32)
    k_sq = torch.zeros(kvh * dh, dtype=torch.float32)
    v_sq = torch.zeros(kvh * dh, dtype=torch.float32)
    n = 0
    for sample in calib_qkv_list:
        q = dequantize_nvfp4(*sample["q"]).float()
        k = dequantize_nvfp4(*sample["k"]).float()
        v = dequantize_nvfp4(*sample["v"]).float()
        q_sq += (q * q).sum(dim=0)
        k_sq += (k * k).sum(dim=0)
        v_sq += (v * v).sum(dim=0)
        n += q.shape[0]

    q_energy = q_sq / max(n, 1)   # (qh*dh,)
    k_energy = k_sq / max(n, 1)   # (kvh*dh,)
    v_energy = v_sq / max(n, 1)

    # Q error at channel (h, d) hits logits of head h via K[kv(h), d]
    k_map = torch.arange(kvh).repeat_interleave(rep)            # kv head per q head
    w_q = k_energy.view(kvh, 1, dh)[k_map].flatten()            # (qh*dh,)
    # K error at (kv, d) hits logits of the rep heads sharing it
    w_k = q_energy.view(qh, dh).view(kvh, rep, dh).sum(dim=1).flatten()
    # V: no cheap diagonal; use mean energy so blocks stay comparable
    w_v = v_energy

    return {
        "q_state": {"w": w_q.contiguous()},
        "k_state": {"w": w_k.contiguous()},
        "v_state": {"w": w_v.contiguous()},
    }


# =============================================================================
# 4/5/6. Dynamic Q/K/V quantization
# =============================================================================

def _dynamic_qkv(quant, scale, state) -> dict[str, torch.Tensor]:
    x = dequantize_nvfp4(quant, scale).float()
    wgt = state["w"] if isinstance(state, dict) else None
    if not isinstance(wgt, torch.Tensor) or wgt.numel() != x.shape[-1]:
        wgt = _uniform_weights(x.shape[-1])
    return _quantize_weighted(x, wgt)


def hif4_dynamic_quantize_q(
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_num_heads: int,
    head_dim: int,
    q_state: Any,
) -> dict[str, torch.Tensor]:
    return _dynamic_qkv(q_quant, q_scale, q_state)


def hif4_dynamic_quantize_k(
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    k_state: Any,
) -> dict[str, torch.Tensor]:
    return _dynamic_qkv(k_quant, k_scale, k_state)


def hif4_dynamic_quantize_v(
    v_quant: torch.Tensor,
    v_scale: torch.Tensor,
    kv_num_heads: int,
    head_dim: int,
    v_state: Any,
) -> dict[str, torch.Tensor]:
    return _dynamic_qkv(v_quant, v_scale, v_state)
