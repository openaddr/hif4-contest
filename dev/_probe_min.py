import numpy as np
import torch
from typing import Any


def hif4_calibration_and_quantize_weight(weight_quant, weight_scale, calib_activation_list):
    t = torch.from_numpy(np.zeros((4, 4), dtype=np.float32))
    w = dequantize_nvfp4(weight_quant, weight_scale).float()
    q = torch.round(w / 0.1)
    return {"weight_params": {"scale_factor": torch.ones(1)},
            "activation_state": {"t": t, "x": q.sum()}}


def dequantize_nvfp4(quant, scale):
    return quant.float() * scale.float()


def hif4_dynamic_quantize_activation(quant, scale, activation_state):
    return {"sign": torch.sign(quant.float().sum())[None],
            "mant": torch.zeros(1), "scale_lv2": torch.ones(1),
            "scale_lv3": torch.ones(1), "scale_factor": torch.ones(1)}


def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    return {"q_state": {}, "k_state": {}, "v_state": None}


def hif4_dynamic_quantize_q(q_quant, q_scale, q_num_heads, head_dim, q_state):
    return {}


def hif4_dynamic_quantize_k(k_quant, k_scale, kv_num_heads, head_dim, k_state):
    return {}


def hif4_dynamic_quantize_v(v_quant, v_scale, kv_num_heads, head_dim, v_state):
    return {}
