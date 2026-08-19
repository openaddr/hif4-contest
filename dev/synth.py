"""Synthetic judge-like benchmark: multi-group, multi-distribution NVFP4 data.

Simulates the contest data pipeline: raw tensors with channel structure and
outliers -> NVFP4 quantization (E2M1 grid, block 16, bf16 scales) -> the same
carrier format as mini_sample. Used to stress-test transfer robustness of
solution variants across distributions mini_sample doesn't cover.
"""
from __future__ import annotations

import math

import torch

E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def quant_nvfp4(x: torch.Tensor, blk: int = 16):
    """Simulate NVFP4: per-block scale = absmax/6 (bf16), values on E2M1 grid."""
    xb = x.float().reshape(-1, blk)
    amax = xb.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    scale = (amax / 6.0).to(torch.bfloat16).float()
    scale = scale.clamp_min(1e-30)
    q = xb / scale
    # round to nearest E2M1 grid point (by magnitude, sign preserved)
    sign = torch.sign(q)
    aq = q.abs()
    idx = torch.bucketize(aq, (E2M1_GRID[1:] + E2M1_GRID[:-1]) / 2.0)
    carrier = sign * E2M1_GRID[idx]
    dq = (carrier * scale).to(torch.bfloat16)
    return dq.reshape(x.shape).contiguous()


def _chan_gains(C: int, spread: float, gen: torch.Generator) -> torch.Tensor:
    return torch.exp((torch.rand(C, generator=gen) - 0.5) * 2 * math.log(10.0) * spread)


def make_linear_group(seed: int, M: int, K: int, tokens=(128, 512), spread=0.5,
                      outlier_p=0.0, w_spread=0.3):
    gen = torch.Generator().manual_seed(seed)
    # weight: per-row scale x per-channel gains x gaussian
    w = (torch.randn(M, 1, generator=gen)
         * _chan_gains(K, w_spread, gen).unsqueeze(0)
         * torch.randn(M, K, generator=gen)) * 0.05
    w_q = quant_nvfp4(w)

    def make_act(T: int):
        x = (torch.randn(T, 1, generator=gen)
             * _chan_gains(K, spread, gen).unsqueeze(0)
             * torch.randn(T, K, generator=gen))
        if outlier_p > 0:
            mask = torch.rand(T, K, generator=gen) < outlier_p
            x = x + mask.float() * torch.randn(T, K, generator=gen) * x.abs().amax() * 3
        return quant_nvfp4(x)

    calib = [(make_act(T), None) for T in tokens]      # placeholder pairs
    test = [(make_act(T), None) for T in tokens]

    # rebuild in proper carrier/scale pair form
    def split(dq):
        # invert: recompute scale per 16-block from dq is lossy; instead keep carriers
        raise RuntimeError("use make_act_raw")

    # simpler: return raw carriers by re-running quant with scale output
    def make_act_pair(T: int):
        x = (torch.randn(T, 1, generator=gen)
             * _chan_gains(K, spread, gen).unsqueeze(0)
             * torch.randn(T, K, generator=gen))
        if outlier_p > 0:
            mask = torch.rand(T, K, generator=gen) < outlier_p
            x = x + mask.float() * torch.randn(T, K, generator=gen) * x.abs().amax() * 3
        xb = x.reshape(-1, 16)
        amax = xb.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
        scale_bf = (amax / 6.0).to(torch.bfloat16)
        scale = scale_bf.float().clamp_min(1e-30)
        q = xb / scale
        sign = torch.sign(q)
        idx = torch.bucketize(q.abs(), (E2M1_GRID[1:] + E2M1_GRID[:-1]) / 2.0)
        carrier = sign * E2M1_GRID[idx]
        carrier = carrier.reshape(T, -1).to(torch.bfloat16)
        scale = scale.reshape(T, -1).to(torch.bfloat16)
        return carrier, scale

    calib = [make_act_pair(T) for T in tokens]
    test = [make_act_pair(T) for T in tokens]

    # weight carrier/scale
    wb = w.reshape(-1, 16)
    wmax = wb.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    ws = ((wmax / 6.0).to(torch.bfloat16).float()).clamp_min(1e-30)
    q = wb / ws
    idx = torch.bucketize(q.abs(), (E2M1_GRID[1:] + E2M1_GRID[:-1]) / 2.0)
    wc = (torch.sign(q) * E2M1_GRID[idx]).reshape(M, -1).to(torch.bfloat16)
    wss = ws.reshape(M, -1).to(torch.bfloat16)

    return {
        "weight": (wc, wss),
        "calib_activation_list": calib,
        "test_activation_list": test,
    }


def make_attn_group(seed: int, qh: int, kvh: int, dh: int, seqlens=(128, 512),
                    spread=0.4, outlier_p=0.0):
    gen = torch.Generator().manual_seed(seed)
    qc, kc = qh * dh, kvh * dh

    def make_pair(T: int, C: int, sp: float):
        x = (torch.randn(T, 1, generator=gen)
             * _chan_gains(C, sp, gen).unsqueeze(0)
             * torch.randn(T, C, generator=gen))
        if outlier_p > 0:
            mask = torch.rand(T, C, generator=gen) < outlier_p
            x = x + mask.float() * torch.randn(T, C, generator=gen) * x.abs().amax() * 3
        xb = x.reshape(-1, 16)
        amax = xb.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
        scale = ((amax / 6.0).to(torch.bfloat16).float()).clamp_min(1e-30)
        q = xb / scale
        idx = torch.bucketize(q.abs(), (E2M1_GRID[1:] + E2M1_GRID[:-1]) / 2.0)
        carrier = (torch.sign(q) * E2M1_GRID[idx]).reshape(T, -1).to(torch.bfloat16)
        return carrier, scale.reshape(T, -1).to(torch.bfloat16)

    def sample(T: int):
        return {
            "q": make_pair(T, qc, spread),
            "k": make_pair(T, kc, spread * 0.7),
            "v": make_pair(T, kc, spread * 0.5),
        }

    return {
        "q_num_heads": qh,
        "kv_num_heads": kvh,
        "head_dim": dh,
        "calib": [sample(T) for T in seqlens],
        "test": [sample(T) for T in seqlens],
    }


def deq(pair):
    from importlib import import_module
    # local import to avoid circular dependency at module load
    import importlib.util, os
    global _dq
    try:
        return _dq(*pair)
    except NameError:
        pass
    spec = importlib.util.spec_from_file_location(
        "_sol_dq", os.path.join(os.path.dirname(__file__), "..", "example", "solution", "solution.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _dq = mod.dequantize_nvfp4
    return _dq(*pair)
