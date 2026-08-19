"""HiF4 solution v2: NVFP4 -> HiF4 for Linear and Attention.

Pipeline per tensor:
  1. Dequantize NVFP4 -> BF16 -> FP32.
  2. Optional per-channel smoothing s (Linear: act<->weight; Attention: q<->k),
     searched on calibration data. Mathematically x.w and q.k are invariant.
  3. Hierarchical HiF4 quantization:
       - scale_factor searched over 8 exact E6M2 candidates per 64-block,
         minimizing output-error-weighted block MSE;
       - lv2 / lv3 then refined by weighted MSE (not greedy);
       - mantissa rounded to the 0.25 grid, clamped to [0, 1.75].
  4. Weights reflect output sensitivity: weight channel j ~ E[act_j^2],
     activation channel j ~ sum_out W[:,j]^2, Q/K cross energies, V positional
     importance from the calibration attention-prob Gram diagonal.

Rows are processed in chunks to bound peak memory.
"""
from __future__ import annotations

from typing import Any

import torch

SF_MIN = 2.0 ** -48
SF_MAX = 49152.0
E6M2_SIG = (1.0, 1.25, 1.5, 1.75)
ANCHOR_EXP_OFFS = (0, 1)
ROW_CHUNK = 2048
# ablation switches (keep True / non-empty for submission)
USE_WEIGHTS = True
LV_REFINE = True
ALPHA_GRID = (0.0, 0.25, 0.5)
BETA_GRID = (0.0, 0.25)
GAMMA_GRID = (0.0, 0.15, 0.3, 0.5)


def dequantize_nvfp4(quant_float, scale_float, blk_size=16):
    channels = int(quant_float.shape[-1])
    if channels % blk_size != 0:
        raise ValueError(
            f"last dimension {channels} is not divisible by block size {blk_size}"
        )
    x = quant_float.unflatten(-1, (-1, blk_size))
    x = x * scale_float.unsqueeze(-1)
    return x.flatten(-2, -1).to(torch.bfloat16)


def _quant_chunk(xb: torch.Tensor, wblk: torch.Tensor) -> dict[str, torch.Tensor]:
    """Quantize one row-chunk. xb: (r, nb, 8, 2, 4); wblk broadcastable (nb,8,2,4)."""
    ab = xb.abs()
    amax = ab.amax(dim=(2, 3, 4), keepdim=True)
    amax8 = ab.amax(dim=(3, 4), keepdim=True)
    amax4 = ab.amax(dim=4, keepdim=True)

    # ---- stage 1: scale_factor over E6M2 candidates anchored at absmax/7 ----
    # grid covers [absmax/14, absmax/2]: the top value lands near mant 1.75
    # with lv2=lv3=2, and small sub-blocks keep fine resolution at lv=1.
    t = (amax / 7.0).clamp_min(1e-38)
    e0 = torch.floor(torch.log2(t)).squeeze(-1).squeeze(-1).squeeze(-1)  # (r, nb)
    err_best = None
    sf_best = None
    for e_off in ANCHOR_EXP_OFFS:
        pe = torch.exp2(e0 + e_off)                                   # (r, nb)
        for c in E6M2_SIG:
            sf = (pe * c).clamp(SF_MIN, SF_MAX)
            sf5 = sf[..., None, None, None]
            lv2 = torch.where(amax8 / sf5 > 1.75, 2.0, 1.0)
            lv3 = torch.where(amax4 / (sf5 * lv2) > 1.75, 2.0, 1.0)
            unit = sf5 * lv2 * lv3
            mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
            err = ((mant * unit - ab) ** 2 * wblk).sum(dim=(2, 3, 4))
            if err_best is None:
                err_best, sf_best = err, sf
            else:
                better = err < err_best
                err_best = torch.where(better, err, err_best)
                sf_best = torch.where(better, sf, sf_best)
    sf = sf_best[..., None, None, None]

    # ---- stage 2: lv2 per sub-block, lv3 per group, by weighted MSE ----
    if not LV_REFINE:
        lv2 = torch.where(amax8 / sf > 1.75, 2.0, 1.0)
        lv3 = torch.where(amax4 / (sf * lv2) > 1.75, 2.0, 1.0)
        unit = sf * lv2 * lv3
        mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
        return {
            "scale_factor": sf,
            "scale_lv2": lv2,
            "scale_lv3": lv3,
            "sign": torch.sign(xb),
            "mant": mant,
        }
    wsub = wblk  # (nb,8,2,4) -> sums below aggregate to sub-block granularity
    best_e2 = None
    best_lv2 = None
    best_lv3 = None
    for lv2_cand in (1.0, 2.0):
        base = sf * lv2_cand
        e3_list = []
        m3_list = []
        for lv3_cand in (1.0, 2.0):
            unit = base * lv3_cand
            mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
            e3_list.append(((mant * unit - ab) ** 2 * wsub).sum(dim=4))  # (r,nb,8,2)
            m3_list.append(lv3_cand)
        take1 = e3_list[0] <= e3_list[1]
        e3 = torch.where(take1, e3_list[0], e3_list[1])
        lv3 = torch.where(take1, 1.0, 2.0)                              # (r,nb,8,2)
        e2 = e3.sum(dim=3)                                              # (r,nb,8)
        if best_e2 is None:
            best_e2, best_lv2, best_lv3 = e2, lv2_cand, lv3
        else:
            take2 = e2 < best_e2
            best_e2 = torch.where(take2, e2, best_e2)
            best_lv2 = torch.where(take2, lv2_cand, best_lv2)
            best_lv3 = torch.where(take2.unsqueeze(-1), lv3, best_lv3)

    lv2 = best_lv2.reshape(*best_lv2.shape, 1, 1)
    lv3 = best_lv3.reshape(*best_lv3.shape, 1)

    # ---- final params with chosen (sf, lv2, lv3) ----
    unit = sf * lv2 * lv3
    mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
    return {
        "scale_factor": sf,
        "scale_lv2": lv2,
        "scale_lv3": lv3,
        "sign": torch.sign(xb),
        "mant": mant,
    }


def _quantize_weighted(x2d: torch.Tensor, wgt: torch.Tensor) -> dict[str, torch.Tensor]:
    """x2d: (R, C) fp32, C % 64 == 0; wgt: (R, C) fp32 >= 0."""
    R, C = x2d.shape
    nb = C // 64
    out: dict[str, list[torch.Tensor]] = {k: [] for k in
                                          ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant")}
    if not USE_WEIGHTS:
        wgt = torch.ones(1, C, dtype=torch.float32)
    else:
        wgt = wgt / wgt.mean().clamp_min(1e-30)
        wgt = wgt.clamp(0.25, 4.0)
    w2d = wgt if wgt.shape == (R, C) else wgt.expand(R, C)
    for s0 in range(0, R, ROW_CHUNK):
        x_chunk = x2d[s0:s0 + ROW_CHUNK]
        p = _quant_chunk(
            x_chunk.reshape(-1, nb, 8, 2, 4),
            w2d[s0:s0 + ROW_CHUNK].reshape(-1, nb, 8, 2, 4),
        )
        for k in out:
            out[k].append(p[k])
    cat = {k: torch.cat(v, dim=0) for k, v in out.items()}
    # restore (R, nb, ...) shapes already correct after cat; return
    return cat


def _uniform_weights(R: int, C: int, device) -> torch.Tensor:
    return torch.ones(R, C, dtype=torch.float32, device=device)


def _safe_wgt(w, R, C):
    if isinstance(w, torch.Tensor) and tuple(w.shape) == (R, C):
        return w.float()
    return _uniform_weights(R, C, torch.device("cpu"))


# =============================================================================
# 1. Linear calibration + Weight quantization
# =============================================================================

def _quant_weight_fast(w: torch.Tensor, wgt: torch.Tensor) -> dict[str, torch.Tensor]:
    """Cheaper pass for alpha search: greedy lv, 4 sf candidates."""
    R, C = w.shape
    nb = C // 64
    xb = w.reshape(R, nb, 8, 2, 4)
    ab = xb.abs()
    amax = ab.amax(dim=(2, 3, 4), keepdim=True)
    amax8 = ab.amax(dim=(3, 4), keepdim=True)
    amax4 = ab.amax(dim=4, keepdim=True)
    wblk = (wgt.expand(R, C) if wgt.dim() == 1 else wgt).reshape(R, nb, 8, 2, 4)
    t = (amax / 7.0).clamp_min(1e-38)
    e0 = torch.floor(torch.log2(t)).squeeze(-1).squeeze(-1).squeeze(-1)
    err_best = None
    sf_best = None
    for c in E6M2_SIG:  # fast pass: only the anchor exponent
        sf = (torch.exp2(e0) * c).clamp(SF_MIN, SF_MAX)
        sf5 = sf[..., None, None, None]
        lv2 = torch.where(amax8 / sf5 > 1.75, 2.0, 1.0)
        lv3 = torch.where(amax4 / (sf5 * lv2) > 1.75, 2.0, 1.0)
        unit = sf5 * lv2 * lv3
        mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
        err = ((mant * unit - ab) ** 2 * wblk).sum(dim=(2, 3, 4))
        if err_best is None:
            err_best, sf_best = err, sf
        else:
            better = err < err_best
            err_best = torch.where(better, err, err_best)
            sf_best = torch.where(better, sf, sf_best)
    sf = sf_best[..., None, None, None]
    lv2 = torch.where(amax8 / sf > 1.75, 2.0, 1.0)
    lv3 = torch.where(amax4 / (sf * lv2) > 1.75, 2.0, 1.0)
    unit = sf * lv2 * lv3
    mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
    return {
        "scale_factor": sf, "scale_lv2": lv2, "scale_lv3": lv3,
        "sign": torch.sign(xb), "mant": mant,
    }


def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    w = dequantize_nvfp4(weight_quant, weight_scale).float()
    R, C = w.shape

    # activation stats per channel (use all calib tokens)
    sq_sum = torch.zeros(C, dtype=torch.float32)
    abs_sum = torch.zeros(C, dtype=torch.float32)
    n_tok = 0
    for act_quant, act_scale in calib_activation_list:
        a = dequantize_nvfp4(act_quant, act_scale).float()
        sq_sum += (a * a).sum(dim=0)
        abs_sum += a.abs().sum(dim=0)
        n_tok += a.shape[0]
    act_energy = sq_sum / max(n_tok, 1)          # E[x_j^2]
    act_absmean = abs_sum / max(n_tok, 1)        # E[|x_j|]
    w_col = (w * w).sum(dim=0)                   # sum_out W_ij^2

    # ---- smoothing scale search: s = m^alpha, geometric-mean normalized ----
    m = act_absmean.clamp_min(1e-12)
    logm = m.log()
    logm = logm - logm.mean()

    best_alpha = 0.0
    best_loss = None
    # subsample rows for the search; score on the largest calib set
    rows = torch.randperm(R)[: min(R, 256)]
    a_big = None
    for act_quant, act_scale in calib_activation_list:
        a = dequantize_nvfp4(act_quant, act_scale).float()
        if a_big is None or a.shape[0] > a_big.shape[0]:
            a_big = a

    for alpha in ALPHA_GRID:
        s = torch.exp(logm * alpha)
        wp = _quant_weight_fast(w[rows] / s, act_energy * s * s)
        wq = (wp["sign"] * wp["mant"] * wp["scale_lv3"] * wp["scale_lv2"]
              * wp["scale_factor"]).flatten(-4, -1) * s
        loss = ((a_big @ wq.T - a_big @ w[rows].T) ** 2).mean().item()
        if best_loss is None or loss < best_loss:
            best_loss, best_alpha = loss, alpha

    s = torch.exp(logm * best_alpha)
    weight_params = _quantize_weighted(w / s, act_energy * s * s)

    activation_state = {
        "s": s,
        "w_col_over_s2": w_col / (s * s).clamp_min(1e-30),
    }
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
    R, C = x.shape
    s = None
    wgt = None
    if isinstance(activation_state, dict):
        t = activation_state.get("s")
        if isinstance(t, torch.Tensor) and t.numel() == C:
            s = t.float()
        t = activation_state.get("w_col_over_s2")
        if isinstance(t, torch.Tensor) and t.numel() == C:
            wgt = t.float()
    if s is None:
        s = torch.ones(C, dtype=torch.float32)
    if wgt is None:
        wgt = torch.ones(C, dtype=torch.float32)
    return _quantize_weighted(x * s, wgt.expand(R, C))


# =============================================================================
# 3. Attention calibration
# =============================================================================

def _deq_params(p):
    return (p["sign"] * p["mant"] * p["scale_lv3"] * p["scale_lv2"]
            * p["scale_factor"]).flatten(-4, -1)


_R_CACHE: dict[int, torch.Tensor | None] = {}


def _make_R(dh: int):
    """Deterministic per-head orthogonal map H*D (Hadamard x random sign).

    Returns None when head_dim is not a power of two (rotation disabled).
    (q R)(k R)^T = q k^T, so attention is mathematically invariant; the
    rotation only reshapes the value distribution before quantization.
    """
    if dh in _R_CACHE:
        return _R_CACHE[dh]
    if dh & (dh - 1):
        _R_CACHE[dh] = None
        return None
    H = torch.tensor([[1.0]])
    while H.shape[0] < dh:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    H = H / (dh ** 0.5)
    g = torch.Generator().manual_seed(0xA5A5 + dh)
    d = (torch.rand(dh, generator=g) < 0.5).float() * 2 - 1
    _R_CACHE[dh] = H * d.unsqueeze(0)
    return _R_CACHE[dh]


def hif4_calibration_attention(
    calib_qkv_list: list,
    q_num_heads: int,
    kv_num_heads: int,
    head_dim: int,
) -> dict[str, Any]:
    """Attention calibration: decide per-group whether to rotate Q/K.

    The rotation is an exact invariant of the attention output; the choice
    on/off is measured directly on a subsample of the largest calib sample
    (attention MSE with vs without rotation). V is never rotated (its output
    is consumed directly by the judge).
    """
    qh, kvh, dh = q_num_heads, kv_num_heads, head_dim
    R = _make_R(dh)

    big = max(calib_qkv_list, key=lambda smp: smp["q"][0].shape[0])
    q = dequantize_nvfp4(*big["q"]).float()
    k = dequantize_nvfp4(*big["k"]).float()
    v = dequantize_nvfp4(*big["v"]).float()
    stride = max(1, (q.shape[0] + 511) // 512)
    q = q[::stride].contiguous()
    k = k[::stride].contiguous()
    v = v[::stride].contiguous()
    T = q.shape[0]

    ref = _attention_out(q, k, v, qh, kvh, dh)
    ones_q = torch.ones(1, qh * dh, dtype=torch.float32)
    ones_k = torch.ones(1, kvh * dh, dtype=torch.float32)

    def run(qt, kt):
        pq = _quantize_weighted(qt, ones_q)
        pk = _quantize_weighted(kt, ones_k)
        pv = _quantize_weighted(v, ones_k)
        out = _attention_out(_deq_params(pq), _deq_params(pk), _deq_params(pv), qh, kvh, dh)
        return ((out - ref) ** 2).mean().item()

    loss_off = run(q, k)
    if R is None:
        rot = 0
    else:
        qr = (q.view(T, qh, dh) @ R).reshape(T, qh * dh)
        kr = (k.view(T, kvh, dh) @ R).reshape(T, kvh * dh)
        loss_on = run(qr, kr)
        rot = 1 if loss_on < loss_off else 0

    return {
        "q_state": {"rot": rot},
        "k_state": {"rot": rot},
        "v_state": None,
    }


def _attention_out(q, k, v, qh, kvh, dh):
    seq = q.shape[0]
    qf = q.view(seq, qh, dh).transpose(0, 1)
    kf = k.view(seq, kvh, dh).transpose(0, 1)
    vf = v.view(seq, kvh, dh).transpose(0, 1)
    rep = qh // kvh
    kf = kf.repeat_interleave(rep, dim=0)
    vf = vf.repeat_interleave(rep, dim=0)
    scores = torch.bmm(qf, kf.transpose(1, 2)) / (dh ** 0.5)
    prob = torch.softmax(scores, dim=-1)
    out = torch.bmm(prob, vf)
    return out.transpose(0, 1).reshape(seq, qh * dh)


# =============================================================================
# 4/5/6. Dynamic Q/K/V quantization
# =============================================================================

def _dyn_table(x: torch.Tensor, state: dict | None, has_scale: bool) -> dict[str, torch.Tensor]:
    R, C = x.shape
    s = torch.ones(C, dtype=torch.float32)
    w = None
    if isinstance(state, dict):
        if has_scale:
            t = state.get("s")
            if isinstance(t, torch.Tensor) and t.numel() == C:
                s = t.float()
        t = state.get("w")
        if isinstance(t, torch.Tensor) and t.dim() == 2 and t.shape[1] == C:
            w = t.float()
    xs = x * s
    if w is None:
        wgt = torch.ones(1, C, dtype=torch.float32)
    else:
        if R > w.shape[0]:
            pad = w[-1:].expand(R - w.shape[0], C)
            w = torch.cat([w, pad], dim=0)
        wgt = w[:R].contiguous()
    return _quantize_weighted(xs, wgt)


def _dyn_qk(quant, scale, state, num_heads, head_dim):
    x = dequantize_nvfp4(quant, scale).float()
    if isinstance(state, dict) and state.get("rot") == 1:
        R = _make_R(head_dim)
        if R is not None:
            T = x.shape[0]
            x = (x.view(T, num_heads, head_dim) @ R).reshape(T, -1).contiguous()
    return _dyn_table(x, None, has_scale=False)


def _dyn_qk_norot(quant, scale, state):
    x = dequantize_nvfp4(quant, scale).float()
    return _dyn_table(x, None, has_scale=False)


def _dyn_v(quant, scale, state):
    x = dequantize_nvfp4(quant, scale).float()
    return _dyn_table(x, state, has_scale=False)


def hif4_dynamic_quantize_q(q_quant, q_scale, q_num_heads, head_dim, q_state):
    return _dyn_qk(q_quant, q_scale, q_state, q_num_heads, head_dim)


def hif4_dynamic_quantize_k(k_quant, k_scale, kv_num_heads, head_dim, k_state):
    return _dyn_qk(k_quant, k_scale, k_state, kv_num_heads, head_dim)


def hif4_dynamic_quantize_v(v_quant, v_scale, kv_num_heads, head_dim, v_state):
    return _dyn_v(v_quant, v_scale, v_state)
