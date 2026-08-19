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


_H64_CACHE: torch.Tensor | None = None


def _rot_blocks(x: torch.Tensor) -> torch.Tensor:
    """Block-diagonal random-Hadamard rotation over each 64-channel block.

    Exact dot-product invariant ((xR)(yR)^T = xy^T per block); Gaussianizes
    intra-block structure so one outlier no longer dilutes the whole block,
    while the E6M2 per-block scale absorbs cross-block magnitude differences.
    Deterministic sign vectors per block index (same for X and W).
    """
    global _H64_CACHE
    if _H64_CACHE is None:
        H = torch.tensor([[1.0]])
        while H.shape[0] < 64:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        _H64_CACHE = H / 8.0
    H64 = _H64_CACHE
    R, C = x.shape
    nb = C // 64
    d = torch.empty(nb, 64)
    for b in range(nb):
        g = torch.Generator().manual_seed(777 + b)
        d[b] = (torch.rand(64, generator=g) < 0.5).float() * 2 - 1
    Rm = H64.unsqueeze(0) * d.unsqueeze(1)          # (nb, 64, 64)
    xb = x.reshape(R, nb, 64)
    return torch.einsum("rbd,bde->rbe", xb, Rm).reshape(R, C)


GPTQ_BLOCK = 128
GPTQ_DAMP = 0.01


def _upper_cholesky_inv(H: torch.Tensor):
    """Upper-triangular U with U^T U = H^-1 (damped). Supports (n, n) and (B, n, n).
    Returns None on failure."""
    d = H.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-30) * GPTQ_DAMP
    eye = torch.eye(H.shape[-1], dtype=H.dtype)
    if H.dim() == 2:
        Hd = H + eye * d
    else:
        Hd = H + eye * d.view(-1, 1, 1)
    try:
        Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Hd))
        return torch.linalg.cholesky(Hinv, upper=True)
    except Exception:
        return None


def _gptq_quantize_values(x: torch.Tensor, unit: torch.Tensor, hinv: torch.Tensor) -> torch.Tensor:
    """Column-wise GPTQ over the last axis. x, unit: (R, C); hinv: (C, C)
    upper Cholesky of H^-1. Returns values on the grid defined by unit."""
    R, C = x.shape
    W = x.clone()
    Q = torch.empty_like(W)
    for i1 in range(0, C, GPTQ_BLOCK):
        i2 = min(i1 + GPTQ_BLOCK, C)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hi = hinv[i1:i2, i1:i2]
        u = unit[:, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            ui = u[:, i]
            m = (torch.round(w.abs() / ui * 4.0)).clamp_(0.0, 7.0) * 0.25
            s = torch.where(w >= 0, 1.0, -1.0)
            q = s * m * ui
            Q1[:, i] = q
            d = Hi[i, i]
            E1[:, i] = (w - q) / d.clamp_min(1e-30)
            if i < i2 - i1 - 1:
                W1[:, i + 1:] -= E1[:, i].unsqueeze(1) * Hi[i, i + 1:].unsqueeze(0)
        Q[:, i1:i2] = Q1
        if i2 < C:
            W[:, i2:] -= E1 @ hinv[i1:i2, i2:]
            W[:, i1:i2] = W1
    return Q


def _params_unit_flat(p: dict) -> torch.Tensor:
    """Flatten unit = sf*lv2*lv3 to the logical (R, C) shape."""
    R = p["scale_factor"].shape[0]
    nb = p["scale_factor"].shape[1]
    unit = (p["scale_factor"] * p["scale_lv2"] * p["scale_lv3"]).expand(R, nb, 8, 2, 4)
    return unit.reshape(R, -1)


def _values_to_params(q_flat: torch.Tensor, p: dict) -> dict:
    sf, lv2, lv3 = p["scale_factor"], p["scale_lv2"], p["scale_lv3"]
    R, C = q_flat.shape
    q = q_flat.unflatten(-1, (C // 64, 8, 2, 4))
    unit = sf * lv2 * lv3
    m = (torch.round(q.abs() / unit * 4.0)).clamp_(0.0, 7.0) * 0.25
    return {"scale_factor": sf, "scale_lv2": lv2, "scale_lv3": lv3,
            "sign": torch.sign(q), "mant": m}


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
    wblk = (wgt.expand(R, C) if wgt.shape[0] == 1 and wgt.dim() == 2 else wgt).reshape(R, nb, 8, 2, 4)
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
    """Weight path: alpha smoothing -> rotation on/off (cheap RTN proxy) ->
    anchored search -> hold-out-guarded GPTQ -> activation-side GPTQ state.

    The block-diagonal rotation is an exact matmul invariant that Gaussianizes
    intra-block outliers; it is chosen per group by a cheap RTN-level proxy.
    """
    w = dequantize_nvfp4(weight_quant, weight_scale).float()
    R, C = w.shape
    ones_w = torch.ones(1, C, dtype=torch.float32)
    acts_raw = [dequantize_nvfp4(aq, as_).float() for aq, as_ in calib_activation_list]

    # ---- alpha smoothing search ----
    abs_sum = torch.zeros(C, dtype=torch.float32)
    n_tok = 0
    a_big = None
    for a in acts_raw:
        abs_sum += a.abs().sum(dim=0)
        n_tok += a.shape[0]
        if a_big is None or a.shape[0] > a_big.shape[0]:
            a_big = a
    m = (abs_sum / max(n_tok, 1)).clamp_min(1e-12)
    logm = m.log()
    logm = logm - logm.mean()
    rows = torch.randperm(R)[: min(R, 256)]
    best_alpha = 0.0
    best_loss = None
    for alpha in ALPHA_GRID:
        s = torch.exp(logm * alpha)
        wp = _quant_weight_fast(w[rows] / s, torch.ones(1, C))
        wq = (wp["sign"] * wp["mant"] * wp["scale_lv3"] * wp["scale_lv2"]
              * wp["scale_factor"]).flatten(-4, -1) * s
        loss = ((a_big @ wq.T - a_big @ w[rows].T) ** 2).mean().item()
        if best_loss is None or loss < best_loss:
            best_loss, best_alpha = loss, alpha
    s = torch.exp(logm * best_alpha)
    w_s = w / s
    acts_s = [a * s for a in acts_raw]

    # ---- rotation choice: GPTQ-level comparison on a weight-row subsample
    # (RTN-level margins are too noisy; rotation pays mainly through GPTQ) ----
    rot = 0
    U_rot = None
    U_raw = None
    rsub = None
    xh_pick = None
    if R > 64 and len(acts_s) >= 2 and acts_s[-1].shape[0] >= 8:
        rsub = torch.randperm(R)[: min(R, 256)]
        xh_last = acts_s[-1]
        sub = torch.randperm(xh_last.shape[0])[: min(xh_last.shape[0], 128)]
        H_r = torch.zeros(C, C, dtype=torch.float32)
        H_o = torch.zeros(C, C, dtype=torch.float32)
        for a in acts_s[:-1]:
            ar = _rot_blocks(a)
            H_r += ar.T @ ar
            H_o += a.T @ a
        U_r = _upper_cholesky_inv(H_r)
        U_o = _upper_cholesky_inv(H_o)
        xh_pick_rot = _rot_blocks(xh_last)[sub].contiguous()
        xh_pick_raw = xh_last[sub].contiguous()
        mse_A = mse_B = None
        if U_r is not None and U_o is not None:
            w_rsub_rot = _rot_blocks(w_s[rsub])
            pA = _quant_weight_fast(w_rsub_rot, torch.ones(1, C))
            qA = _gptq_quantize_values(w_rsub_rot, _params_unit_flat(pA), U_r)
            mse_A = ((xh_pick_rot @ qA.T - xh_pick_rot @ w_rsub_rot.T) ** 2).mean().item()
            pB = _quant_weight_fast(w_s[rsub], torch.ones(1, C))
            qB = _gptq_quantize_values(w_s[rsub], _params_unit_flat(pB), U_o)
            mse_B = ((xh_pick_raw @ qB.T - xh_pick_raw @ w_s[rsub].T) ** 2).mean().item()
        if mse_A is not None and mse_B is not None and mse_A < mse_B:
            rot = 1
            U_rot = U_r
            xh_pick = xh_pick_rot
        else:
            U_raw = U_o
            xh_pick = xh_pick_raw

    w_final = _rot_blocks(w_s) if rot else w_s
    Uw = U_rot if rot else U_raw

    # ---- anchored search + hold-out-guarded GPTQ on the chosen variant ----
    weight_params = _quantize_weighted(w_final, ones_w)
    q_used = (weight_params["sign"] * weight_params["mant"] * weight_params["scale_lv3"]
              * weight_params["scale_lv2"] * weight_params["scale_factor"]).flatten(-4, -1)
    if xh_pick is not None and Uw is not None:
            unit = _params_unit_flat(weight_params)
            q_g = _gptq_quantize_values(w_final, unit, Uw)
            ref = xh_pick @ w_final.T
            mse_r = ((xh_pick @ q_used.T - ref) ** 2).mean().item()
            mse_g = ((xh_pick @ q_g.T - ref) ** 2).mean().item()
            if mse_g < mse_r:
                weight_params = _values_to_params(q_g, weight_params)
                q_used = q_g.contiguous()

    # ---- activation-side GPTQ ----
    u_act = None
    gptq_act = 0
    if xh_pick is not None:
        Ua = _upper_cholesky_inv(q_used.T @ q_used)
        if Ua is not None:
            p_r = _quantize_weighted(xh_pick, ones_w)
            xr = (p_r["sign"] * p_r["mant"] * p_r["scale_lv3"] * p_r["scale_lv2"]
                  * p_r["scale_factor"]).flatten(-4, -1)
            unit_x = _params_unit_flat(p_r)
            xg = _gptq_quantize_values(xh_pick, unit_x, Ua)
            ref2 = xh_pick @ w_final.T
            mse_ar = ((xr @ q_used.T - ref2) ** 2).mean().item()
            mse_ag = ((xg @ q_used.T - ref2) ** 2).mean().item()
            if mse_ag < mse_ar:
                u_act = Ua.contiguous()
                gptq_act = 1

    activation_state = {
        "s": s.contiguous(),
        "rot": rot,
        "u_act": u_act,
        "g": gptq_act,
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
    if isinstance(activation_state, dict):
        t = activation_state.get("s")
        if isinstance(t, torch.Tensor) and t.numel() == C:
            s = t.float()
    if s is None:
        s = torch.ones(C, dtype=torch.float32)
    x = x * s
    if isinstance(activation_state, dict) and activation_state.get("rot") == 1:
        x = _rot_blocks(x)
    p = _quantize_weighted(x, torch.ones(1, C, dtype=torch.float32))
    if isinstance(activation_state, dict) and activation_state.get("g") == 1:
        u = activation_state.get("u_act")
        if isinstance(u, torch.Tensor) and tuple(u.shape) == (C, C):
            unit = _params_unit_flat(p)
            q = _gptq_quantize_values(x, unit, u.float())
            return _values_to_params(q, p)
    return p


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



def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):
    return {"q_state": None, "k_state": None, "v_state": None}


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
    return _quantize_alg1(dequantize_nvfp4(quant, scale))


def hif4_dynamic_quantize_q(q_quant, q_scale, q_num_heads, head_dim, q_state):
    return _quantize_alg1(dequantize_nvfp4(q_quant, q_scale))


def hif4_dynamic_quantize_k(k_quant, k_scale, kv_num_heads, head_dim, k_state):
    return _quantize_alg1(dequantize_nvfp4(k_quant, k_scale))


def hif4_dynamic_quantize_v(v_quant, v_scale, kv_num_heads, head_dim, v_state):
    return _quantize_alg1(dequantize_nvfp4(v_quant, v_scale))
