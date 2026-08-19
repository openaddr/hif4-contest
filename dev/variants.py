"""Quantizer variants used to diagnose the online-score mystery.

Variants:
  greedy   - power-of-2 sf + greedy lv (my original reconstruction)
  norm7    - sf = E6M2(absmax / 7), greedy lv relative to it
  usearch  - E6M2 candidate search + greedy lv, UNWEIGHTED (pure element MSE)
  usearch_lv - E6M2 candidate search + lv refinement, UNWEIGHTED
All share mantissa rounding on |x| with sign kept separately.
"""
from __future__ import annotations

import torch

SF_MIN = 2.0 ** -48
SF_MAX = 49152.0
CANDS_T = torch.tensor(
    (0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75), dtype=torch.float32
)


def _blocks(x):
    C = x.shape[-1]
    return x.reshape(*x.shape[:-1], C // 64, 8, 2, 4)


def _finish(xb, sf, lv2, lv3):
    ab = xb.abs()
    unit = sf * lv2 * lv3
    mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
    return {"scale_factor": sf, "scale_lv2": lv2, "scale_lv3": lv3,
            "sign": torch.sign(xb), "mant": mant}


def deq(p):
    return (p["sign"] * p["mant"] * p["scale_lv3"] * p["scale_lv2"] * p["scale_factor"]).flatten(-4, -1)


def quant_greedy(x):
    xb = _blocks(x)
    ab = xb.abs()
    amax = ab.amax(dim=(2, 3, 4), keepdim=True)
    amax8 = ab.amax(dim=(3, 4), keepdim=True)
    amax4 = ab.amax(dim=4, keepdim=True)
    sf = torch.exp2(torch.floor(torch.log2(amax.clamp_min(1e-38)))).clamp(SF_MIN, SF_MAX)
    lv2 = torch.where(amax8 / sf > 1.75, 2.0, 1.0)
    lv3 = torch.where(amax4 / (sf * lv2) > 1.75, 2.0, 1.0)
    return _finish(xb, sf, lv2, lv3)


def _e6m2_round(t):
    t = t.clamp(SF_MIN, SF_MAX)
    e = torch.floor(torch.log2(t.clamp_min(1e-38)))
    return (torch.round(t / torch.exp2(e) * 4.0) / 4.0 * torch.exp2(e))


def quant_norm7(x):
    xb = _blocks(x)
    ab = xb.abs()
    amax = ab.amax(dim=(2, 3, 4), keepdim=True)
    amax8 = ab.amax(dim=(3, 4), keepdim=True)
    amax4 = ab.amax(dim=4, keepdim=True)
    sf = _e6m2_round(amax / 7.0)
    lv2 = torch.where(amax8 / sf > 1.75, 2.0, 1.0)
    lv3 = torch.where(amax4 / (sf * lv2) > 1.75, 2.0, 1.0)
    return _finish(xb, sf, lv2, lv3)


def _search(x, weights=None, refine_lv=False):
    xb = _blocks(x)
    ab = xb.abs()
    amax = ab.amax(dim=(2, 3, 4), keepdim=True)
    amax8 = ab.amax(dim=(3, 4), keepdim=True)
    amax4 = ab.amax(dim=4, keepdim=True)
    pe = torch.exp2(torch.floor(torch.log2(amax.clamp_min(1e-38))))
    wblk = weights.reshape(*xb.shape) if weights is not None else 1.0
    err_best = None
    idx_best = None
    for i in range(CANDS_T.numel()):
        sf = (pe * CANDS_T[i]).clamp(SF_MIN, SF_MAX)
        lv2 = torch.where(amax8 / sf > 1.75, 2.0, 1.0)
        lv3 = torch.where(amax4 / (sf * lv2) > 1.75, 2.0, 1.0)
        unit = sf * lv2 * lv3
        mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
        err = ((mant * unit - ab) ** 2 * wblk).sum(dim=(2, 3, 4))
        if err_best is None:
            err_best, idx_best = err, torch.zeros_like(err, dtype=torch.int64)
        else:
            better = err < err_best
            err_best = torch.where(better, err, err_best)
            idx_best = torch.where(better, i, idx_best)
    sf = (pe * CANDS_T[idx_best.reshape(*err_best.shape, 1, 1, 1)]).clamp(SF_MIN, SF_MAX)
    if not refine_lv:
        lv2 = torch.where(amax8 / sf > 1.75, 2.0, 1.0)
        lv3 = torch.where(amax4 / (sf * lv2) > 1.75, 2.0, 1.0)
        return _finish(xb, sf, lv2, lv3)
    # lv refinement
    best_e2 = None
    best_lv2 = None
    best_lv3 = None
    for lv2_c in (1.0, 2.0):
        base = sf * lv2_c
        e3_list = []
        for lv3_c in (1.0, 2.0):
            unit = base * lv3_c
            mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
            e3_list.append(((mant * unit - ab) ** 2 * wblk).sum(dim=4))
        take1 = e3_list[0] <= e3_list[1]
        e3 = torch.where(take1, e3_list[0], e3_list[1])
        lv3 = torch.where(take1, 1.0, 2.0)
        e2 = e3.sum(dim=3)
        if best_e2 is None:
            best_e2, best_lv2, best_lv3 = e2, lv2_c, lv3
        else:
            take2 = e2 < best_e2
            best_e2 = torch.where(take2, e2, best_e2)
            best_lv2 = torch.where(take2, lv2_c, best_lv2)
            best_lv3 = torch.where(take2.unsqueeze(-1), lv3, best_lv3)
    lv2 = best_lv2.reshape(*best_lv2.shape, 1, 1)
    lv3 = best_lv3.reshape(*best_lv3.shape, 1)
    return _finish(xb, sf, lv2, lv3)


def quant_usearch(x):
    return _search(x, None, refine_lv=False)


def quant_usearch_lv(x):
    return _search(x, None, refine_lv=True)


def quant_v2(x, weights=None):
    """Search anchored at amax/7: E6M2 grid points in [amax/14, amax/2],
    greedy lv + lv refinement, weighted or plain MSE."""
    xb = _blocks(x)
    ab = xb.abs()
    amax = ab.amax(dim=(2, 3, 4), keepdim=True)
    amax8 = ab.amax(dim=(3, 4), keepdim=True)
    amax4 = ab.amax(dim=4, keepdim=True)
    t = (amax / 7.0).clamp_min(1e-38)
    e0 = torch.floor(torch.log2(t))
    wblk = weights.reshape(*xb.shape) if weights is not None else 1.0

    best = None
    best_sf = None
    for e_off in (-1, 0, 1):
        pe = torch.exp2(e0.squeeze(-1).squeeze(-1).squeeze(-1) + e_off)  # (R, nb)
        for c in (1.0, 1.25, 1.5, 1.75):
            sf = (pe * c).clamp(SF_MIN, SF_MAX)                      # (R, nb)
            sf5 = sf[..., None, None, None]
            lv2 = torch.where(amax8 / sf5 > 1.75, 2.0, 1.0)
            lv3 = torch.where(amax4 / (sf5 * lv2) > 1.75, 2.0, 1.0)
            unit = sf5 * lv2 * lv3
            mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
            err = ((mant * unit - ab) ** 2 * wblk).sum(dim=(2, 3, 4))
            if best is None:
                best, best_sf = err, sf
            else:
                take = err < best
                best = torch.where(take, err, best)
                best_sf = torch.where(take, sf, best_sf)
    sf = best_sf[..., None, None, None]
    best_e2 = None
    best_lv2 = None
    best_lv3 = None
    for lv2_c in (1.0, 2.0):
        base = sf * lv2_c
        e3_list = []
        for lv3_c in (1.0, 2.0):
            unit = base * lv3_c
            mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
            e3_list.append(((mant * unit - ab) ** 2 * wblk).sum(dim=4))
        take1 = e3_list[0] <= e3_list[1]
        e3 = torch.where(take1, e3_list[0], e3_list[1])
        lv3 = torch.where(take1, 1.0, 2.0)
        e2 = e3.sum(dim=3)
        if best_e2 is None:
            best_e2, best_lv2, best_lv3 = e2, lv2_c, lv3
        else:
            take2 = e2 < best_e2
            best_e2 = torch.where(take2, e2, best_e2)
            best_lv2 = torch.where(take2, lv2_c, best_lv2)
            best_lv3 = torch.where(take2.unsqueeze(-1), lv3, best_lv3)
    lv2 = best_lv2.reshape(*best_lv2.shape, 1, 1)
    lv3 = best_lv3.reshape(*best_lv3.shape, 1)
    return _finish(xb, sf, lv2, lv3)


def quant_v2n(x, weights=None, exp_offs=(0, 1), sigs=(1.0, 1.25, 1.5, 1.75), refine=True):
    """Parameterized anchored search for speed/quality tuning."""
    xb = _blocks(x)
    ab = xb.abs()
    amax = ab.amax(dim=(2, 3, 4), keepdim=True)
    amax8 = ab.amax(dim=(3, 4), keepdim=True)
    amax4 = ab.amax(dim=4, keepdim=True)
    t = (amax / 7.0).clamp_min(1e-38)
    e0 = torch.floor(torch.log2(t)).squeeze(-1).squeeze(-1).squeeze(-1)
    wblk = weights.reshape(*xb.shape) if weights is not None else 1.0
    best = None
    best_sf = None
    for e_off in exp_offs:
        pe = torch.exp2(e0 + e_off)
        for c in sigs:
            sf = (pe * c).clamp(SF_MIN, SF_MAX)
            sf5 = sf[..., None, None, None]
            lv2 = torch.where(amax8 / sf5 > 1.75, 2.0, 1.0)
            lv3 = torch.where(amax4 / (sf5 * lv2) > 1.75, 2.0, 1.0)
            unit = sf5 * lv2 * lv3
            mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
            err = ((mant * unit - ab) ** 2 * wblk).sum(dim=(2, 3, 4))
            if best is None:
                best, best_sf = err, sf
            else:
                take = err < best
                best = torch.where(take, err, best)
                best_sf = torch.where(take, sf, best_sf)
    sf = best_sf[..., None, None, None]
    if not refine:
        lv2 = torch.where(amax8 / sf > 1.75, 2.0, 1.0)
        lv3 = torch.where(amax4 / (sf * lv2) > 1.75, 2.0, 1.0)
        return _finish(xb, sf, lv2, lv3)
    best_e2 = None
    best_lv2 = None
    best_lv3 = None
    for lv2_c in (1.0, 2.0):
        base = sf * lv2_c
        e3_list = []
        for lv3_c in (1.0, 2.0):
            unit = base * lv3_c
            mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
            e3_list.append(((mant * unit - ab) ** 2 * wblk).sum(dim=4))
        take1 = e3_list[0] <= e3_list[1]
        e3 = torch.where(take1, e3_list[0], e3_list[1])
        lv3 = torch.where(take1, 1.0, 2.0)
        e2 = e3.sum(dim=3)
        if best_e2 is None:
            best_e2, best_lv2, best_lv3 = e2, lv2_c, lv3
        else:
            take2 = e2 < best_e2
            best_e2 = torch.where(take2, e2, best_e2)
            best_lv2 = torch.where(take2, lv2_c, best_lv2)
            best_lv3 = torch.where(take2.unsqueeze(-1), lv3, best_lv3)
    return _finish(xb, sf, best_lv2.reshape(*best_lv2.shape, 1, 1), best_lv3.reshape(*best_lv3.shape, 1))


_E6M2_MIN = 2.0 ** (-48)
_E6M2_MAX = 49152.0


def _encode_e6m2(x):
    xc = x.clamp(min=1e-30)
    e = torch.floor(torch.log2(xc))
    m = torch.round(xc * (2.0 ** (2 - e)))
    m = torch.clamp(m, 4, 8)
    out = m * (2.0 ** (e - 2))
    return torch.clamp(out, _E6M2_MIN, _E6M2_MAX)


def quant_alg1(x):
    """EXACT paper Algorithm 1 -- the judge's standard baseline (probe scored 0)."""
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
    x_scaled = (xr * rec[..., None, None, None]
                * (2.0 ** (-e1_8[..., None, None]))
                * (2.0 ** (-e1_16[..., None])))
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
