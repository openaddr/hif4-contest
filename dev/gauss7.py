"""Anchor study: anchor sf at median sub-block peak vs global peak.

If a median-anchored greedy ranking (~zero extra cost) approaches the
refined optimum, we get the +5pp element-wise gain for free.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "example", "solution"))
import torch
import solution as S

torch.manual_seed(123)
R, C = 4096, 4096
x = torch.randn(R, C)
nb = C // 64
xb = x.reshape(R, nb, 8, 2, 4)
ab = xb.abs()
amax8 = ab.amax(dim=(3, 4), keepdim=True)
amax4 = ab.amax(dim=4, keepdim=True)
amax = ab.amax(dim=(2, 3, 4), keepdim=True)
total = x.numel()
alg1 = 6.922078e-03


def refined_err(sf):
    sf5 = sf[..., None, None, None]
    best = None
    for lv2_c in (1.0, 2.0):
        e3_list = []
        for lv3_c in (1.0, 2.0):
            unit = sf5 * lv2_c * lv3_c
            mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
            e3_list.append(((mant * unit - ab) ** 2).sum(dim=4))
        e3 = torch.where(e3_list[0] <= e3_list[1], e3_list[0], e3_list[1])
        e2 = e3.sum(dim=3)
        best = e2 if best is None else torch.minimum(best, e2)
    return best.sum(dim=2)


def greedy_err(sf):
    sf5 = sf[..., None, None, None]
    lv2 = torch.where(amax8 / sf5 > 1.75, 2.0, 1.0)
    lv3 = torch.where(amax4 / (sf5 * lv2) > 1.75, 2.0, 1.0)
    unit = sf5 * lv2 * lv3
    mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
    return ((mant * unit - ab) ** 2).sum(dim=(2, 3, 4))


def grid_around(anchor, ratios):
    e0 = torch.floor(torch.log2(anchor.clamp_min(1e-38))).squeeze(-1).squeeze(-1).squeeze(-1)
    out = []
    for r in ratios:
        k = int(torch.floor(torch.log2(torch.tensor(r))).item())
        sig = r / (2 ** k)
        out.append((torch.exp2(e0 + k) * sig).clamp(S.SF_MIN, S.SF_MAX))
    return out


def run(name, cands, mode):
    eb = None
    sb = None
    for sf in cands:
        e = refined_err(sf) if mode == "r" else greedy_err(sf)
        if eb is None:
            eb, sb = e, sf
        else:
            m = e < eb
            eb = torch.where(m, e, eb)
            sb = torch.where(m, sf, sb)
    if mode == "g":
        eb = refined_err(sb)
    print(f"{name:34s} n={len(cands)}{mode}  gain {100*(1-eb.sum().item()/total/alg1):5.2f}%")


med8 = amax8.median(dim=2).values  # (R,nb,1,1,1) median over the 8 sub-blocks
anchor_med = (med8 / 1.75).clamp_min(1e-38)
anchor_max = (amax / 7.0).clamp_min(1e-38)

SIG = [1.0, 1.25, 1.5, 1.75]
run("max-anchor greedy-8 (current)", grid_around(anchor_max, SIG + [2 * s for s in SIG]), "g")
run("med-anchor  greedy-8", grid_around(anchor_med, SIG + [2 * s for s in SIG]), "g")
run("med-anchor  greedy-6 [1..1.75,2,2.5]", grid_around(anchor_med, [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]), "g")
run("med-anchor  greedy-4 [1..1.75]", grid_around(anchor_med, SIG), "g")
run("med-anchor  refined-4 [1..1.75]", grid_around(anchor_med, SIG), "r")
run("med-anchor  refined-6", grid_around(anchor_med, [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]), "r")
run("med-anchor  refined-2 [1.25,1.5]", grid_around(anchor_med, [1.25, 1.5]), "r")
run("med-anchor  refined-3 [1.0,1.5,2.0]", grid_around(anchor_med, [1.0, 1.5, 2.0]), "r")
