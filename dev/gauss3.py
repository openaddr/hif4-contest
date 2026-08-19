"""Decompose the 37.6% element-wise headroom: which search gets it cheapest?

Variants (all end with exact per-sub-block lv2 / per-group lv3 refinement):
  V0 current    : 8 cand (e0,e0+1) greedy-lv select, refine winner
  V1 wide-greedy: 12 cand (e0-1..e0+2) greedy-lv select, refine winner
  V2 neighborhood: V0 then +-1,+-2 grid steps around winner with refined-lv eval
  V3 full       : 16 cand refined-lv select (= true optimum on this grid)
  V4 wider      : 20 cand (e0-2..e0+2) refined-lv select
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
amax = ab.amax(dim=(2, 3, 4), keepdim=True)
t = (amax / 7.0).clamp_min(1e-38)
e0 = torch.floor(torch.log2(t)).squeeze(-1).squeeze(-1).squeeze(-1)  # (R,nb)

E6M2 = S.E6M2_SIG


def refined_err(sf):
    """Exact min error over lv tree for given sf grid (R,nb)."""
    sf5 = sf[..., None, None, None]
    best = None
    for lv2_c in (1.0, 2.0):
        e3_list = []
        for lv3_c in (1.0, 2.0):
            unit = sf5 * lv2_c * lv3_c
            mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
            e3_list.append(((mant * unit - ab) ** 2).sum(dim=4))
        take1 = e3_list[0] <= e3_list[1]
        e2 = torch.where(take1, e3_list[0], e3_list[1])
        best = e2 if best is None else torch.minimum(best, e2)
    return best.sum(dim=3).sum(dim=2)


def greedy_err(sf):
    amax8 = ab.amax(dim=(3, 4), keepdim=True)
    amax4 = ab.amax(dim=4, keepdim=True)
    sf5 = sf[..., None, None, None]
    lv2 = torch.where(amax8 / sf5 > 1.75, 2.0, 1.0)
    lv3 = torch.where(amax4 / (sf5 * lv2) > 1.75, 2.0, 1.0)
    unit = sf5 * lv2 * lv3
    mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
    return ((mant * unit - ab) ** 2).sum(dim=(2, 3, 4))


def cands(e_offs):
    out = []
    for eo in e_offs:
        pe = torch.exp2(e0 + eo)
        for c in E6M2:
            out.append((pe * c).clamp(S.SF_MIN, S.SF_MAX))
    return out


def sel_greedy(cl):
    eb = None
    sb = None
    for sf in cl:
        e = greedy_err(sf)
        if eb is None:
            eb, sb = e, sf
        else:
            m = e < eb
            eb = torch.where(m, e, eb)
            sb = torch.where(m, sf, sb)
    return refined_err(sb)


def sel_refined(cl):
    eb = None
    sb = None
    for sf in cl:
        e = refined_err(sf)
        if eb is None:
            eb, sb = e, sf
        else:
            m = e < eb
            eb = torch.where(m, e, eb)
            sb = torch.where(m, sf, sb)
    return eb


total = x.numel()
res = {}
res["V0 current 8g+ref"] = sel_greedy(cands([0, 1]))
res["V1 wide 12g+ref"] = sel_greedy(cands([-1, 0, 1, 2]))
res["V3 full 16r"] = sel_refined(cands([-1, 0, 1, 2]))
res["V4 wider 20r"] = sel_refined(cands([-2, -1, 0, 1, 2]))

# V2: V0 winner, then neighborhood refined re-search
eb = None
sb = None
for sf in cands([0, 1]):
    e = greedy_err(sf)
    if eb is None:
        eb, sb = e, sf
    else:
        m = e < eb
        eb = torch.where(m, e, eb)
        sb = torch.where(m, sf, sb)
# build sorted 16-grid, find index of sb, take +-1 +-2
grid = cands([-1, 0, 1, 2])  # 16 candidates, globally sorted? exp2(e0+eo)*c ascending in (eo,c)
idx = None
for j, sf in enumerate(grid):
    m = sf == sb
    idx = m.long() if idx is None else (idx + m.long()).clamp(max=15)
for d in (-2, -1, 1, 2):
    j = (idx + d).clamp(0, 15)
    take = torch.zeros_like(sb, dtype=torch.bool)
    # gather candidate sf per block via one-hot free approach: use scatter on flattened
    flat = j.reshape(-1)
    sf_sel = torch.stack(grid, 0).reshape(16, -1)[flat, torch.arange(flat.numel())].reshape_as(sb)
    e = refined_err(sf_sel)
    m = e < eb
    eb = torch.where(m, e, eb)
    sb = torch.where(m, sf_sel, sb)
res["V2 neigh 8g+4r"] = eb

# alg1 reference for gain normalization
alg1 = 6.922078e-03
for k, v in res.items():
    e = v.sum().item() / total
    print(f"{k:22s} MSE {e:.6e}  gain over alg1 {100*(1-e/alg1):5.2f}%")
