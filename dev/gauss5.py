"""Valid optimum + threshold-tuned greedy ranking search.

Q1: true valid optimum over 16-candidate grid with per-sub-block lv2 min.
Q2: can a lowered greedy threshold (t8 for lv2, t4 for lv3) make greedy
    ranking approximate refined ranking at zero extra cost?
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
t = (amax / 7.0).clamp_min(1e-38)
e0 = torch.floor(torch.log2(t)).squeeze(-1).squeeze(-1).squeeze(-1)


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
        best = e2 if best is None else torch.minimum(best, e2)  # per sub-block
    return best.sum(dim=2)


def greedy_err(sf, t8, t4):
    sf5 = sf[..., None, None, None]
    lv2 = torch.where(amax8 / sf5 > t8, 2.0, 1.0)
    lv3 = torch.where(amax4 / (sf5 * lv2) > t4, 2.0, 1.0)
    unit = sf5 * lv2 * lv3
    mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
    return ((mant * unit - ab) ** 2).sum(dim=(2, 3, 4))


def cands(e_offs):
    out = []
    for eo in e_offs:
        pe = torch.exp2(e0 + eo)
        for c in S.E6M2_SIG:
            out.append((pe * c).clamp(S.SF_MIN, S.SF_MAX))
    return out


total = x.numel()
alg1 = 6.922078e-03

# --- Q1: valid optimum on 16-grid
cl16 = cands([-1, 0, 1, 2])
eb = None
sb = None
for sf in cl16:
    e = refined_err(sf)
    if eb is None:
        eb, sb = e, sf
    else:
        m = e < eb
        eb = torch.where(m, e, eb)
        sb = torch.where(m, sf, sb)
print(f"valid opt 16-grid   gain over alg1 {100*(1-eb.sum().item()/total/alg1):5.2f}%")

# --- Q2: threshold-tuned greedy ranking (winner then VALID refined eval)
for t8, t4 in [(1.75, 1.75), (1.5, 1.5), (1.4, 1.4), (1.3, 1.3), (1.2, 1.1),
               (1.1, 1.05), (1.0, 1.0), (1.3, 1.1), (1.2, 1.3), (1.4, 1.2)]:
    eb2 = None
    sb2 = None
    for sf in cl16:
        e = greedy_err(sf, t8, t4)
        if eb2 is None:
            eb2, sb2 = e, sf
        else:
            m = e < eb2
            eb2 = torch.where(m, e, eb2)
            sb2 = torch.where(m, sf, sb2)
    fin = refined_err(sb2).sum().item() / total
    print(f"greedy rank t8={t4:.2f} t4={t4:.2f} -> refined winner gain {100*(1-fin/alg1):5.2f}%")
