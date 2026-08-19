"""Cost-effective refined-ranking subsets: gain vs mant-eval count.

V-opt16 = 12.09% (upper bound on this grid). Current = 6.63% (16-greedy)
/ 7.89% (8-greedy V0). We want max gain per mant-eval.
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
        best = e2 if best is None else torch.minimum(best, e2)
    return best.sum(dim=2)


def greedy_err(sf):
    sf5 = sf[..., None, None, None]
    lv2 = torch.where(amax8 / sf5 > 1.75, 2.0, 1.0)
    lv3 = torch.where(amax4 / (sf5 * lv2) > 1.75, 2.0, 1.0)
    unit = sf5 * lv2 * lv3
    mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
    return ((mant * unit - ab) ** 2).sum(dim=(2, 3, 4))


def by_ratio(ratios):
    out = []
    for r in ratios:
        e = e0 + torch.floor(torch.log2(torch.tensor(r))).item()
        c = r / (2 ** (e - e0.item() if isinstance(e, float) else 0))
    # simpler: r = 2^k * sig; handle explicit decomposition
    out = []
    for r in ratios:
        k = int(torch.floor(torch.log2(torch.tensor(r))).item())
        sig = r / (2 ** k)
        out.append((torch.exp2(e0 + k) * sig).clamp(S.SF_MIN, S.SF_MAX))
    return out


total = x.numel()
alg1 = 6.922078e-03


def report(name, sf_list, mode):
    eb = None
    sb = None
    for sf in sf_list:
        e = refined_err(sf) if mode == "r" else greedy_err(sf)
        if eb is None:
            eb, sb = e, sf
        else:
            m = e < eb
            eb = torch.where(m, e, eb)
            sb = torch.where(m, sf, sb)
    if mode == "g":  # refine the greedy winner
        eb = refined_err(sb)
    n_mant = len(sf_list) * (4 if mode == "r" else 1) + (4 if mode == "g" else 4)
    print(f"{name:26s} mant-evals~{n_mant:3d}  gain {100*(1-eb.sum().item()/total/alg1):5.2f}%")


e0c = [0, 1]
cur8 = []
for eo in e0c:
    pe = torch.exp2(e0 + eo)
    for c in S.E6M2_SIG:
        cur8.append((pe * c).clamp(S.SF_MIN, S.SF_MAX))

R4L = by_ratio([1.0, 1.25, 1.5, 1.75])
R5 = by_ratio([0.875, 1.0, 1.25, 1.5, 1.75])
R6 = by_ratio([1.0, 1.25, 1.5, 1.75, 2.0, 2.5])
R7 = by_ratio([0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5])

report("cur8 greedy+ref (baseline)", cur8, "g")
report("R4L refined", R4L, "r")
report("R5 refined", R5, "r")
report("R6 refined", R6, "r")
report("R7 refined", R7, "r")
report("cur8 refined", cur8, "r")

# hybrid: greedy-8 winner + 2 fixed challengers, refined-eval all three
eb = None
sb = None
for sf in cur8:
    e = greedy_err(sf)
    if eb is None:
        eb, sb = e, sf
    else:
        m = e < eb
        eb = torch.where(m, e, eb)
        sb = torch.where(m, sf, sb)
chal = by_ratio([1.25, 1.75])
eb3 = refined_err(sb)
sb3 = sb
for sf in chal:
    e = refined_err(sf)
    m = e < eb3
    eb3 = torch.where(m, e, eb3)
    sb3 = torch.where(m, sf, sb3)
print(f"{'hybrid greedy8+2chal':26s} mant-evals~ 20  gain {100*(1-eb3.sum().item()/total/alg1):5.2f}%")
