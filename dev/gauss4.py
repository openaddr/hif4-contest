"""Reconcile gauss2 vs gauss3 on a few blocks: valid (per-sub-block lv2) vs
invalid (per-group lv2) aggregation, same 16-candidate grid."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "example", "solution"))
import torch
import solution as S

torch.manual_seed(123)
R, C = 64, 64
x = torch.randn(R, C)
nb = 1
xb = x.reshape(R, nb, 8, 2, 4)
ab = xb.abs()
amax = ab.amax(dim=(2, 3, 4), keepdim=True)
t = (amax / 7.0).clamp_min(1e-38)
e0 = torch.floor(torch.log2(t)).reshape(R)


def err_valid(sf5):
    """min over lv2 (per sub-block) of sum over groups of min over lv3."""
    best = None
    for lv2_c in (1.0, 2.0):
        e3_list = []
        for lv3_c in (1.0, 2.0):
            unit = sf5 * lv2_c * lv3_c
            mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
            e3_list.append(((mant * unit - ab) ** 2).sum(dim=4))  # (R,1,8,2)
        e3 = torch.where(e3_list[0] <= e3_list[1], e3_list[0], e3_list[1])
        e2 = e3.sum(dim=3)                                          # (R,1,8)
        best = e2 if best is None else torch.minimum(best, e2)
    return best.sum(dim=(1, 2))                                     # (R,)


def err_groupmin(sf5):
    """INVALID: min over lv2 independently per group."""
    best = None
    for lv2_c in (1.0, 2.0):
        e3_list = []
        for lv3_c in (1.0, 2.0):
            unit = sf5 * lv2_c * lv3_c
            mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
            e3_list.append(((mant * unit - ab) ** 2).sum(dim=4))
        e3 = torch.where(e3_list[0] <= e3_list[1], e3_list[0], e3_list[1])
        best = e3 if best is None else torch.minimum(best, e3)
    return best.sum(dim=(1, 2, 3))


best_v = None
best_g = None
sf_best_v = None
for eo in (-1, 0, 1, 2):
    pe = torch.exp2(e0 + eo)
    for c in S.E6M2_SIG:
        sf = (pe * c).clamp(S.SF_MIN, S.SF_MAX).reshape(R, 1, 1, 1, 1)
        ev = err_valid(sf)
        eg = err_groupmin(sf)
        if best_v is None:
            best_v, best_g, sf_best_v = ev, eg, sf.reshape(R)
        else:
            m = ev < best_v
            best_v = torch.where(m, ev, best_v)
            sf_best_v = torch.where(m, sf.reshape(R), sf_best_v)
            best_g = torch.minimum(best_g, eg)

print("valid   per-block mean err:", best_v.mean().item() / 64)
print("groupmin per-block mean err:", best_g.mean().item() / 64)
print("sf/e0 distribution of valid winner:")
rat = sf_best_v / torch.exp2(e0)
import collections
print(collections.Counter(rat.tolist()).most_common(20))
