"""Pure-Gaussian element-wise study: how close is our _quant_chunk to the
per-block optimum, and how much does alg1 (judge baseline) leave behind?

If current-vs-optimal gap is < 1% there is nothing left on the element-wise
side for attention (judge attn data ~ per-block Gaussian -> transfers ~100%).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "example", "solution"))
import torch
import solution as S

torch.manual_seed(123)
R, C = 4096, 4096
x = torch.randn(R, C)


def quant_alg1(x2d):
    """Judge baseline: paper Algorithm 1 exactly."""
    R2, C2 = x2d.shape
    xb2 = x2d.reshape(R2, C2 // 64, 8, 2, 4)
    ab2 = xb2.abs()
    amax = ab2.amax(dim=(2, 3, 4), keepdim=True)
    sf = (amax / 7.0).clamp(S.SF_MIN, S.SF_MAX)
    # E6M2 quantize: 2^-48..49152 grid, sigs {1,1.25,1.5,1.75}
    e = torch.floor(torch.log2(sf.clamp_min(1e-38)))
    pe = torch.exp2(e)
    best = None
    for eo in (0.0, 1.0):
        for c in S.E6M2_SIG:
            cand = (pe * c).clamp(S.SF_MIN, S.SF_MAX)
            d = (cand - sf).abs()
            if best is None:
                best = cand
                bd = d
            else:
                m = d < bd
                best = torch.where(m, cand, best)
                bd = torch.where(m, d, bd)
    sf = best
    sf5 = sf
    amax8 = ab2.amax(dim=(3, 4), keepdim=True)
    amax4 = ab2.amax(dim=4, keepdim=True)
    lv2 = torch.where(amax8 / sf5 >= 4.0, 2.0, 1.0)
    lv3 = torch.where(amax4 / (sf5 * lv2) >= 2.0, 2.0, 1.0)
    unit = sf5 * lv2 * lv3
    mant = torch.clamp(torch.floor(ab2 / unit * 4.0 + 0.5) / 4.0, 0.0, 1.75)
    return {"scale_factor": sf, "scale_lv2": lv2, "scale_lv3": lv3,
            "sign": torch.sign(xb2), "mant": mant}
nb = C // 64
xb = x.reshape(R, nb, 8, 2, 4)
ones = torch.ones(1, C)


def deq(p):
    return (p["sign"] * p["mant"] * p["scale_lv3"] * p["scale_lv2"]
            * p["scale_factor"]).flatten(-4, -1)


def mse(q):
    return ((q - x) ** 2).mean().item()


# --- current pipeline ---
p_cur = S._quantize_weighted(x, ones)
e_cur = mse(deq(p_cur))

# --- alg1 judge baseline ---
p_alg1 = quant_alg1(x)
e_alg1 = mse(deq(p_alg1))

# --- per-block exhaustive optimum (sf over wide E6M2 grid x exact lv tree) ---
ab = xb.abs()
amax = ab.amax(dim=(2, 3, 4), keepdim=True)
amax8 = ab.amax(dim=(3, 4), keepdim=True)
amax4 = ab.amax(dim=4, keepdim=True)

# wide E6M2 candidate set: exps e0-1..e0+2 x 4 sigs = 16 candidates
t = (amax / 7.0).clamp_min(1e-38)
e0 = torch.floor(torch.log2(t)).squeeze(-1).squeeze(-1).squeeze(-1)  # (R,nb)

best_err = None
for e_off in (-1, 0, 1, 2):
    pe = torch.exp2(e0 + e_off)
    for c in S.E6M2_SIG:
        sf = (pe * c).clamp(S.SF_MIN, S.SF_MAX)
        sf5 = sf[..., None, None, None]
        # exact per-sub-block lv2 x per-group lv3 (same structure as stage 2)
        best_e2 = None
        best_err_blk = None
        for lv2_c in (1.0, 2.0):
            base = sf5 * lv2_c
            e3_list = []
            for lv3_c in (1.0, 2.0):
                unit = base * lv3_c
                mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
                e3_list.append(((mant * unit - ab) ** 2).sum(dim=4))
            take1 = e3_list[0] <= e3_list[1]
            e3 = torch.where(take1, e3_list[0], e3_list[1])
            e2 = e3.sum(dim=3)
            if best_e2 is None:
                best_e2 = e2
            else:
                best_e2 = torch.minimum(best_e2, e2)
        err = best_e2  # (R,nb)
        if best_err is None:
            best_err = err
        else:
            best_err = torch.minimum(best_err, err)
e_opt = best_err.sum().item() / x.numel()

print(f"alg1      MSE {e_alg1:.6e}  (baseline)")
print(f"current   MSE {e_cur:.6e}  gain over alg1 {100*(1-e_cur/e_alg1):.2f}%")
print(f"optimal   MSE {e_opt:.6e}  gain over alg1 {100*(1-e_opt/e_alg1):.2f}%")
print(f"current vs optimal gap: {100*(e_cur/e_opt-1):.2f}%")
