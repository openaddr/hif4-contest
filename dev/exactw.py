"""Is the NVFP4 input grid exploitable? Compare element-wise search quality:
  R6 (current candidates, ratio<=2.5)  vs  WIDE (all E6M2, ratio 0.5..8)
on REAL mini data (NVFP4-dequantized) — linear acts, W, and attention q/k/v.
Also report where the wide-grid winner sits (inside vs outside R6 range).
"""
import sys, os, importlib.util
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "example", "solution"))


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load_mod(os.path.join(ROOT, "..", "example", "solution", "solution.py"), "sol")

torch.manual_seed(0)


def refined_err(sf, ab):
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


def study(x, name):
    R, C = x.shape
    nb = C // 64
    xb = x.reshape(R, nb, 8, 2, 4)
    ab = xb.abs()
    amax = ab.amax(dim=(2, 3, 4), keepdim=True)
    t = (amax / 7.0).clamp_min(1e-38)
    e0 = torch.floor(torch.log2(t)).squeeze(-1).squeeze(-1).squeeze(-1)

    # R6: current
    best6 = None
    sf6 = None
    for k, sig in S.CAND_GRID:
        sf = (torch.exp2(e0 + k) * sig).clamp(S.SF_MIN, S.SF_MAX)
        e = refined_err(sf, ab)
        if best6 is None:
            best6, sf6 = e, sf
        else:
            m = e < best6
            best6 = torch.where(m, e, best6)
            sf6 = torch.where(m, sf, sf6)

    # WIDE: every E6M2 value with ratio in [0.5, 8] (e_off -1..+3)
    bestw = None
    sfw = None
    for eo in (-1, 0, 1, 2, 3):
        pe = torch.exp2(e0 + eo)
        for sig in S.E6M2_SIG:
            sf = (pe * sig).clamp(S.SF_MIN, S.SF_MAX)
            e = refined_err(sf, ab)
            if bestw is None:
                bestw, sfw = e, sf
            else:
                m = e < bestw
                bestw = torch.where(m, e, bestw)
                sfw = torch.where(m, sf, sfw)

    total = x.numel()
    e6 = best6.sum().item() / total
    ew = bestw.sum().item() / total
    ratio = sfw / torch.exp2(e0)
    outside = ((ratio < 0.999) | (ratio > 2.51)).float().mean().item()
    zeroish = (bestw < 1e-12).float().mean().item()
    print(f"{name:12s} R6 {e6:.4e}  wide {ew:.4e}  wide/R6 {ew/e6:.3f}  "
          f"winner outside R6: {100*outside:.1f}%  exact blocks: {100*zeroish:.2f}%")


lin = torch.load(os.path.join(ROOT, "..", "example", "mini_sample", "linear.pt"),
                 weights_only=True, map_location="cpu")[0]
xact = S.dequantize_nvfp4(*lin["calib_activation_list"][2]).float()
w = S.dequantize_nvfp4(*lin["weight"]).float()
study(xact, "lin act")
study(w, "lin W")
xt = S.dequantize_nvfp4(*lin["test_activation_list"][3]).float()
study(xt, "lin test act")

at = torch.load(os.path.join(ROOT, "..", "example", "mini_sample", "attn.pt"),
                weights_only=True, map_location="cpu")[0]
for nm in ("q", "k", "v"):
    xt2 = S.dequantize_nvfp4(*at["test"][3][nm]).float()
    study(xt2, f"attn {nm}")
