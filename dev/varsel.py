"""Per-row variant selection for linear activations via projected output MSE.

Variants: (A) current pipeline quantization; (B) same search on a shifted
candidate grid (ratios x1.25 — one E6M2 step coarser). Selection per row by
||xq @ Wproj - x @ Wproj||^2 with Wproj = W_s^T @ P, P seeded random (N, 64).
Scored against the TRUE output reference x @ W^T on mini tests.
"""
import sys, os, importlib.util
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "example", "solution"))
sys.path.insert(0, ROOT)


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load_mod(os.path.join(ROOT, "..", "example", "solution", "solution.py"), "sol")
import hif4  # noqa: E402

torch.manual_seed(0)
lin = torch.load(os.path.join(ROOT, "..", "example", "mini_sample", "linear.pt"),
                 weights_only=True, map_location="cpu")[0]
wq, ws = lin["weight"]
calib, tests = lin["calib_activation_list"], lin["test_activation_list"]
w_ref = hif4.dequantize_nvfp4(wq, ws).float()

out = S.hif4_calibration_and_quantize_weight(wq, ws, calib)
state = out["activation_state"]
mode, s = state["mode"], state["s"]


def tf(x):
    x = x * s
    if mode == 1:
        return S._rot_blocks(x)
    return x


W_s = tf(w_ref)
N, C = W_s.shape
g = torch.Generator().manual_seed(1234)
P = torch.randn(N, 64, generator=g)
W_proj = (W_s.T @ P).contiguous()          # (C, 64)

# variant B quantizer: shifted candidate grid
GRID_B = tuple((k, sig * 1.25 if sig < 1.75 else (2.0 if k == 0 else 2.5))
               for k, sig in S.CAND_GRID)


def quant_with_grid(x2d, grid):
    R, Cc = x2d.shape
    nb = Cc // 64
    ones = torch.ones(1, Cc)
    out_list = {k: [] for k in ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant")}
    xb = x2d.abs()
    ab = xb.reshape(-1, nb, 8, 2, 4)
    # replicate _quant_chunk with custom grid
    ab_ = ab
    amax = ab_.amax(dim=(2, 3, 4), keepdim=True)
    t = (amax / 7.0).clamp_min(1e-38)
    e0 = torch.floor(torch.log2(t)).squeeze(-1).squeeze(-1).squeeze(-1)
    err_best = sf_best = lv2_best = lv3_best = None
    for k_off, sig in grid:
        sf = (torch.exp2(e0 + k_off) * sig).clamp(S.SF_MIN, S.SF_MAX)
        sf5 = sf[..., None, None, None]
        best_e2 = best_l2 = best_l3 = None
        for lv2_c in (1.0, 2.0):
            e3_list = []
            for lv3_c in (1.0, 2.0):
                unit = sf5 * lv2_c * lv3_c
                mant = torch.clamp(torch.round(ab_ / unit * 4.0) / 4.0, 0.0, 1.75)
                e3_list.append(((mant * unit - ab_) ** 2).sum(dim=4))
            take1 = e3_list[0] <= e3_list[1]
            e3 = torch.where(take1, e3_list[0], e3_list[1])
            lv3 = torch.where(take1, 1.0, 2.0)
            e2 = e3.sum(dim=3)
            if best_e2 is None:
                best_e2, best_l2, best_l3 = e2, lv2_c, lv3
            else:
                take2 = e2 < best_e2
                best_e2 = torch.where(take2, e2, best_e2)
                best_l2 = torch.where(take2, lv2_c, best_l2)
                best_l3 = torch.where(take2.unsqueeze(-1), lv3, best_l3)
        err = best_e2.sum(dim=2)
        if err_best is None:
            err_best, sf_best = err, sf
            lv2_best, lv3_best = best_l2, best_l3
        else:
            take = err < err_best
            take2 = take.unsqueeze(-1)
            take3 = take2.unsqueeze(-1)
            err_best = torch.where(take, err, err_best)
            sf_best = torch.where(take, sf, sf_best)
            lv2_best = torch.where(take2, best_l2, lv2_best)
            lv3_best = torch.where(take3, best_l3, lv3_best)
    out_list = {
        "scale_factor": sf_best[..., None, None, None],
        "scale_lv2": lv2_best[..., None, None],
        "scale_lv3": lv3_best[..., None],
        "sign": torch.sign(x2d.reshape(-1, nb, 8, 2, 4)),
    }
    unit = out_list["scale_factor"] * out_list["scale_lv2"] * out_list["scale_lv3"]
    out_list["mant"] = torch.clamp(
        torch.round(ab_ / unit * 4.0) / 4.0, 0.0, 1.75)
    return out_list


tot_cur = tot_sel = 0.0
for pair in tests:
    x_ref = S.dequantize_nvfp4(*pair).float()
    ref = x_ref @ w_ref.T
    xs = tf(x_ref)
    pA = S.hif4_dynamic_quantize_activation(pair[0], pair[1], state)
    xqA = S._deq_params(pA)
    pB = quant_with_grid(xs, GRID_B)
    xqB = S._deq_params(pB)
    # per-row selection by projected output MSE
    yA = xqA @ W_proj
    yB = xqB @ W_proj
    y0 = xs @ W_proj
    eA = ((yA - y0) ** 2).sum(dim=1)
    eB = ((yB - y0) ** 2).sum(dim=1)
    pickA = eA <= eB
    xq_sel = torch.where(pickA.unsqueeze(1), xqA, xqB)
    # eval against TRUE reference (untransformed product)
    tot_cur += ((xqA @ W_s.T - ref) ** 2).mean().item()
    tot_sel += ((xq_sel @ W_s.T - ref) ** 2).mean().item()
n = len(tests)
print(f"current {tot_cur/n:.4e}  -> selected {tot_sel/n:.4e}  ({100*(1-tot_sel/tot_cur):+.1f}%)")
