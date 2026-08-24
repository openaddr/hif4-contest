"""Mechanism diagnostic: oracle channel-flattening s vs ICM fit vs baseline.

Decodes whether free-form smoothing has ANY juice in the deployed pipeline
(rotation + GPTQ + refinement re-chosen per s), using the TRUE shared channel
gains as an upper bound on what any calibration fit could know."""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "dev"))
import hif4  # noqa: E402
import variants as V  # noqa: E402
import exp_smooth as E  # noqa: E402

SOL = E.load_sol()
E2M1 = E.E2M1_GRID


def build(seed, N, C, spread=0.5, w_spread=0.3, share=1.0):
    gen = torch.Generator().manual_seed(seed)
    gx = E._gains(C, spread, gen)
    gw = E._gains(C, w_spread, gen)
    gx2 = E._gains(C, spread, gen)
    if share < 1.0:
        lg = gx.log(); lg2 = gx2.log()
        lg_t = share * (lg - lg.mean()) + math.sqrt(1 - share ** 2) * (lg2 - lg2.mean())
        gx_t = (lg_t + lg.mean()).exp()
    else:
        gx_t = gx

    def make_act(T, gains):
        x = (torch.randn(T, 1, generator=gen) * gains.unsqueeze(0)
             * torch.randn(T, C, generator=gen))
        return E._nvfp4_pair(x)

    w = (torch.randn(N, 1, generator=gen) * gw.unsqueeze(0)
         * torch.randn(N, C, generator=gen)) * 0.05
    return {"weight": E._nvfp4_pair(w),
            "calib": [make_act(T, gx) for T in (128, 512, 512)],
            "test": [make_act(T, gx_t) for T in (128, 512)],
            "gx": gx, "gx_test": gx_t}


def run(group, s_vec, tag):
    """Full pipeline with s forced; returns mean pp score + diagnostics."""
    orig = SOL._freeform_s
    SOL._freeform_s = lambda acts, w, s_base, logm: s_vec.contiguous()
    SOL.SMOOTH_MODE = "ff_icm"
    SOL.SMOOTH_GUARD = False
    SOL.SMOOTH_DEBUG.clear()
    torch.manual_seed(0)
    t0 = time.perf_counter()
    cal = SOL.hif4_calibration_and_quantize_weight(*group["weight"], group["calib"])
    dt = time.perf_counter() - t0
    SOL._freeform_s = orig
    SOL.SMOOTH_MODE = "base"
    SOL.SMOOTH_GUARD = True
    w_ref = hif4.dequantize_nvfp4(*group["weight"])
    w_std = V.deq(V.quant_alg1(w_ref.float()))
    w_play = hif4.hif4_dequantize(cal["weight_params"])
    scores = []
    for pair in group["test"]:
        x_ref = hif4.dequantize_nvfp4(*pair)
        ref = hif4.linear_ref(x_ref, w_ref)
        x_std = V.deq(V.quant_alg1(x_ref.float()))
        mse_std = ((hif4.linear_ref(x_std, w_std) - ref) ** 2).mean().item()
        p = SOL.hif4_dynamic_quantize_activation(pair[0], pair[1], cal["activation_state"])
        mse_play = ((hif4.linear_ref(hif4.hif4_dequantize(p), w_play) - ref) ** 2).mean().item()
        scores.append((mse_std - mse_play) / mse_std * 100.0)
    mode = cal["activation_state"]["mode"]
    print(f"  {tag:24s} mean={sum(scores)/len(scores):+7.2f}pp mode={mode} "
          f"cal={dt:.1f}s s[{s_vec.min():.2f},{s_vec.max():.2f}] "
          f"logstd={s_vec.log().std():.2f}")
    return sum(scores) / len(scores)


def fit_icm(group):
    acts = [SOL.dequantize_nvfp4(*p).float() for p in group["calib"]]
    w = SOL.dequantize_nvfp4(*group["weight"]).float()
    xf = torch.cat([acts[0][:64], acts[1][:96]], dim=0)
    wsub = w[torch.randperm(w.shape[0])[:192]]
    gw_col = (w * w).sum(0) + 1e-30
    gx_col = (xf * xf).sum(0) + 1e-30
    return SOL._icm_search(xf, wsub, torch.ones(w.shape[1]), gw_col, gx_col)


def proxy(group, s_vec):
    acts = [SOL.dequantize_nvfp4(*p).float() for p in group["calib"]]
    w = SOL.dequantize_nvfp4(*group["weight"]).float()
    xf = torch.cat([acts[0][:80], acts[1][:80]])
    wsub = w[torch.randperm(w.shape[0])[:192]]
    xh = acts[2][::4][:160]
    return SOL._joint_proxy(s_vec, xf, wsub), SOL._joint_proxy(s_vec, xh, wsub)


def main():
    for seed in (5100, 5231):
        print(f"== shared C=2048 spread0.5 seed={seed}")
        g = build(seed, 2048, 2048)
        C = 2048
        lg = g["gx"].log(); lg = lg - lg.mean()
        base = None
        cands = [("s=1 (base)", torch.ones(C))]
        for tau in (0.25, 0.5, 0.75, 1.0):
            s = (-tau * lg).clamp(-6, 6).exp()
            s = s / torch.exp(s.log().mean())
            cands.append((f"oracle gx^-{tau}", s))
        s_icm = fit_icm(g)
        cands.append(("icm fit", s_icm))
        for tag, sv in cands:
            sc = run(g, sv, tag)
            if tag.startswith("s=1"):
                base = sc
        # proxy diagnostics
        print("   proxy (fit, hold):")
        for tag, sv in cands:
            jf, jh = proxy(g, sv)
            print(f"     {tag:22s} fit={jf:.4e} hold={jh:.4e}")


if __name__ == "__main__":
    main()
