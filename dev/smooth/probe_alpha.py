"""Instrument the baseline alpha smoothing: which alpha wins, what s looks like,
and the proxy losses per alpha, on mini + one stress config."""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "dev"))

spec = importlib.util.spec_from_file_location(
    "_sol", os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def probe(name, w_quant, w_scale, calib):
    w = S.dequantize_nvfp4(w_quant, w_scale).float()
    R, C = w.shape
    acts_raw = [S.dequantize_nvfp4(aq, as_).float() for aq, as_ in calib]
    abs_sum = torch.zeros(C)
    for a in acts_raw:
        abs_sum += a.abs().sum(dim=0)
    n_tok = sum(a.shape[0] for a in acts_raw)
    m = (abs_sum / max(n_tok, 1)).clamp_min(1e-12)
    logm = m.log(); logm = logm - logm.mean()
    rows = torch.randperm(R)[:min(R, 256)]
    torch.manual_seed(123)
    rows = torch.randperm(R)[:min(R, 256)]
    w_rows = w[rows]
    a_big = max(acts_raw, key=lambda a: a.shape[0])
    a_wr = a_big @ w_rows.T
    print(f"== {name}: C={C} R={R} ncalib={len(acts_raw)}")
    print(f"   logm range [{logm.min():.3f},{logm.max():.3f}] std {logm.std():.3f}")
    # weight column magnitudes (per-channel rms)
    wcol = w.pow(2).mean(dim=0).sqrt()
    logw = wcol.log(); logw = logw - logw.mean()
    print(f"   logw range [{logw.min():.3f},{logw.max():.3f}] std {logw.std():.3f}")
    # correlation between act channel means and weight column rms
    c1 = torch.corrcoef(torch.stack([logm, logw]))[0, 1].item()
    print(f"   corr(logm, logw) = {c1:+.3f}")
    # per-sample channel-mean correlation with the pooled m (structure sharing)
    with torch.no_grad():
        ms = [a.abs().mean(dim=0).log() for a in acts_raw]
        for i in range(1, len(ms)):
            ms[i] = ms[i] - ms[i].mean()
        base = ms[0] - ms[0].mean()
        cs = [torch.corrcoef(torch.stack([base, x]))[0, 1].item() for x in ms[1:]]
    print(f"   sample-vs-sample channel-mean corr: {[f'{c:+.2f}' for c in cs]}")
    for alpha in (0.0, 0.15, 0.3, 0.5, 0.75, 1.0, -0.25, -0.5):
        s = torch.exp(logm * alpha)
        wp = S._quant_weight_fast(w_rows / s, torch.ones(1, C))
        wq = (wp["sign"] * wp["mant"] * wp["scale_lv3"] * wp["scale_lv2"]
              * wp["scale_factor"]).flatten(-4, -1) * s
        loss = ((a_big @ wq.T - a_wr) ** 2).mean().item()
        rel = ((wq - w_rows) ** 2).sum() / (w_rows ** 2).sum()
        print(f"   alpha={alpha:+.2f}: proxy_loss={loss:.4e} w_rel_mse={rel:.4e} s_range=[{s.min():.3f},{s.max():.3f}]")


mini = torch.load(os.path.join(ROOT, "example", "mini_sample", "linear.pt"),
                  weights_only=True, map_location="cpu")[0]
probe("mini", *mini["weight"], mini["calib_activation_list"])

import synth  # noqa: E402
g = synth.make_linear_group(1, 2048, 1024, tokens=(128, 512), spread=0.5)
probe("synth_iid(spread.5)", *g["weight"], g["calib_activation_list"])
g2 = synth.make_linear_group(5, 2048, 1024, tokens=(128, 512), spread=0.05)
probe("synth_iid(flat.05)", *g2["weight"], g2["calib_activation_list"])
