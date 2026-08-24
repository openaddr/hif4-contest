"""Measure cross-sample channel-structure sharing properly (clamp before log)."""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location(
    "_sol", os.path.join(os.path.dirname(os.path.abspath(__file__)), "solution.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def chan_m(pair):
    a = S.dequantize_nvfp4(*pair).float()
    return a.abs().mean(dim=0)


def corr(x, y):
    x = x - x.mean(); y = y - y.mean()
    d = (x.norm() * y.norm()).item()
    return (x @ y).item() / d if d > 0 else float("nan")


mini = torch.load(os.path.join(ROOT, "example/mini_sample/linear.pt"),
                  weights_only=True, map_location="cpu")[0]
cal = mini["calib_activation_list"]; tst = mini["test_activation_list"]
cm = [chan_m(p).clamp_min(1e-6).log() for p in cal]
tm = [chan_m(p).clamp_min(1e-6).log() for p in tst]
pool = sum(cm) / len(cm)
print("== mini channel-mean (log) correlations")
print("   calib[i] vs pooled-calib:", [f"{corr(c, pool):+.3f}" for c in cm])
print("   test[i]  vs pooled-calib:", [f"{corr(t, pool):+.3f}" for t in tm])
print("   test[i]  vs calib[4]:    ", [f"{corr(t, cm[4]):+.3f}" for t in tm])
print("   test[i]  vs calib[0](T10):", [f"{corr(t, cm[0]):+.3f}" for t in tm])
pool_t = sum(tm) / len(tm)
print("   test pooled vs calib pooled:", f"{corr(pool_t, pool):+.3f}")

sys.path.insert(0, os.path.join(ROOT, "dev"))
import synth  # noqa: E402
g = synth.make_linear_group(1, 2048, 1024, tokens=(128, 512), spread=0.5)
gm = [chan_m(p).clamp_min(1e-6).log() for p in g["calib_activation_list"]]
gt = [chan_m(p).clamp_min(1e-6).log() for p in g["test_activation_list"]]
gp = sum(gm) / len(gm)
print("== synth_iid: calib vs pooled:", [f"{corr(c, gp):+.3f}" for c in gm])
print("   test vs pooled:", [f"{corr(t, gp):+.3f}" for t in gt])
