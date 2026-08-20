"""Shared harness for the final-squeeze offline experiments (A/B/C).

Loads example/solution/solution.py via importlib (like dev/varsel.py) and
scores the mini linear group exactly like dev/diag3.py: baseline =
dev.variants.quant_alg1 (exact paper Algorithm 1) on both weight and
activation; score per test = (mse_std - mse_play) / mse_std.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

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
import variants as V  # noqa: E402

LIN = torch.load(os.path.join(ROOT, "..", "example", "mini_sample", "linear.pt"),
                 weights_only=True, map_location="cpu")[0]
W_REF = hif4.dequantize_nvfp4(*LIN["weight"])


def std_baseline():
    """Precompute the standard (quant_alg1) weight + per-test activation MSEs."""
    w_std = V.deq(V.quant_alg1(W_REF.float()))
    std_mses = []
    for pair in LIN["test_activation_list"]:
        x_ref = hif4.dequantize_nvfp4(*pair)
        ref = hif4.linear_ref(x_ref, W_REF)
        x_std = V.deq(V.quant_alg1(x_ref.float()))
        std_mses.append(((hif4.linear_ref(x_std, w_std) - ref) ** 2).mean().item())
    return std_mses


def score(weight_params, activation_state, std_mses):
    """Score a (weight_params, activation_state) pair on the 5 mini tests."""
    w_play = hif4.hif4_dequantize(weight_params)
    scores = []
    for pair, mse_std in zip(LIN["test_activation_list"], std_mses):
        x_ref = hif4.dequantize_nvfp4(*pair)
        ref = hif4.linear_ref(x_ref, W_REF)
        p = S.hif4_dynamic_quantize_activation(pair[0], pair[1], activation_state)
        x_play = hif4.hif4_dequantize(p)
        mse_play = ((hif4.linear_ref(x_play, w_play) - ref) ** 2).mean().item()
        scores.append((mse_std - mse_play) / mse_std)
    return scores


def run_pipeline(weight=None, calib=None):
    """Timed run of S.hif4_calibration_and_quantize_weight on the mini group."""
    if weight is None:
        weight = LIN["weight"]
    if calib is None:
        calib = LIN["calib_activation_list"]
    torch.manual_seed(0)
    t0 = time.perf_counter()
    out = S.hif4_calibration_and_quantize_weight(weight[0], weight[1], calib)
    dt = time.perf_counter() - t0
    return out, dt


BASELINE_SCORES = (0.8209, 0.8134, 0.8540, 0.8470, 0.8570)


def report(tag, scores, dt=None, base=None):
    base = base if base is not None else BASELINE_SCORES
    m = sum(scores) / len(scores)
    mb = sum(base) / len(base)
    diffs = [s - b for s, b in zip(scores, base)]
    line = " ".join(f"{s:+.4f}" for s in scores)
    dl = " ".join(f"{d:+.4f}" for d in diffs)
    print(f"[{tag}] 5-test: {line}  mean {m:+.4f}")
    print(f"[{tag}] vs base: {dl}  mean_diff {m - mb:+.4f} (pp: {(m - mb) * 100:+.2f})")
    if dt is not None:
        print(f"[{tag}] calibration time: {dt:.2f}s")
    return m
