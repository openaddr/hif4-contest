"""Validate a deploy-aware proxy (rotation-aware, act quantized, weight side
down-weighted) against the oracle tau end-to-end curve."""
from __future__ import annotations

import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "dev"))
import exp_smooth as E  # noqa: E402

SOL = E.load_sol()


def j_dep(s, xh, wsub, kappa=0.0):
    """Deploy-aware proxy: for mode in {0,1}: rotate both sides (structure-
    independent fixed rotation), quantize the ACT side with the fast table
    quantizer, weight side quantized with weight kappa (0 = free, GPTQ+
    refinement absorb it).  Score in output space vs unquantized reference.
    Returns min over modes."""
    C = s.shape[0]
    ones = torch.ones(1, C)
    best = None
    for md in (0, 1):
        xs = xh * s
        ws = wsub / s
        if md == 1:
            xs = SOL._rot_blocks(xs)
            ws = SOL._rot_blocks(ws)
        xp = SOL._quant_weight_fast(xs, ones)
        xq = (xp["sign"] * xp["mant"] * xp["scale_lv3"] * xp["scale_lv2"]
              * xp["scale_factor"]).flatten(-4, -1)
        ref = xh @ wsub.T
        err_act = ((xq @ ws.T - ref) ** 2).mean().item()
        if kappa > 0.0:
            wp = SOL._quant_weight_fast(ws, ones)
            wq = (wp["sign"] * wp["mant"] * wp["scale_lv3"] * wp["scale_lv2"]
                  * wp["scale_factor"]).flatten(-4, -1) * s
            err_w = ((xh * s @ wq.T / 1.0 - 0) ** 2)
            refw = xh @ wsub.T
            err_w = (((xh * s) @ wq.T - refw) ** 2).mean().item()
        else:
            err_w = 0.0
        j = err_act + kappa * err_w
        best = j if best is None else min(best, j)
    return best


def main():
    import diag_oracle as D  # noqa: E402  (build() returns gains)
    for seed in (5100, 5231):
        print(f"== seed {seed}")
        g = D.build(seed, 2048, 2048)
        acts = [SOL.dequantize_nvfp4(*p).float() for p in g["calib"]]
        w = SOL.dequantize_nvfp4(*g["weight"]).float()
        xf = torch.cat([acts[0][:80], acts[1][:80]])
        xh = acts[2][::4][:160]
        wsub = w[torch.randperm(w.shape[0])[:192]]
        lg = g["gx"].log(); lg = lg - lg.mean()
        for kap in (0.0, 0.3, 1.0):
            js = []
            for tau in (0.0, 0.25, 0.5, 0.75, 1.0):
                s = (-tau * lg).clamp(-6, 6).exp()
                s = s / torch.exp(s.log().mean())
                js.append(j_dep(s, xh, wsub, kap))
            print(f"   kappa={kap}: " + "  ".join(
                f"tau{t}={j:.3e}" for t, j in zip((0.0, 0.25, 0.5, 0.75, 1.0), js)))


if __name__ == "__main__":
    main()
