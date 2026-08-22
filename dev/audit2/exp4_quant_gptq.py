"""E-C2: _quant_chunk vs _quant_chunk_vec (KB=2/4) below the 4M threshold.
E-6b: in-place torch GPTQ column loop for R > 2048 (weight GPTQ).
"""
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
from exp2_refine import med  # noqa: E402

sol = harness.load_variant()


def qc2(xb, wb, grid):
    return sol._quant_chunk_vec(xb, wb, grid)


def e_c2():
    print("=== C2: _quant_chunk (plain) vs _quant_chunk_vec KB2/KB4 ===")
    print(f"{'R':>6s} {'C':>6s} {'cand':>5s} {'plain':>8s} {'vec2':>8s} {'vec4':>8s} "
          f"{'best':>6s} ident")
    torch.manual_seed(21)
    cases = [
        # dynamic-activation shaped (6-cand grid)
        (10, 2048, 6), (128, 2048, 6), (256, 2048, 6), (512, 2048, 6),
        (1024, 2048, 6), (512, 4096, 6), (256, 8192, 6), (1024, 1024, 6),
        (2048, 1024, 6), (512, 1024, 6),
        # weight shaped (16-cand grid)
        (1024, 512, 16), (2048, 512, 16), (1024, 1024, 16), (2048, 1024, 16),
        (2048, 2048, 16), (1024, 4096, 16),
    ]
    for R, C, nc in cases:
        grid = sol.CAND_GRID if nc == 6 else sol.CAND_GRID_W
        xb = (torch.randn(R, C // 64, 8, 2, 4) * 3)
        wb = torch.rand(C // 64, 8, 2, 4) + 0.1
        a = sol._quant_chunk(xb, wb, grid)
        b = sol._quant_chunk_vec(xb, wb, grid)
        ok = all(torch.equal(a[k], b[k]) for k in a)
        tp = med(lambda: sol._quant_chunk(xb, wb, grid))
        t2 = med(lambda: sol._quant_chunk_vec(xb, wb, grid))
        old = sol.KB if hasattr(sol, "KB") else None
        # KB4 variant: temporarily monkeypatch by slicing grid in twos is not
        # trivial; approximate KB4 by calling the KB=2 impl twice as measured
        # proxy is invalid -> skip KB4 (round-1 showed KB2 best at big sizes).
        best = "plain" if tp <= t2 else "vec2"
        print(f"{R:6d} {C:6d} {nc:5d} {tp:8.3f} {t2:8.3f} {'-':>8s} "
              f"{best:>6s} {ok}")
        sys.stdout.flush()


def _gptq_torch_ip(x, unit, hinv):
    """In-place twin of _gptq_quantize_values_torch: identical op sequence,
    allocations removed (abs/div_/mul_/round_/clamp_ chain; q built inside the
    `s` buffer; E via (w-q).div_)."""
    R, C = x.shape
    GB = sol.GPTQ_BLOCK
    W = x.clone()
    Q = torch.empty_like(W)
    for i1 in range(0, C, GB):
        i2 = min(i1 + GB, C)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hi = hinv[i1:i2, i1:i2]
        u = unit[:, i1:i2]
        last = i2 - i1 - 1
        for i in range(i2 - i1):
            w = W1[:, i]
            ui = u[:, i]
            m = w.abs()
            m.div_(ui)
            m.mul_(4.0)
            m.round_()
            m.clamp_(0.0, 7.0)
            m.mul_(0.25)
            s = torch.where(w >= 0, 1.0, -1.0)
            s.mul_(m)
            s.mul_(ui)                       # s now holds q
            Q1[:, i] = s
            d = Hi[i, i].clamp_min(1e-30)
            E1[:, i] = (w - s).div_(d)
            if i < last:
                W1[:, i + 1:] -= E1[:, i].unsqueeze(1) * Hi[i, i + 1:].unsqueeze(0)
        Q[:, i1:i2] = Q1
        if i2 < C:
            W[:, i2:] -= E1 @ hinv[i1:i2, i2:]
            W[:, i1:i2] = W1
    return Q


def e_ip():
    print("\n=== 6b: torch GPTQ orig vs in-place (R>2048 weight shapes) ===")
    print(f"{'R':>6s} {'C':>6s} {'orig':>8s} {'inpl':>8s} {'save%':>6s} ident")
    for C, R in ((2048, 8192), (4096, 4096), (4096, 8192), (8192, 8192),
                 (2048, 4096), (4096, 3072)):
        torch.manual_seed(9)
        A = torch.randn(4 * C, C)
        U = sol._upper_cholesky_inv(A.T @ A + torch.eye(C) * C)
        x = torch.randn(R, C) * 3
        u = torch.rand(R, C) + 0.5
        a = sol._gptq_quantize_values_torch(x, u, U)
        b = _gptq_torch_ip(x, u, U)
        ok = torch.equal(a, b)
        t0 = med(lambda: sol._gptq_quantize_values_torch(x, u, U))
        t1 = med(lambda: _gptq_torch_ip(x, u, U))
        print(f"{R:6d} {C:6d} {t0:8.3f} {t1:8.3f} {100*(t0-t1)/t0:5.1f}% {ok}")
        sys.stdout.flush()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("c2", "all"):
        e_c2()
    if which in ("ip", "all"):
        e_ip()
