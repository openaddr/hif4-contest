"""E1: numpy-vs-torch GPTQ dispatch boundary, R in (2048, 4096], C grid.
E6: numpy twin of _gptq_quantize_batched (attn q/k/v path).
Interleaved reps, medians. Bit-identity via torch.equal on every rep.
"""
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

sol = harness.load_variant()


def med(fn, reps=3):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def make_hinv(C, sol):
    torch.manual_seed(11)
    A = torch.randn(4 * C, C)
    H = A.T @ A + torch.eye(C) * C
    return sol._upper_cholesky_inv(H)


def e1():
    print("=== E1: _gptq_quantize_values torch vs numpy (block 128) ===")
    print(f"{'R':>6s} {'C':>6s} {'torch s':>9s} {'numpy s':>9s} {'winner':>8s} ident")
    GB = sol.GPTQ_BLOCK
    for C in (2048, 4096, 8192):
        U = make_hinv(C, sol)
        for R in (2048, 2560, 3072, 4096, 8192):
            torch.manual_seed(5)
            x = torch.randn(R, C) * 3
            u = torch.rand(R, C) + 0.5
            a = sol._gptq_quantize_values_torch(x, u, U)
            ok = torch.equal(sol._gptq_quantize_values_np(x, u, U), a)
            tt = med(lambda: sol._gptq_quantize_values_torch(x, u, U))
            tn = med(lambda: sol._gptq_quantize_values_np(x, u, U))
            w = "numpy" if tn < tt else "torch"
            print(f"{R:6d} {C:6d} {tt:9.3f} {tn:9.3f} {w:>8s} {ok}")
            sys.stdout.flush()


# --- E6: numpy twin of _gptq_quantize_batched -------------------------------

def _gptq_batched_np(x, unit, hinv):
    """numpy twin of _gptq_quantize_batched: per-column elementwise in numpy,
    cross-block matmul in torch (identical accumulation)."""
    B, R, n = x.shape
    per_batch = hinv.dim() == 3
    W = x.clone()
    Q = torch.empty_like(W)
    unp = (unit if unit.is_contiguous() else unit.contiguous()).numpy()
    hnp = hinv.contiguous().numpy()
    npr_, npa_, npw_, npc_ = np.round, np.abs, np.where, np.clip
    one, mone = np.float32(1.0), np.float32(-1.0)
    for i1 in range(0, n, sol.GPTQ_BLOCK):
        i2 = min(i1 + sol.GPTQ_BLOCK, n)
        W1 = W[..., i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        w1, q1, e1 = W1.numpy(), Q1.numpy(), E1.numpy()
        Hi = hnp[..., i1:i2, i1:i2]
        u = unp[..., i1:i2]
        last = i2 - i1 - 1
        for i in range(i2 - i1):
            w = w1[..., i]
            ui = u[..., i]
            m = npr_(npa_(w) / ui * 4.0)
            npc_(m, 0.0, 7.0, out=m)
            m *= 0.25
            s = npw_(w >= 0, one, mone)
            q = s * m * ui
            q1[..., i] = q
            d = Hi[..., i, i]
            if not per_batch:
                if d < 1e-30:
                    d = np.float32(1e-30)
                e1[..., i] = (w - q) / d
            else:
                d = np.maximum(d, np.float32(1e-30))
                e1[..., i] = (w - q) / d[..., None]
            if i < last:
                seg = Hi[..., i, i + 1:]
                if per_batch:
                    seg = seg[..., None, :]
                w1[..., i + 1:] -= e1[..., i][..., None] * seg
        Q[..., i1:i2] = Q1
        if i2 < n:
            W[..., i2:] -= torch.matmul(E1, hinv[..., i1:i2, i2:])
            W[..., i1:i2] = W1
    return Q


def e6():
    print("\n=== E6: _gptq_quantize_batched torch vs numpy (attn shapes) ===")
    print(f"{'B':>4s} {'R':>6s} {'n':>5s} {'hinv':>6s} {'torch s':>9s} {'numpy s':>9s} ident")
    for B, R, n, pb in ((16, 512, 256, 1), (16, 1024, 256, 1), (16, 1024, 256, 0),
                        (2, 1024, 256, 1), (2, 512, 256, 1), (16, 128, 256, 1),
                        (2, 2048, 256, 1), (32, 1024, 128, 1)):
        torch.manual_seed(7)
        x = torch.randn(B, R, n) * 2
        u = torch.rand(B, R, n) + 0.5
        if pb:
            A = torch.randn(4 * n, n)
            H = torch.stack([A.T @ A + torch.eye(n) * n] * B)
        else:
            A = torch.randn(4 * n, n)
            H = A.T @ A + torch.eye(n) * n
        a = sol._gptq_quantize_batched(x, u, H)
        b = _gptq_batched_np(x, u, H)
        ok = torch.equal(a, b)
        tt = med(lambda: sol._gptq_quantize_batched(x, u, H))
        tn = med(lambda: _gptq_batched_np(x, u, H))
        print(f"{B:4d} {R:6d} {n:5d} {'B,n,n' if pb else 'n,n':>6s} "
              f"{tt:9.3f} {tn:9.3f} {ok}")
        sys.stdout.flush()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("e1", "all"):
        e1()
    if which in ("e6", "all"):
        e6()
