"""E7: _gptq_quantize_values_np with preallocated out= buffers (no per-column
temporaries). Same op sequence/order as v25's numpy path -> bit-identical.
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


def _gptq_np_v2(x, unit, hinv):
    R, C = x.shape
    GB = sol.GPTQ_BLOCK
    W = x.clone()
    Q = torch.empty_like(W)
    unp = (unit if unit.is_contiguous() else unit.contiguous()).numpy()
    hnp = hinv.contiguous().numpy()
    npr_, npa_, npw_ = np.round, np.abs, np.where
    one, mone = np.float32(1.0), np.float32(-1.0)
    b = np.empty(R, dtype=np.float32)
    eb = np.empty(R, dtype=np.float32)
    tb = np.empty((R, GB - 1), dtype=np.float32)
    for i1 in range(0, C, GB):
        i2 = min(i1 + GB, C)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        w1, q1, e1 = W1.numpy(), Q1.numpy(), E1.numpy()
        Hi = hnp[i1:i2, i1:i2]
        u = unp[:, i1:i2]
        last = i2 - i1 - 1
        for i in range(i2 - i1):
            w = w1[:, i]
            ui = u[:, i]
            np.abs(w, out=b)
            np.divide(b, ui, out=b)
            np.multiply(b, 4.0, out=b)
            npr_(b, out=b)
            np.clip(b, 0.0, 7.0, out=b)
            np.multiply(b, np.float32(0.25), out=b)
            s = npw_(w >= 0, one, mone)
            np.multiply(s, b, out=b)      # b = s*m
            np.multiply(b, ui, out=b)     # b = (s*m)*ui = q
            q1[:, i] = b
            d = Hi[i, i]
            if d < 1e-30:
                d = np.float32(1e-30)
            np.subtract(w, b, out=eb)
            np.divide(eb, d, out=eb)
            e1[:, i] = eb
            if i < last:
                L = last - i
                np.multiply(eb[:, None], Hi[i, i + 1:], out=tb[:, :L])
                np.subtract(w1[:, i + 1:], tb[:, :L], out=w1[:, i + 1:])
        Q[:, i1:i2] = Q1
        if i2 < C:
            W[:, i2:] -= E1 @ hinv[i1:i2, i2:]
            W[:, i1:i2] = W1
    return Q


def main():
    print("=== E7: numpy GPTQ v25 vs out=-buffered ===")
    print(f"{'R':>6s} {'C':>6s} {'v25':>8s} {'v2':>8s} {'save%':>6s} ident")
    fails = 0
    torch.manual_seed(17)
    for C, R in ((2048, 1024), (2048, 512), (2048, 128), (2048, 10),
                 (4096, 1024), (8192, 512), (1024, 2048), (512, 256)):
        A = torch.randn(4 * C, C)
        U = sol._upper_cholesky_inv(A.T @ A + torch.eye(C) * C)
        x = torch.randn(R, C) * 3
        u = torch.rand(R, C) + 0.5
        a = sol._gptq_quantize_values_np(x, u, U)
        b = _gptq_np_v2(x, u, U)
        ok = torch.equal(a, b)
        fails += 0 if ok else 1
        t0 = med(lambda: sol._gptq_quantize_values_np(x, u, U))
        t1 = med(lambda: _gptq_np_v2(x, u, U))
        print(f"{R:6d} {C:6d} {t0:8.3f} {t1:8.3f} {100*(t0-t1)/t0:5.1f}% {ok}")
        sys.stdout.flush()
    # randomized stress incl. zeros/negatives
    rng = np.random.default_rng(31)
    for trial in range(20):
        R = int(rng.integers(1, 400))
        Cb = 128 * int(rng.integers(1, 6))
        x = torch.randn(R, Cb) * 3
        if trial % 3 == 1:
            x[0] = 0.0
        if trial % 3 == 2:
            x[:, 0] = -0.0
        A = torch.randn(4 * Cb, Cb)
        U = sol._upper_cholesky_inv(A.T @ A + torch.eye(Cb) * Cb)
        u = (torch.rand(R, Cb) + 0.5)
        if not torch.equal(sol._gptq_quantize_values_np(x, u, U), _gptq_np_v2(x, u, U)):
            fails += 1
            print(f"  stress FAIL trial={trial}")
    print(f"[unit] np_v2: {28 - fails}/28 bit-identical")


if __name__ == "__main__":
    main()
