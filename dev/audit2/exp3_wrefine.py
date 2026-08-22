"""E-C3: weight-refine round optimization + REFINE_W_CHUNK sweep.

_opt restructures the sweep/round/chunk nest to chunk/round order. PROOF of
identity: rows are independent (a flip in row r updates only v4[r], A[r];
Gxx is read-only), so each row's flip sequence is exactly rounds 1..20 in
order under either nesting. Per chunk we then hoist the loop-invariants
(-2*d, (d*d)*colE), cache the v4 bounds (scatter-maintained), use in-place
kernels and compute dirn at (chunk,1) -- same element-for-element mask as
_flip_sel (see exp2_refine.py).
"""
import os
import statistics
import sys
import time

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


def refine_w_core(w_final, q_used, weight_params, xh, Gxx, chunk, opt):
    """Both structures, parameterized. Returns (wn, accepted) pre-holdout."""
    N, C = q_used.shape
    unit_w = sol._params_unit_flat(weight_params)
    d = 0.25 * unit_w
    v4 = torch.round(q_used / unit_w * 4.0)
    colE = Gxx.diagonal()
    A = (q_used - w_final) @ Gxx
    INF = float("inf")
    ROUNDS = sol.REFINE_ROUNDS

    def round_opt(i1, i2, neg2d, d2col, bpos, bneg, g, up, legal, keep, gb):
        torch.abs(A[i1:i2], out=g)
        g.mul_(neg2d)
        g.add_(d2col)
        torch.lt(A[i1:i2], 0.0, out=up)
        torch.where(up, bpos, bneg, out=legal)
        torch.lt(g, 0.0, out=keep)
        keep &= legal
        g.masked_fill_(keep.logical_not_(), INF)
        idx = g.argmin(dim=1, keepdim=True)
        fin = torch.isfinite(g.gather(1, idx))
        dr = torch.where(up.gather(1, idx), 1.0, -1.0) * fin.float()
        v4[i1:i2].scatter_add_(1, idx, dr)
        nv = v4[i1:i2].gather(1, idx)
        bpos.scatter_(1, idx, nv < 7.0)
        bneg.scatter_(1, idx, nv > -7.0)
        coef = dr * d[i1:i2].gather(1, idx)
        torch.index_select(Gxx, 0, idx[:, 0], out=gb)
        gb.mul_(coef)
        A[i1:i2] += gb

    def round_orig(i1, i2):
        g, dirn = sol._flip_sel(d[i1:i2], A[i1:i2], colE, v4[i1:i2])
        idx = g.argmin(dim=1, keepdim=True)
        fin = torch.isfinite(g.gather(1, idx))
        dr = dirn.gather(1, idx) * fin.float()
        v4[i1:i2].scatter_add_(1, idx, dr)
        A[i1:i2] += (dr * d[i1:i2].gather(1, idx)) * Gxx[idx[:, 0]]

    if opt:
        for i1 in range(0, N, chunk):
            i2 = min(i1 + chunk, N)
            neg2d = -2.0 * d[i1:i2]
            d2col = (d[i1:i2] * d[i1:i2]) * colE
            bpos = v4[i1:i2] < 7.0
            bneg = v4[i1:i2] > -7.0
            g = torch.empty(i2 - i1, C)
            gb = torch.empty(i2 - i1, C)
            up = torch.empty(i2 - i1, C, dtype=torch.bool)
            legal = torch.empty(i2 - i1, C, dtype=torch.bool)
            keep = torch.empty(i2 - i1, C, dtype=torch.bool)
            for _ in range(sol.REFINE_W_SWEEPS):
                for _ in range(ROUNDS):
                    round_opt(i1, i2, neg2d, d2col, bpos, bneg, g, up, legal, keep, gb)
    else:
        for _ in range(sol.REFINE_W_SWEEPS):
            for _ in range(ROUNDS):
                for i1 in range(0, N, chunk):
                    round_orig(i1, min(i1 + chunk, N))
    return v4 * d


def setup(C, N, seed=3):
    torch.manual_seed(seed)
    w = torch.randn(N, C) * 0.05
    q = ((torch.randn(N, C) * 8).round() * 0.25) * (torch.rand(N, C) + 0.5) * 0.1
    X = torch.randn(650, C)
    Gxx = X.T @ X + torch.eye(C) * C
    xh = torch.randn(1024, C)
    # params-shaped dict for _params_unit_flat
    nb = C // 64
    sf = torch.rand(N, nb, 1, 1, 1) + 0.5
    lv2 = torch.ones(N, nb, 1, 1, 1)
    lv3 = torch.ones(N, nb, 1, 1, 1)
    p = {"scale_factor": sf, "scale_lv2": lv2, "scale_lv3": lv3}
    return w, q, p, Gxx, xh


def main():
    print("=== C3: weight refine orig(chunk sweep) vs opt(restructured) ===")
    print(f"{'N':>6s} {'C':>6s} {'chunk':>6s} {'orig s':>8s} {'opt s':>8s} ident")
    for C, N in ((2048, 8192), (4096, 4096), (1024, 4096)):
        w, q, p, Gxx, xh = setup(C, N)
        for chunk in (1024, 2048, 4096, 8192):
            a = refine_w_core(w, q, p, xh, Gxx, chunk, opt=False)
            b = refine_w_core(w, q, p, xh, Gxx, chunk, opt=True)
            # chunk-invariance check too (rows independent)
            a2 = refine_w_core(w, q, p, xh, Gxx, 8192 if chunk != 8192 else 1024,
                               opt=False)
            ok = torch.equal(a, b)
            okc = torch.equal(a, a2)
            t0 = med(lambda: refine_w_core(w, q, p, xh, Gxx, chunk, opt=False))
            t1 = med(lambda: refine_w_core(w, q, p, xh, Gxx, chunk, opt=True))
            print(f"{N:6d} {C:6d} {chunk:6d} {t0:8.3f} {t1:8.3f} "
                  f"ident={ok} chunkinv={okc}")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
