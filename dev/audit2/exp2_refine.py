"""E-C4: optimized _refine_act_values / weight-refine round loop.
Key changes (all bit-identical by construction):
  - hoist neg2d = -2.0*d and d2col = (d*d)*col2 out of the round loop (they
    are (T,C)-sized but loop-invariant; recomputed every round in v25)
  - cache the v4 legality bounds (bpos = v4<7, bneg = v4>-7) and maintain
    them by (T,1) scatter after each flip instead of two (T,C) comparisons
    per round (PROOF: bounds change only at the flipped column; scatter
    writes exactly the new comparison value at that column)
  - dirn computed at (T,1) after gather (where(up_g,1,-1) == where(up,1,-1).gather)
  - in-place/out= kernels: abs(M, out=g), masked_fill_, logical_not_, and_,
    index_select(out=), reused buffers -> fewer (T,C) allocations
Selection semantics: g = where(legal & (g<0), g, inf), argmin, isfinite --
element-for-element the same mask as v25's _flip_sel.
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


def refine_act_opt(x, values, unit, gw, gwf):
    v4 = torch.round(values / unit * 4.0)
    d = 0.25 * unit
    col2 = gw.diagonal()
    M = (v4 * d) @ gw - x @ gwf
    T = values.shape[0]
    n_sweeps = 12 if T <= 256 else 8 if T <= 512 else 5
    neg2d = -2.0 * d
    d2col = (d * d) * col2
    bpos = v4 < 7.0
    bneg = v4 > -7.0
    INF = float("inf")
    g = torch.empty_like(M)
    up = torch.empty(M.shape, dtype=torch.bool)
    legal = torch.empty(M.shape, dtype=torch.bool)
    keep = torch.empty(M.shape, dtype=torch.bool)
    gb = torch.empty_like(M)
    for _ in range(n_sweeps):
        for _ in range(sol.REFINE_ROUNDS):
            torch.abs(M, out=g)
            g.mul_(neg2d)
            g.add_(d2col)
            torch.lt(M, 0.0, out=up)
            torch.where(up, bpos, bneg, out=legal)
            torch.lt(g, 0.0, out=keep)
            keep &= legal
            g.masked_fill_(keep.logical_not_(), INF)
            idx = g.argmin(dim=1, keepdim=True)
            fin = torch.isfinite(g.gather(1, idx))
            dr = torch.where(up.gather(1, idx), 1.0, -1.0) * fin.float()
            v4.scatter_add_(1, idx, dr)
            nv = v4.gather(1, idx)
            bpos.scatter_(1, idx, nv < 7.0)
            bneg.scatter_(1, idx, nv > -7.0)
            coef = dr * d.gather(1, idx)
            torch.index_select(gw, 0, idx[:, 0], out=gb)
            gb.mul_(coef)
            M.add_(gb)
    return v4 * d


def unit_test_act(nt=30):
    rng = torch.Generator().manual_seed(123)
    fails = 0
    for tr in range(nt):
        T = int(torch.randint(1, 300, (1,), generator=rng))
        C = 64 * int(torch.randint(1, 33, (1,), generator=rng))
        x = torch.randn(T, C, generator=rng) * 3
        u = (torch.rand(T, C, generator=rng) + 0.5) * 0.1
        # grid-snapped values (tie/bound storms) half the time
        values = ((torch.randn(T, C, generator=rng) * 8).round() * 0.25) * u \
            if tr % 2 else (x.clone())
        A = torch.randn(C, C, generator=rng)
        gw = A.T @ A + torch.eye(C) * C
        B = torch.randn(C, C, generator=rng)
        gwf = B.T @ A
        a = sol._refine_act_values(x, values, u, gw, gwf)
        b = refine_act_opt(x, values, u, gw, gwf)
        if not torch.equal(a, b):
            fails += 1
            nd = (a != b).sum().item()
            print(f"  FAIL T={T} C={C} ndiff={nd}")
    print(f"[unit] refine_act_opt: {nt - fails}/{nt} bit-identical")
    return fails == 0


def bench_act(shapes):
    print("=== C4: _refine_act_values orig vs opt ===")
    print(f"{'T':>6s} {'C':>6s} {'orig s':>8s} {'opt s':>8s} {'save':>7s} {'%':>6s}")
    for T, C in shapes:
        rng = torch.Generator().manual_seed(T * 1000 + C)
        x = torch.randn(T, C, generator=rng) * 3
        u = (torch.rand(T, C, generator=rng) + 0.5) * 0.1
        values = ((torch.randn(T, C, generator=rng) * 8).round() * 0.25) * u
        A = torch.randn(C, C, generator=rng)
        gw = A.T @ A + torch.eye(C) * C
        gwf = torch.randn(C, C, generator=rng).T @ A
        a = sol._refine_act_values(x, values, u, gw, gwf)
        b = refine_act_opt(x, values, u, gw, gwf)
        ok = torch.equal(a, b)
        t0 = med(lambda: sol._refine_act_values(x, values, u, gw, gwf))
        t1 = med(lambda: refine_act_opt(x, values, u, gw, gwf))
        print(f"{T:6d} {C:6d} {t0:8.3f} {t1:8.3f} {t0-t1:7.3f} "
              f"{100*(t0-t1)/t0:5.1f}% ident={ok}")
        sys.stdout.flush()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("unit", "all"):
        unit_test_act()
    if which in ("bench", "all"):
        bench_act([(10, 2048), (128, 2048), (256, 2048), (512, 2048),
                   (1024, 2048), (512, 1024), (1024, 4096), (512, 4096),
                   (1024, 1024)])
