"""E8: numpy round loop for tiny-T act refinement (dispatch-bound regime).
Same op sequence as the optimized torch round; fp32 numpy ops share the torch
buffers. argmin tie semantics: both torch CPU and numpy return the FIRST
minimal index -- verified empirically incl. tie storms (all-inf rounds).
"""
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
from exp2_refine import refine_act_opt, med  # noqa: E402

sol = harness.load_variant()
INF = float("inf")


def _rounds_np(M, v4, d, neg2d, d2col, gw, n_sweeps, rounds):
    Mn = M.numpy()
    v4n = v4.numpy()
    dn = d.numpy()
    nn2 = neg2d.numpy()
    d2c = d2col.numpy()
    gwn = gw.numpy() if gw.is_contiguous() else gw.contiguous().numpy()
    T = Mn.shape[0]
    g = np.empty_like(Mn)
    up = np.empty(Mn.shape, dtype=bool)
    keep = np.empty(Mn.shape, dtype=bool)
    bpos = v4n < 7.0
    bneg = v4n > -7.0
    ar = np.arange(T)
    one, mone = np.float32(1.0), np.float32(-1.0)
    for _ in range(n_sweeps):
        for _ in range(rounds):
            np.abs(Mn, out=g)
            np.multiply(g, nn2, out=g)
            np.add(g, d2c, out=g)
            np.less(Mn, 0.0, out=up)
            legal = np.where(up, bpos, bneg)
            np.less(g, 0.0, out=keep)
            np.logical_and(keep, legal, out=keep)
            np.logical_not(keep, out=keep)
            g[keep] = INF
            idx = g.argmin(axis=1)
            gg = g[ar, idx]
            fin = np.isfinite(gg)
            dr = np.where(up[ar, idx], one, mone)
            dr *= fin
            v4n[ar, idx] += dr
            nv = v4n[ar, idx]
            bpos[ar, idx] = nv < 7.0
            bneg[ar, idx] = nv > -7.0
            coef = dr * dn[ar, idx]
            gb = gwn[idx]
            gb *= coef[:, None]
            Mn += gb


def refine_act_np(x, values, unit, gw, gwf):
    v4 = torch.round(values / unit * 4.0)
    d = 0.25 * unit
    col2 = gw.diagonal()
    M = (v4 * d) @ gw - x @ gwf
    T = values.shape[0]
    n_sweeps = 12 if T <= 256 else 8 if T <= 512 else 5
    neg2d = -2.0 * d
    d2col = (d * d) * col2
    _rounds_np(M, v4, d, neg2d, d2col, gw, n_sweeps, sol.REFINE_ROUNDS)
    return v4 * d


def main():
    print("=== E8: tiny-T refinement torch(opt) vs numpy ===")
    print(f"{'T':>6s} {'C':>6s} {'orig':>8s} {'opt':>8s} {'np':>8s} ident(o-np)")
    ok_all = True
    for T, C in ((10, 2048), (10, 1024), (10, 4096), (16, 2048), (32, 2048),
                 (64, 2048), (128, 2048)):
        rng = torch.Generator().manual_seed(T * 1000 + C)
        x = torch.randn(T, C, generator=rng) * 3
        u = (torch.rand(T, C, generator=rng) + 0.5) * 0.1
        values = ((torch.randn(T, C, generator=rng) * 8).round() * 0.25) * u
        A = torch.randn(C, C, generator=rng)
        gw = A.T @ A + torch.eye(C) * C
        gwf = torch.randn(C, C, generator=rng).T @ A
        a = sol._refine_act_values(x, values, u, gw, gwf)
        b = refine_act_opt(x, values, u, gw, gwf)
        c = refine_act_np(x, values, u, gw, gwf)
        ok = torch.equal(a, b) and torch.equal(a, c)
        ok_all = ok_all and ok
        t0 = med(lambda: sol._refine_act_values(x, values, u, gw, gwf))
        t1 = med(lambda: refine_act_opt(x, values, u, gw, gwf))
        t2 = med(lambda: refine_act_np(x, values, u, gw, gwf))
        print(f"{T:6d} {C:6d} {t0:8.3f} {t1:8.3f} {t2:8.3f} {ok}")
        sys.stdout.flush()
    # tie-storm randomized unit
    rng = np.random.default_rng(99)
    fails = 0
    for tr in range(24):
        T = int(rng.integers(1, 17))
        C = 64 * int(rng.integers(1, 33))
        xr = torch.randn(T, C) * 3
        u = (torch.rand(T, C) + 0.5) * 0.1
        values = ((torch.randn(T, C) * 8).round() * 0.25) * u if tr % 2 else xr.clone()
        if tr % 4 == 3:
            values = values.round() * 0  # all-zero ties / no-flip rounds
        A = torch.randn(C, C)
        gw = A.T @ A + torch.eye(C) * C
        gwf = torch.randn(C, C).T @ A
        a = sol._refine_act_values(xr, values, u, gw, gwf)
        c = refine_act_np(xr, values, u, gw, gwf)
        if not torch.equal(a, c):
            fails += 1
            print(f"  tie-storm FAIL T={T} C={C}")
    print(f"[unit] np tiny-T: {24 - fails}/24 bit-identical (tie storms)")
    print("ALL OK" if ok_all and fails == 0 else "FAILURES PRESENT")


if __name__ == "__main__":
    main()
