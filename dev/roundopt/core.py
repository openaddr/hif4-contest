"""roundopt/core: bit-identical active-set round loop + tracing replica.

Central fact (verified empirically in analyze.py): the ship round loop is
per-row independent.  The rank-1 update M += coef[:,None] * gw[idx] applies
row r's own (idx[r], coef[r]) only; a row whose best gain is INF (no legal
improving flip) gets coef 0 and its M row / v4 row / bounds are unchanged,
so it can never acquire a flip later: rows freeze monotonically.  The ship
loop still recomputes the full (T,C) gain matrix + argmin every round for
all n_sweeps*REFINE_ROUNDS rounds.

`rounds_active` compacts to the flipping (active) set each round and stops
when it empties, using the IDENTICAL op sequence per active row, hence a
bit-identical flip sequence and final v4 (torch.equal).

Value-identity notes (why skipping no-op updates is safe):
  - the ship loop adds dr=0 rows to v4[idx] (+-0.0): -0.0 + 0.0 -> +0.0
    changes only the zero SIGN; v4 feeds only <7/>-7 tests and v4*d (and
    torch.equal compares values, -0.0 == +0.0).  Same for M += 0*gw[idx].
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SOL_PATH = os.path.join(ROOT, "example", "solution", "solution.py")
_ROUND_INF = float("inf")


def load_sol():
    spec = importlib.util.spec_from_file_location("_ro_sol", SOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sweeps_for(T, S):
    return 32 if T <= 256 else 14 if T <= 512 else 6


# ---------------------------------------------------------------------------
# 1. replica of the ship round loop, with optional flip tracing
# ---------------------------------------------------------------------------

def rounds_ship_torch(M, v4, d, neg2d, d2col, gw, total_rounds, trace=None,
                      curve=None):
    """Verbatim replica of the solution T>32 loop (op order from
    _round_hoisted + apply block).  Optional trace: (rnd,row,col,dr) for
    dr != 0; curve: |active| (rows that flipped) per round."""
    bpos = v4 < 7.0
    bneg = v4 > -7.0
    g = torch.empty_like(M)
    up = torch.empty(M.shape, dtype=torch.bool)
    legal = torch.empty(M.shape, dtype=torch.bool)
    keep = torch.empty(M.shape, dtype=torch.bool)
    gb = torch.empty_like(M)
    for rnd in range(total_rounds):
        torch.abs(M, out=g)
        g.mul_(neg2d)
        g.add_(d2col)
        torch.lt(M, 0.0, out=up)
        torch.where(up, bpos, bneg, out=legal)
        torch.lt(g, 0.0, out=keep)
        keep &= legal
        g.masked_fill_(keep.logical_not_(), _ROUND_INF)
        idx = g.argmin(dim=1, keepdim=True)
        fin = torch.isfinite(g.gather(1, idx))
        dr = torch.where(up.gather(1, idx), 1.0, -1.0) * fin.float()
        idx_l = idx[:, 0].tolist()
        dr_l = dr[:, 0].tolist()
        v4.scatter_add_(1, idx, dr)
        nv = v4.gather(1, idx)
        bpos.scatter_(1, idx, nv < 7.0)
        bneg.scatter_(1, idx, nv > -7.0)
        coef = dr * d.gather(1, idx)
        torch.index_select(gw, 0, idx[:, 0], out=gb)
        gb.mul_(coef)
        M += gb
        if trace is not None or curve is not None:
            for r, (c, dv) in enumerate(zip(idx_l, dr_l)):
                if dv != 0.0 and trace is not None:
                    trace.append((rnd, r, c, dv))
            if curve is not None:
                curve.append(sum(1 for dv in dr_l if dv != 0.0))


def rounds_ship_np(M, v4, d, neg2d, d2col, gw, n_sweeps, rounds, trace=None,
                   curve=None):
    """Verbatim replica of solution _rounds_np with optional tracing."""
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
    rnd = 0
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
            g[keep] = _ROUND_INF
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
            if trace is not None or curve is not None:
                nfl = 0
                for r in range(T):
                    if dr[r] != 0.0:
                        nfl += 1
                        if trace is not None:
                            trace.append((rnd, r, int(idx[r]), float(dr[r])))
                if curve is not None:
                    curve.append(nfl)
            rnd += 1


def refine_ship(x, values, unit, gw, gwf, S=None, trace=None, curve=None):
    """Replica of _refine_act_values (both T paths) with optional tracing."""
    if S is None:
        S = load_sol()
    v4 = torch.round(values / unit * 4.0)
    d = 0.25 * unit
    col2 = gw.diagonal()
    M = (v4 * d) @ gw - x @ gwf
    T = values.shape[0]
    n_sweeps = sweeps_for(T, S)
    neg2d = -2.0 * d
    d2col = (d * d) * col2
    if T <= 32:
        rounds_ship_np(M, v4, d, neg2d, d2col, gw, n_sweeps, S.REFINE_ROUNDS,
                       trace=trace, curve=curve)
        d2 = None
        for _ in range(8):
            gchk, _ = S._flip_sel(d, M, col2, v4)
            if not bool((gchk < 0).any()):
                break
            rounds_ship_np(M, v4, d, neg2d, d2col, gw, 1, S.REFINE_ROUNDS,
                           trace=trace)
        return v4 * d
    rounds_ship_torch(M, v4, d, neg2d, d2col, gw,
                      n_sweeps * S.REFINE_ROUNDS, trace=trace, curve=curve)
    return v4 * d


# ---------------------------------------------------------------------------
# 2. the active-set optimized round loop
# ---------------------------------------------------------------------------

def _round_one(Ma, neg2da, d2ca, bpos, bneg, g, up, legal, keep):
    """One round on the active block; op order identical to _round_hoisted."""
    torch.abs(Ma, out=g)
    g.mul_(neg2da)
    g.add_(d2ca)
    torch.lt(Ma, 0.0, out=up)
    torch.where(up, bpos, bneg, out=legal)
    torch.lt(g, 0.0, out=keep)
    keep &= legal
    g.masked_fill_(keep.logical_not_(), _ROUND_INF)
    idx = g.argmin(dim=1, keepdim=True)
    fin = torch.isfinite(g.gather(1, idx))
    dr = torch.where(up.gather(1, idx), 1.0, -1.0) * fin.float()
    return idx, dr, fin


def rounds_np_active(M, v4, d, neg2d, d2col, gw, total_rounds, trace=None,
                     curve=None, rows_map=None, rnd0=0):
    """numpy twin of _rounds_np with ONE change: a round in which NO row
    flips ends the loop.  Freeze monotonicity (per-row state changes only
    via the row's own flips; a non-flipping round leaves every row bit-
    unchanged up to zero signs) makes the early exit exact.  Full storage
    is kept (no compaction) so M/v4 stay live for callers that read them
    afterwards (tiny-T deepening check).  rows_map maps the row axis to
    global ids for tracing."""
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
    rows = ar if rows_map is None else rows_map.numpy()
    one, mone = np.float32(1.0), np.float32(-1.0)
    for _r in range(total_rounds):
        rnd = _r + rnd0
        np.abs(Mn, out=g)
        np.multiply(g, nn2, out=g)
        np.add(g, d2c, out=g)
        np.less(Mn, 0.0, out=up)
        legal = np.where(up, bpos, bneg)
        np.less(g, 0.0, out=keep)
        np.logical_and(keep, legal, out=keep)
        np.logical_not(keep, out=keep)
        g[keep] = _ROUND_INF
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
        if trace is not None or curve is not None:
            for r in range(T):
                if dr[r] != 0.0:
                    if trace is not None:
                        trace.append((rnd, int(rows[r]), int(idx[r]),
                                      float(dr[r])))
            if curve is not None:
                curve.append(int((dr != 0.0).sum()))
        if not fin.any():
            break


def _round_one_v2(Ma, neg2da, d2ca, ipos, ineg, g, up, ill):
    """Pass-reduced round; SEQUENCE-identical to _round_hoisted.

    Differences and why the flip sequence is unchanged:
      - g = d2col + |M|*(-2d) via one addcmul (bit-identical to mul_+add_;
        verified no-FMA on this CPU and torch.equal across the suite).
      - legality kept as ILLEGAL masks ipos=v4>=7 (cannot step up) /
        ineg=v4<=-7; ill = where(M<0, ipos, ineg); g masked to INF only
        where ILLEGAL (ship also fills non-improving legal entries).  If a
        row has any legal negative gain, the argmin is over the same
        negative-legal values in both versions -> identical index (negatives
        sort below positives).  Otherwise dr=0 in both (ship fin=isfinite
        on all-INF argmin; v2 fin=g_sel<0) and the round is a value no-op
        for the row (+-0.0 adds only).  Direction from M.gather < 0 (same
        predicate as up.gather).
      - M += coef*gw[idx] via addcmul (bit-identical, one pass less).
    """
    torch.abs(Ma, out=g)
    torch.addcmul(d2ca, g, neg2da, out=g)
    torch.lt(Ma, 0.0, out=up)
    torch.where(up, ipos, ineg, out=ill)
    g.masked_fill_(ill, _ROUND_INF)
    idx = g.argmin(dim=1, keepdim=True)
    gsel = g.gather(1, idx)
    fin = gsel < 0.0
    dr = torch.where(Ma.gather(1, idx) < 0.0, 1.0, -1.0) * fin.float()
    return idx, dr, fin


def rounds_active(M, v4, d, neg2d, d2col, gw, total_rounds, trace=None,
                  curve=None, np_thresh=0, v2=False):
    """Active-set twin of the ship T>32 round loop.

    Row r's state changes only via its own flips (rank-1 update is per-row;
    coef=0 rows are unchanged), so the flipping set shrinks monotonically:
    each round runs the IDENTICAL op sequence on the active rows only, and
    the loop ends when the active set empties.  v4 is updated in place with
    the final values of every row; M is consumed (frozen rows are stale).
    """
    T, C = M.shape
    if T == 0 or total_rounds <= 0:
        return
    Ma = M
    da, neg2da, d2ca = d, neg2d, d2col
    rows = torch.arange(T, dtype=torch.long)
    v4a = v4
    local = False                    # buffers still alias M/v4
    if v2:
        ipos = v4 >= 7.0             # cannot step up / down (illegal masks)
        ineg = v4 <= -7.0
    else:
        bpos = v4 < 7.0
        bneg = v4 > -7.0
    g = torch.empty(T, C, dtype=torch.float32)
    gb = torch.empty(T, C, dtype=torch.float32)
    up = torch.empty(T, C, dtype=torch.bool)
    if v2:
        ill = torch.empty(T, C, dtype=torch.bool)
    else:
        legal = torch.empty(T, C, dtype=torch.bool)
        keep = torch.empty(T, C, dtype=torch.bool)
    A = T
    rnd = 0

    def writeback():
        if local:
            v4[rows] = v4a
            M[rows] = Ma

    while rnd < total_rounds and A > 0:
        if np_thresh and A <= np_thresh:
            # tiny active block: numpy twin (dispatch-bound in torch),
            # operating on the compact buffers in place
            rounds_np_active(Ma, v4a, da, neg2da, d2ca, gw,
                             total_rounds - rnd, trace=trace,
                             rows_map=rows, rnd0=rnd)
            writeback()
            return
        if v2:
            idx, dr, fin = _round_one_v2(Ma, neg2da, d2ca, ipos, ineg,
                                         g, up, ill)
        else:
            idx, dr, fin = _round_one(Ma, neg2da, d2ca, bpos, bneg,
                                      g, up, legal, keep)
        v4a.scatter_add_(1, idx, dr)
        nv = v4a.gather(1, idx)
        if v2:
            ipos.scatter_(1, idx, nv >= 7.0)
            ineg.scatter_(1, idx, nv <= -7.0)
        else:
            bpos.scatter_(1, idx, nv < 7.0)
            bneg.scatter_(1, idx, nv > -7.0)
        coef = dr * da.gather(1, idx)
        torch.index_select(gw, 0, idx[:, 0], out=gb)
        if v2:
            torch.addcmul(Ma, gb, coef, out=Ma)
        else:
            gb.mul_(coef)
            Ma += gb
        if trace is not None:
            idx_l = idx[:, 0].tolist()
            dr_l = dr[:, 0].tolist()
            rows_l = rows.tolist()
            for r in range(A):
                if dr_l[r] != 0.0:
                    trace.append((rnd, rows_l[r], idx_l[r], dr_l[r]))
        if curve is not None:
            curve.append(int(fin.sum()))
        if bool(fin.all()):
            rnd += 1
            continue
        m = fin[:, 0]
        if local:
            v4[rows[~m]] = v4a[~m]
            M[rows[~m]] = Ma[~m]
        rows = rows[m]
        Ma = Ma[m]
        v4a = v4a[m]
        da = da[m]
        neg2da = neg2da[m]
        d2ca = d2ca[m]
        if v2:
            ipos = ipos[m]
            ineg = ineg[m]
        else:
            bpos = bpos[m]
            bneg = bneg[m]
        local = True
        A = rows.shape[0]
        g = torch.empty(A, C, dtype=torch.float32)
        gb = torch.empty(A, C, dtype=torch.float32)
        up = torch.empty(A, C, dtype=torch.bool)
        if v2:
            ill = torch.empty(A, C, dtype=torch.bool)
        else:
            legal = torch.empty(A, C, dtype=torch.bool)
            keep = torch.empty(A, C, dtype=torch.bool)
        rnd += 1
    writeback()


def refine_active(x, values, unit, gw, gwf, S=None, trace=None, curve=None,
                  np_thresh=0, np_tiny=True, v2=False):
    """Drop-in twin of _refine_act_values using the active-set loops."""
    if S is None:
        S = load_sol()
    v4 = torch.round(values / unit * 4.0)
    d = 0.25 * unit
    col2 = gw.diagonal()
    M = (v4 * d) @ gw - x @ gwf
    T = values.shape[0]
    n_sweeps = sweeps_for(T, S)
    neg2d = -2.0 * d
    d2col = (d * d) * col2
    if T <= 32:
        if np_tiny:
            rounds_np_active(M, v4, d, neg2d, d2col, gw,
                             n_sweeps * S.REFINE_ROUNDS, trace=trace)
            for _ in range(8):
                gchk, _ = S._flip_sel(d, M, col2, v4)
                if not bool((gchk < 0).any()):
                    break
                rounds_np_active(M, v4, d, neg2d, d2col, gw,
                                 S.REFINE_ROUNDS, trace=trace)
        else:
            S._rounds_np(M, v4, d, neg2d, d2col, gw, n_sweeps,
                         S.REFINE_ROUNDS)
            for _ in range(8):
                gchk, _ = S._flip_sel(d, M, col2, v4)
                if not bool((gchk < 0).any()):
                    break
                S._rounds_np(M, v4, d, neg2d, d2col, gw, 1, S.REFINE_ROUNDS)
        return v4 * d
    rounds_active(M, v4, d, neg2d, d2col, gw,
                  n_sweeps * S.REFINE_ROUNDS, trace=trace, curve=curve,
                  np_thresh=np_thresh, v2=v2)
    return v4 * d
