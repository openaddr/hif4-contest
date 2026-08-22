"""roundopt/make_patch: produce the patched solution copy (solution.py is
NEVER modified).

Ships three exact changes into dev/roundopt/patched/solution.py:
  A. _rounds_np: single flattened loop + EARLY EXIT when a round produces
     no flip (rows freeze monotonically; a no-flip round leaves every row
     value-unchanged, so later rounds are no-ops).
  B. _refine_act_values T>32 loop -> _rounds_active (active-set compaction
     + pass-reduced round + numpy tail at A<=32).
  C. _refine_weight_values chunk loop -> _rounds_active on each chunk.

Every replacement asserts exactly one match in the current source.
Usage: python dev/roundopt/make_patch.py   (then self_check on patched/)
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SOL = os.path.join(ROOT, "example", "solution", "solution.py")
OUT_DIR = os.path.join(HERE, "patched")
OUT = os.path.join(OUT_DIR, "solution.py")

ROUNDS_NP_OLD = '''def _rounds_np(M, v4, d, neg2d, d2col, gw, n_sweeps, rounds):
    """numpy twin of the optimized round loop for tiny T (<= 32), where the
    torch version is dispatch-bound (~14 kernels/round on (T,C)). Same op
    sequence in fp32 sharing the torch buffers; argmin returns the first
    minimal index in both torch CPU and numpy. Verified torch.equal incl.
    tie-storm inputs."""
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
'''

ROUNDS_NP_NEW = '''def _rounds_np(M, v4, d, neg2d, d2col, gw, n_sweeps, rounds):
    """numpy twin of the optimized round loop for tiny T (<= 32), where the
    torch version is dispatch-bound (~14 kernels/round on (T,C)). Same op
    sequence in fp32 sharing the torch buffers; argmin returns the first
    minimal index in both torch CPU and numpy. Verified torch.equal incl.
    tie-storm inputs.  roundopt: flattened to one loop with an EARLY EXIT
    when a round flips nothing -- rows freeze monotonically (each row's
    state changes only via its own flips), so every later round is a no-op
    and skipping them is value-identical."""
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
    for _ in range(n_sweeps * rounds):
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
        if not fin.any():
            break
'''

ROUNDS_ACTIVE_SRC = '''

_ROUND_NP_THRESH = 32       # active rows at/below: numpy tail (dispatch)


def _rounds_active(M, v4, d, neg2d, d2col, gw, total_rounds):
    """Active-set round loop; SEQUENCE-identical to the hoisted loop.

    Facts (roundopt, verified on 160 realistic refine calls):
      * rows are independent: the rank-1 update M += coef[:,None]*gw[idx]
        touches row r with row r's own (idx[r], coef[r]) only, and v4 /
        bounds likewise; a row without a legal improving flip (dr=0) is
        value-unchanged, hence can never flip later -- rows freeze
        monotonically and the flipping set shrinks each round.
      * per active row the round ops are the same values in the same order,
        so selections and flips are bit-identical; frozen rows are dropped
        (their remaining += 0.0 updates only affect zero signs).

    Pass-reduced round (bit/sequence identical, fewer (A,C) passes):
      - g = d2col + |M|*(-2d) in ONE addcmul (no-FMA on CPU: bit-equal to
        mul_+add_, torch.equal verified across the suite);
      - legality as ILLEGAL masks (v4>=7 / v4<=-7): g filled INF only where
        ILLEGAL; fin = (g_sel < 0).  When a flip exists the argmin sees the
        same negative-legal values (negatives < positives) -> same index;
        otherwise dr=0 in both variants and the round is a value no-op.
      - M += coef*gw[idx] via addcmul (one pass less).
    v4 is updated in place for every row; M rows of still-active rows are
    written back (frozen rows keep their last in-place values, which are
    value-final since later updates add +-0.0 only).
    """
    T, C = M.shape
    if T == 0 or total_rounds <= 0:
        return
    Ma = M
    da, neg2da, d2ca = d, neg2d, d2col
    rows = torch.arange(T, dtype=torch.long)
    v4a = v4
    local = False                    # buffers still alias M/v4
    ipos = v4 >= 7.0                 # illegal-step masks (cannot move)
    ineg = v4 <= -7.0
    g = torch.empty(T, C, dtype=torch.float32)
    gb = torch.empty(T, C, dtype=torch.float32)
    up = torch.empty(T, C, dtype=torch.bool)
    ill = torch.empty(T, C, dtype=torch.bool)
    A = T
    rnd = 0

    def _writeback():
        if local:
            v4[rows] = v4a
            M[rows] = Ma

    while rnd < total_rounds and A > 0:
        if A <= _ROUND_NP_THRESH:
            # tiny active block: numpy twin (torch is dispatch-bound here),
            # in place on the compact buffers, with the same early exit
            _rounds_np(Ma, v4a, da, neg2da, d2ca, gw, 1, total_rounds - rnd)
            _writeback()
            return
        torch.abs(Ma, out=g)
        torch.addcmul(d2ca, g, neg2da, out=g)
        torch.lt(Ma, 0.0, out=up)
        torch.where(up, ipos, ineg, out=ill)
        g.masked_fill_(ill, _ROUND_INF)
        idx = g.argmin(dim=1, keepdim=True)
        fin = g.gather(1, idx) < 0.0
        dr = torch.where(Ma.gather(1, idx) < 0.0, 1.0, -1.0) * fin.float()
        v4a.scatter_add_(1, idx, dr)
        nv = v4a.gather(1, idx)
        ipos.scatter_(1, idx, nv >= 7.0)
        ineg.scatter_(1, idx, nv <= -7.0)
        coef = dr * da.gather(1, idx)
        torch.index_select(gw, 0, idx[:, 0], out=gb)
        torch.addcmul(Ma, gb, coef, out=Ma)
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
        ipos = ipos[m]
        ineg = ineg[m]
        local = True
        A = rows.shape[0]
        g = torch.empty(A, C, dtype=torch.float32)
        gb = torch.empty(A, C, dtype=torch.float32)
        up = torch.empty(A, C, dtype=torch.bool)
        ill = torch.empty(A, C, dtype=torch.bool)
        rnd += 1
    _writeback()
'''

ACT_LOOP_OLD = '''    bpos = v4 < 7.0           # legality bounds; change only at flipped cols
    bneg = v4 > -7.0
    g = torch.empty_like(M)
    up = torch.empty(M.shape, dtype=torch.bool)
    legal = torch.empty(M.shape, dtype=torch.bool)
    keep = torch.empty(M.shape, dtype=torch.bool)
    gb = torch.empty_like(M)
    for _ in range(n_sweeps):
        for _ in range(REFINE_ROUNDS):
            idx, dr = _round_hoisted(M, neg2d, d2col, bpos, bneg,
                                     g, up, legal, keep)
            v4.scatter_add_(1, idx, dr)
            _bound_update(v4, bpos, bneg, idx)
            coef = dr * d.gather(1, idx)
            torch.index_select(gw, 0, idx[:, 0], out=gb)
            gb.mul_(coef)
            M += gb
    return v4 * d
'''

ACT_LOOP_NEW = '''    _rounds_active(M, v4, d, neg2d, d2col, gw, n_sweeps * REFINE_ROUNDS)
    return v4 * d
'''

W_LOOP_OLD = '''        neg2d = -2.0 * d[i1:i2]
        d2col = (d[i1:i2] * d[i1:i2]) * colE
        bpos = v4[i1:i2] < 7.0
        bneg = v4[i1:i2] > -7.0
        rc = i2 - i1
        g = torch.empty(rc, C, dtype=torch.float32)
        gb = torch.empty(rc, C, dtype=torch.float32)
        up = torch.empty(rc, C, dtype=torch.bool)
        legal = torch.empty(rc, C, dtype=torch.bool)
        keep = torch.empty(rc, C, dtype=torch.bool)
        Ac = A[i1:i2]
        v4c = v4[i1:i2]
        for _ in range(REFINE_W_SWEEPS):
            for _ in range(REFINE_ROUNDS):
                idx, dr = _round_hoisted(Ac, neg2d, d2col, bpos, bneg,
                                         g, up, legal, keep)
                v4c.scatter_add_(1, idx, dr)
                _bound_update(v4c, bpos, bneg, idx)
                coef = dr * d[i1:i2].gather(1, idx)
                torch.index_select(Gxx, 0, idx[:, 0], out=gb)
                gb.mul_(coef)
                Ac += gb
'''

W_LOOP_NEW = '''        neg2d = -2.0 * d[i1:i2]
        d2col = (d[i1:i2] * d[i1:i2]) * colE
        Ac = A[i1:i2]
        v4c = v4[i1:i2]
        _rounds_active(Ac, v4c, d[i1:i2], neg2d, d2col, Gxx,
                       REFINE_W_SWEEPS * REFINE_ROUNDS)
'''


def sub(src, old, new, tag):
    n = src.count(old)
    if n != 1:
        raise RuntimeError(f"patch target {tag!r}: {n} matches (want 1)")
    return src.replace(old, new)


def main():
    with open(SOL, encoding="utf-8") as f:
        src = f.read()
    src = sub(src, ROUNDS_NP_OLD, ROUNDS_NP_NEW, "_rounds_np")
    src = sub(src, ACT_LOOP_OLD, ACT_LOOP_NEW, "act loop")
    src = sub(src, W_LOOP_OLD, W_LOOP_NEW, "weight loop")
    anchor = "\n\ndef _refine_act_values("
    src = sub(src, anchor, ROUNDS_ACTIVE_SRC + anchor, "_rounds_active insert")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        f.write(src)
    print(f"wrote {OUT} ({len(src)} bytes)")


if __name__ == "__main__":
    main()
