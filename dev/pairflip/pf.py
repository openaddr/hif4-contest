"""pairflip shared harness: cooperative pair-flip (2-opt) probe machinery.

Objective algebra (transformed space, per row r; rows are independent):
    J(v) = || v @ q_used^T - x @ w_final^T ||^2
         = <v, v@gw> - 2<v, x@gwf> + const,   M = v@gw - x@gwf
A single flip of (r, c) by s in {-1,+1} grid steps changes J by
    dJ(s, c) = 2*s*d_rc*M_rc + d_rc^2 * gw_cc          (greedy top-1 eats this)
A PAIR of flips (r,c1,s1),(r,c2,s2) additionally gets the interaction term
    2*s1*s2*d_rc1*d_rc2*gw[c1,c2]
which no per-element greedy sees.  Two individually non-improving flips
(A(c) >= 0 for all c at greedy convergence) can therefore be jointly
improving: a "cooperative pair".  Multi-step single moves are dominated at
1-step convergence (dJ(2) < 0 requires |M| > d*gw_cc while dJ(1) >= 0
requires |M| <= d*gw_cc/2), so pairs are genuinely the next search frontier.

Scoring = dev/smooth/mini_holdout.py convention:
    pp = (mse_std - mse_play) / mse_std * 100
and since the smoothing/rotation invariants make the transformed-space
residual EXACTLY the output residual,  dMSE = dJ_sum / (T * N).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "dev"))
import hif4          # noqa: E402
import variants as V  # noqa: E402

SOL_PATH = os.path.join(HERE, "solution.py")
_INF = float("inf")


def load_sol(name="_pairflip_sol"):
    spec = importlib.util.spec_from_file_location(name, SOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# context extraction (ship output point)
# ---------------------------------------------------------------------------

def make_ctx(SOL, pair, st, check=True):
    """Replicate the ship dynamic call and recover the refinement context.

    Returns dict with x (transformed), v4 (ship grid values), d, unit,
    p_base (scales), gw/gwf (fp32).  Verifies the recovered grid repacks
    bit-identically to the ship params (same mant/sign)."""
    C = pair[0].shape[1]
    s = st.get("s")
    s = s.float() if isinstance(s, torch.Tensor) and s.numel() == C \
        else torch.ones(C, dtype=torch.float32)
    mode = st.get("mode") or 0
    x = hif4.dequantize_nvfp4(pair[0], pair[1]).float()
    x_t = x * s
    if mode == 1:
        x_t = SOL._rot_blocks(x_t)
    p_ship = SOL.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
    v_ship = hif4.hif4_dequantize(p_ship)            # judge space == transformed
    p_base = SOL._quantize_weighted(x_t, torch.ones(1, C, dtype=torch.float32))
    unit = SOL._params_unit_flat(p_base)
    v4 = torch.round(v_ship / unit * 4.0)
    d = 0.25 * unit
    if check:
        p_chk = SOL._values_to_params(v4 * d, p_base)
        assert torch.equal(p_chk["mant"], p_ship["mant"]), "mant mismatch"
        assert torch.equal(p_chk["sign"], p_ship["sign"]), "sign mismatch"
    gw = st["gw"].float()
    gwf = st["gwf"].float()
    return {"x": x_t, "v4": v4, "d": d, "unit": unit, "p_base": p_base,
            "p_ship": p_ship, "gw": gw, "gwf": gwf, "s": s, "mode": mode,
            "T": pair[0].shape[0], "C": C}


def resid_image(ctx):
    """M = v@gw - x@gwf at the current grid point."""
    return (ctx["v4"] * ctx["d"]) @ ctx["gw"] - ctx["x"] @ ctx["gwf"]


def J_raw(ctx):
    """J(v) = <v, v@gw> - 2<v, x@gwf>  (constant dropped; exact up to c)."""
    v = ctx["v4"] * ctx["d"]
    return float(((v @ ctx["gw"]) * v).sum().item()
                 - 2.0 * ((ctx["x"] @ ctx["gwf"]) * v).sum().item())


def single_stats(ctx, M=None):
    """Per-element best single-flip delta A(c) = min over legal s of dJ.
    A >= 0 everywhere iff the point is greedy-converged."""
    if M is None:
        M = resid_image(ctx)
    v4, d, gw = ctx["v4"], ctx["d"], ctx["gw"]
    d2c = (d * d) * gw.diagonal()
    gp = 2.0 * d * M + d2c            # +1 step, legal iff v4 < 7
    gm = -2.0 * d * M + d2c           # -1 step, legal iff v4 > -7
    gp = torch.where(v4 < 7.0, gp, torch.full_like(gp, _INF))
    gm = torch.where(v4 > -7.0, gm, torch.full_like(gm, _INF))
    return M, torch.minimum(gp, gm)


# ---------------------------------------------------------------------------
# cooperative pair scan
# ---------------------------------------------------------------------------

def _exact_pair_dJ(d1, d2, M1, M2, g11, g22, g12, v41, v42):
    """Exact best pair delta + winning signs over the 4 sign combos.
    All inputs (K,) tensors for one row's top-K candidate pairs."""
    best = torch.full_like(d1, _INF)
    bs1 = torch.zeros_like(d1)
    bs2 = torch.zeros_like(d1)
    b1p = v41 < 7.0
    b1m = v41 > -7.0
    b2p = v42 < 7.0
    b2m = v42 > -7.0
    for s1 in (1.0, -1.0):
        l1 = b1p if s1 > 0 else b1m
        t1 = 2.0 * s1 * d1 * M1 + d1 * d1 * g11
        for s2 in (1.0, -1.0):
            l2 = b2p if s2 > 0 else b2m
            dJ = (t1 + 2.0 * s2 * d2 * M2 + d2 * d2 * g22
                  + 2.0 * s1 * s2 * d1 * d2 * g12)
            dJ = torch.where(l1 & l2, dJ, torch.full_like(dJ, _INF))
            better = dJ < best
            best = torch.where(better, dJ, best)
            bs1 = torch.where(better, torch.full_like(bs1, s1), bs1)
            bs2 = torch.where(better, torch.full_like(bs2, s2), bs2)
    return best, bs1, bs2


def pair_scan(ctx, M=None, A=None, topk_n=64):
    """Cooperative-pair scan at the current point.

    For candidate pairs the exact pair delta over all 4 sign combos is
    evaluated (legality-aware).  Prefilters only prune what is PROVABLY
    impossible:
      row level:  min pair delta >= A1 + A2 - 2*dmax^2*max_offdiag|gw| ,
                  skipped when that bound is >= 0
      col level:  c1 can only participate if A(c1) < max_c2 2*d1*d2*|gw12|

    Returns dict with:
      rows_A_neg   rows with any improving single flip (greedy NOT done)
      row_amin     (T,) min single-flip delta per row (convergence margin)
      cand_rows    rows reaching the (S,S) pair enumeration
      coop_rows    rows with >=1 exactly-cooperative pair (dJ < 0)
      Bneg_pairs   candidate-set pairs with B < 0 (necessary condition)
      row_best     (T,) exact best pair dJ (INF where none negative)
      picks        (T,4) long tensor (c1, s1, c2, s2), -1 where none
      sum_dJ       sum of negative row_best  (exact one-pair-sweep gain)
    """
    if M is None or A is None:
        M, A = single_stats(ctx, M)
    v4, d, gw = ctx["v4"], ctx["d"], ctx["gw"]
    T, C = M.shape
    gwa = gw.abs().clone()
    gwa.fill_diagonal_(0.0)
    gwa_off_max = gwa.max().item()
    col2 = gw.diagonal()

    row_best = torch.full((T,), _INF)
    picks = torch.full((T, 4), -1, dtype=torch.long)
    row_amin = torch.full((T,), _INF)
    cand_rows = 0
    coop_rows = 0
    Bneg_total = 0

    for r in range(T):
        Ar = A[r]
        fin = torch.isfinite(Ar)
        if int(fin.sum()) < 2:
            row_amin[r] = _INF
            continue
        row_amin[r] = float(Ar[fin].min())
        two_min = torch.topk(Ar[fin], 2, largest=False).values.sum().item()
        pmax_scal = 2.0 * float(d[r].max()) ** 2 * gwa_off_max
        if two_min >= pmax_scal:
            continue                      # provably no cooperative pair
        cand_rows += 1
        mvec = (d[r].unsqueeze(0) * gwa).max(dim=1).values  # (C,) max_c2 d2|gw12|
        S = torch.nonzero(fin & (Ar < 2.0 * d[r] * mvec),
                          as_tuple=False).squeeze(1)
        if S.numel() < 2:
            continue
        AS = Ar[S]
        dS = d[r][S]
        PS = 2.0 * dS.unsqueeze(1) * dS.unsqueeze(0) * gwa[S][:, S]
        B = AS.unsqueeze(1) + AS.unsqueeze(0) - PS
        Bneg_total += int((B < 0).sum().item())
        Bf = B.flatten()
        k = min(topk_n, Bf.numel())
        idx = torch.topk(Bf, k, largest=False).indices
        i1 = S[idx // S.numel()]
        i2 = S[idx % S.numel()]
        dJ, s1b, s2b = _exact_pair_dJ(d[r][i1], d[r][i2], M[r][i1], M[r][i2],
                                      col2[i1], col2[i2], gw[i1, i2],
                                      v4[r][i1], v4[r][i2])
        j = int(dJ.argmin().item())
        if float(dJ[j]) < 0.0:
            coop_rows += 1
            row_best[r] = float(dJ[j])
            picks[r, 0] = int(i1[j])
            picks[r, 1] = int(s1b[j].item())
            picks[r, 2] = int(i2[j])
            picks[r, 3] = int(s2b[j].item())
    return {"rows_A_neg": int((row_amin < 0).sum().item()),
            "row_amin": row_amin, "cand_rows": cand_rows,
            "coop_rows": coop_rows, "Bneg_pairs": Bneg_total,
            "row_best": row_best, "picks": picks,
            "applied_rows": int((picks[:, 0] >= 0).sum().item()),
            "sum_dJ": float(torch.clamp(row_best, max=0.0).sum().item())}


def apply_picks(ctx, M, picks):
    """Apply one pair per row; update v4 in place and M by the exact rank
    updates.  No return value; scan stats already hold the applied dJ."""
    v4, d, gw = ctx["v4"], ctx["d"], ctx["gw"]
    rows = torch.nonzero(picks[:, 0] >= 0, as_tuple=False).squeeze(1)
    for r in rows.tolist():
        c1, s1, c2, s2 = picks[r].tolist()
        M[r] += (s1 * d[r, c1]) * gw[c1] + (s2 * d[r, c2]) * gw[c2]
        v4[r, c1] += s1
        v4[r, c2] += s2


def reconverge_singles(SOL, ctx, max_sweeps=512):
    """Continue greedy single flips from the current grid to full convergence.
    Returns (sweeps_used, seconds, converged_bool).  A large S is safe: the
    ship round loop early-exits once every row freezes (A == 0)."""
    x, unit, gw, gwf, d = ctx["x"], ctx["unit"], ctx["gw"], ctx["gwf"], ctx["d"]
    vals = ctx["v4"] * d
    t0 = time.perf_counter()
    S = 200
    while True:
        vals = SOL._refine_act_values(x, vals, unit, gw, gwf,
                                      sweep_override=S)
        ctx["v4"] = torch.round(vals / d)      # d = 0.25*unit -> vals/d = v4
        _, A = single_stats(ctx)
        finite = A[torch.isfinite(A)]
        conv = bool((finite >= 0).all()) if finite.numel() else True
        if conv or S >= max_sweeps:
            return S, time.perf_counter() - t0, conv
        S = min(S * 2, max_sweeps)


def pair_machine(SOL, ctx, max_sweeps=40, topk_n=64, verbose=False,
                 rel_eps=1e-11):
    """Iterated pair-flip search: scan -> apply per-row best pair ->
    re-converge singles -> rescan, until no cooperative pair remains.

    dJ accounting is complete: each sweep records BOTH the applied pair dJ
    and the dJ of the single flips the pairs UNLOCKED (re-convergence
    descent) -- the cascade is part of the mechanism's value.

    Returns (stats_list, total_seconds, total_dJ_pairs, total_dJ_singles)."""
    stats = []
    t0 = time.perf_counter()
    tot_pairs = 0.0
    tot_singles = 0.0
    for it in range(max_sweeps):
        M, A = single_stats(ctx)
        st = pair_scan(ctx, M, A, topk_n=topk_n)
        st["iter"] = it
        stats.append(st)
        gain = -st["sum_dJ"]
        J_scale = max(1.0, abs(J_raw(ctx)))
        if st["coop_rows"] == 0 or gain <= rel_eps * J_scale:
            break
        tot_pairs += st["sum_dJ"]
        apply_picks(ctx, M, st["picks"])
        J_before = J_raw(ctx)
        rs, ts, conv = reconverge_singles(SOL, ctx)
        st["dJ_unlocked_singles"] = J_raw(ctx) - J_before
        tot_singles += st["dJ_unlocked_singles"]
        st["reconv_sweeps"] = rs
        st["reconv_s"] = round(ts, 3)
        if verbose:
            print(f"    pair sweep {it}: coop_rows={st['coop_rows']} "
                  f"sum_dJ={st['sum_dJ']:.3e} "
                  f"dJ_singles={st['dJ_unlocked_singles']:.3e} "
                  f"reconv_sweeps={rs} ({ts:.1f}s)", flush=True)
        if not conv:
            break
    return stats, time.perf_counter() - t0, tot_pairs, tot_singles


# ---------------------------------------------------------------------------
# scoring (mini_holdout convention)
# ---------------------------------------------------------------------------

def case_mse_std(w_ref, pair):
    x_ref = hif4.dequantize_nvfp4(*pair)
    ref = hif4.linear_ref(x_ref, w_ref)
    w_std = V.deq(V.quant_alg1(w_ref.float()))
    x_std = V.deq(V.quant_alg1(x_ref.float()))
    mse_std = ((hif4.linear_ref(x_std, w_std) - ref) ** 2).mean().item()
    return x_ref, ref, mse_std


def pp_of_params(p, w_play, ref, mse_std):
    xm = hif4.hif4_dequantize(p)
    mse = ((hif4.linear_ref(xm, w_play) - ref) ** 2).mean().item()
    return (mse_std - mse) / mse_std * 100.0, mse
