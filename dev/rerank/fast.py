"""dev/rerank/fast.py: slimmed outlier-group act grid re-rank (decomp2 3iii).

PROTOTYPE COST STRUCTURE (dev/decomp2/anatomy.py act_rerank, reflip branch,
passes=3; profiled in profile_proto.py, C=2048 outlier, T=10):
  91%  FULL_refine     96 per-block _refine_act_values (32-40 sweeps x 20
                       numpy rounds; one FULL refinement per block-iteration)
   4%  Jt_try matmuls  2x (T,C)@(C,C) per block (x@gwf recomputed per block)
   2%  cand_requant    block_values_batched on tiny per-block tensors
  <1%  dJ einsums, clones, scatters.

MECHANISM REDISCOVERY (instrumented, _tune_tmp.py): the prototype's dJ
candidate ranking is dead weight -- after the ship refinement the incumbent
is locally optimal, so EVERY candidate's plain dJ is > 0, argmin picks the
identity row (97-100% of tries), and the clamp maps it to the FIXED last
candidate (sf offset+1 sig 1.25, uniform lv2=lv3=2 -- the coarsest grid).
Accepted swaps are ~100% that path: the mechanism is per-block re-grid +
re-refinement (basin hopping on the carried-Gram objective), NOT plain
candidate improvement.  The slim version therefore swaps masked blocks
directly onto the coarse candidate (use_dj=True keeps the vectorized dJ
selection: batched block_values_batched + einsum over all active pairs of
the iteration; measured to pick a different candidate on 0-2.8% of tries).

SLIM ALGORITHM (rows are independent in J and in the greedy flips, so the
prototype's block-major Gauss-Seidel per row is reproduced with all rows
perturbed in parallel -- one perturbed 64-block per row per iteration):
  (a) mask: only (row, 64-block) pairs with grid-limited evidence (any v4
      pinned at the +-7 bound after the ship refinement; cheap v4 mask).
      NOTE: measured dense (~97% of pairs) on these groups -- it is a
      correct detector but the sparsity premise fails at 64-block
      granularity; the time win comes from (c)-(e).
  (b) only masked pairs are perturbed/refined (never the whole (T, C) grid).
  (c) incremental objective: M = v @ gw - x @ gwf maintained exactly via
      per-pair rank-64 updates + the flip loop's in-place M updates; J and
      acceptance from M -- zero (T,C)@(C,C) matmuls after init.
  (d) candidates batched across the iteration's pairs (einsum/gather), or
      skipped entirely (default, measured equivalent).
  (e) re-flip acceptance: ref_sweeps (1-3) sweeps at the T<=256 tier cost
      after the per-block swap; accept per ROW on the exact J_t (rows are
      independent -> exact accept/revert, same criterion as the prototype).

Subcommands:
  check [--C ..] [--sweeps k] [--passes k] [--dj] [--maxp N]
  time  (3 reps per (T, C) bucket, outlier groups; medians)

solution.py is NEVER modified; dev/decomp2 is only read.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.dirname(HERE)
ROOT = os.path.dirname(DEV)
D2 = os.path.join(DEV, "decomp2")
for _p in (DEV, D2):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import anatomy as A      # noqa: E402
import hif4 as H         # noqa: E402
import study2 as S2      # noqa: E402
import variants as V     # noqa: E402

SOL = S2.sol()
RES = os.path.join(HERE, "results.json")


# ---------------------------------------------------------------------------
# short refinement (ship flip machinery, custom sweep count; M in place)
# ---------------------------------------------------------------------------
def refine_short(v, unit, M, gw, n_sweeps):
    """Greedy top-1 lattice refinement, n_sweeps sweeps at the ship tier cost
    (REFINE_ROUNDS rounds each).  Same ops/order as _refine_act_values; M and
    v4 are updated in place (M stays the exact residual image)."""
    v4 = torch.round(v / unit * 4.0)
    d = 0.25 * unit
    col2 = gw.diagonal()
    neg2d = -2.0 * d
    d2col = (d * d) * col2
    T = v.shape[0]
    if T <= 32:
        SOL._rounds_np(M, v4, d, neg2d, d2col, gw, n_sweeps, SOL.REFINE_ROUNDS)
    else:
        bpos = v4 < 7.0
        bneg = v4 > -7.0
        g = torch.empty_like(M)
        up = torch.empty(M.shape, dtype=torch.bool)
        legal = torch.empty(M.shape, dtype=torch.bool)
        keep = torch.empty(M.shape, dtype=torch.bool)
        gb = torch.empty_like(M)
        for _ in range(n_sweeps):
            for _ in range(SOL.REFINE_ROUNDS):
                idx, dr = SOL._round_hoisted(M, neg2d, d2col, bpos, bneg,
                                             g, up, legal, keep)
                v4.scatter_add_(1, idx, dr)
                SOL._bound_update(v4, bpos, bneg, idx)
                coef = dr * d.gather(1, idx)
                torch.index_select(gw, 0, idx[:, 0], out=gb)
                gb.mul_(coef)
                M += gb
    return v4 * d


def _cand_last(x_blk):
    """The prototype's identity-clamp candidate (index K5-2): sf candidate
    (exp offset +1, sig 1.25) with uniform lv2 = lv3 = 2.  Bit-identical to
    block_values_batched's last entry (same op sequence)."""
    ab = x_blk.abs()
    amax = ab.amax(dim=1)
    e0 = torch.floor(torch.log2((amax / 7.0).clamp_min(1e-38)))
    sf = (torch.exp2(e0 + 1.0) * 1.25).clamp(SOL.SF_MIN, SOL.SF_MAX)
    u = (sf * 4.0).unsqueeze(1)
    mant = torch.clamp(torch.round(ab / u * 4.0) / 4.0, 0.0, 1.75)
    V = mant * u * torch.sign(x_blk)
    return V, sf, u.expand_as(V)


def _tadd(tm, key, t0):
    tm[key] = tm.get(key, 0.0) + (time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# persistent refine state: hoists the per-call invariants (v4, d, neg2d,
# d2col, numpy views) that _refine_act_values recomputes on every call, so
# one try costs only its rounds.  Flip semantics: the ship's own
# SOL._rounds_np, bit-exact.
# ---------------------------------------------------------------------------


class _RefineState:
    def __init__(self, v, unit, M, gw):
        self.gw = gw
        self.v4 = torch.round(v / unit * 4.0)
        self.d = 0.25 * unit
        self.M = M
        self.neg2d = -2.0 * self.d
        self.col2n = gw.diagonal().contiguous().numpy()
        self.d2col = (self.d * self.d) * gw.diagonal()
        self.v4n = self.v4.numpy()
        self.dn = self.d.numpy()
        self.Mn = self.M.numpy()
        self.nn2 = self.neg2d.numpy()
        self.d2c = self.d2col.numpy()

    def perturb(self, t, b, V, u_p, gwl):
        """Swap row t's block b to candidate values V (torch, 64) on unit
        u_p (torch, 64); gwl[b] = gw[b*64:(b+1)*64, :].  Returns the saved
        pre-try ROW state (v4/M saved row-wide: the rounds flip columns
        OUTSIDE the swapped block too, and a rejected try must revert the
        whole row exactly, like the prototype's per-row accept/revert)."""
        sl = slice(b * 64, (b + 1) * 64)
        saved = (self.v4n[t].copy(), self.Mn[t].copy(),
                 self.dn[t, sl].copy(), self.nn2[t, sl].copy(),
                 self.d2c[t, sl].copy())
        d_new = (0.25 * u_p).numpy()
        v4_new = torch.round(V / u_p * 4.0).numpy()
        v_old = torch.from_numpy(self.v4n[t, sl] * self.dn[t, sl])
        self.dn[t, sl] = d_new
        self.nn2[t, sl] = -2.0 * d_new
        self.d2c[t, sl] = d_new * d_new * self.col2n[sl]
        self.v4n[t, sl] = v4_new
        self.Mn[t] += ((V - v_old) @ gwl[b]).numpy()
        return saved

    def restore(self, t, b, saved):
        sl = slice(b * 64, (b + 1) * 64)
        v4r, Mr, ds, n2s, d2s = saved
        self.v4n[t] = v4r
        self.Mn[t] = Mr
        self.dn[t, sl] = ds
        self.nn2[t, sl] = n2s
        self.d2c[t, sl] = d2s

    def rounds(self, n_sweeps):
        """The ship's own numpy round loop (bit-exact flip semantics) on the
        persistent buffers; M and v4 are updated in place."""
        SOL._rounds_np(self.M, self.v4, self.d, self.neg2d, self.d2col,
                       self.gw, n_sweeps, SOL.REFINE_ROUNDS)

    def Jrow(self, B):
        return ((self.v4 * self.d) * (self.M - B)).sum(dim=1)

    def values(self):
        return self.v4 * self.d


# ---------------------------------------------------------------------------
# the slim re-rank
# ---------------------------------------------------------------------------
def rerank_fast(x, v_in, unit_in, p_ship, st, cands=None, passes=3,
                ref_sweeps=2, use_dj=False, max_pairs=None, sat_tol=6.999,
                eps=1e-9):
    """Same contract as anatomy.act_rerank(reflip=True): returns
    (v, unit, sf_sel, lv2_sel, lv3_sel, info).  info: m (masked pairs),
    J hist, accepted counts, per-section wall times, pair index."""
    t_all = time.perf_counter()
    tm: dict[str, float] = {}
    if cands is None:
        cands = SOL.CAND_GRID
    gw = st["gw"].float()
    gwf = st["gwf"].float()
    T, C = x.shape
    if ref_sweeps <= 0:      # deploy config: spend the round budget per C
        ref_sweeps = 4 if C <= 1024 else 1
        if C > 1024 and T <= 32 and (max_pairs is None or max_pairs > 240):
            max_pairs = 240      # cap the try count to the time budget
    nb = C // 64
    v = v_in.clone()
    unit = unit_in.clone()
    sf_sel = p_ship["scale_factor"].reshape(T, nb).clone()
    lv2_sel = p_ship["scale_lv2"].reshape(T, nb, 8).clone()
    lv3_sel = p_ship["scale_lv3"].reshape(T, nb, 8, 2).clone()

    # ---- (a) grid-limited evidence mask from v4 --------------------------
    t0 = time.perf_counter()
    v4 = torch.round(v / unit * 4.0)
    blkmask = (v4.abs() >= sat_tol).reshape(T, nb, 64).any(dim=2)
    pairs = blkmask.nonzero(as_tuple=False)          # (m, 2)
    m = int(pairs.shape[0])
    info: dict = {"m": m, "tm": tm, "J_hist": [], "acc_hist": [],
                  "pairs": (pairs[:, 0], pairs[:, 1])}
    if m == 0:
        info["dt"] = time.perf_counter() - t_all
        return v, unit, sf_sel, lv2_sel, lv3_sel, info
    sel = [pairs[pairs[:, 0] == t, 1].tolist() for t in range(T)]
    if max_pairs is not None and m > max_pairs:
        # budget the try count: global top max_pairs by bound-pinned element
        # count (the gains concentrate in the high-pressure outlier rows),
        # then cap the per-row length so the iteration count (= max row
        # length) stays bounded -- one hot row must not hold the budget.
        press = (v4.abs() >= sat_tol).reshape(T, nb, 64).sum(dim=2)
        key = press[pairs[:, 0], pairs[:, 1]] * (T * nb) \
            + pairs[:, 0] * nb + pairs[:, 1]
        keep = key.topk(max_pairs, largest=True).indices
        kp = set(map(tuple, pairs[keep].tolist()))
        sel = [[b for b in sel[t] if (t, b) in kp] for t in range(T)]
        row_cap = 28 if T <= 32 else max(2, int(max_pairs) // T)
        sel2 = []
        for t in range(T):
            bs = sel[t]
            if len(bs) > row_cap:
                bs = sorted(bs, key=lambda b: (-int(press[t, b]), b))[:row_cap]
            sel2.append(bs)
        sel = sel2
        info["m_capped"] = int(sum(len(s) for s in sel))
    x3 = x.reshape(T, nb, 64)
    gwl = [gw[b * 64:(b + 1) * 64, :] for b in range(nb)]
    _tadd(tm, "mask", t0)

    # ---- (c) exact residual image M = v @ gw - x @ gwf -------------------
    t0 = time.perf_counter()
    B = x @ gwf
    M = v @ gw - B
    RS = _RefineState(v, unit, M, gw)
    Jt = RS.Jrow(B)
    _tadd(tm, "init_M", t0)
    info["J_hist"].append(float(Jt.sum()))
    Gww = None
    if use_dj:
        gwl2 = [gw[b * 64:(b + 1) * 64, b * 64:(b + 1) * 64] for b in range(nb)]
        Gww = torch.stack(gwl2)
    max_len = max(len(s) for s in sel)
    for it in range(passes):
        t0p = time.perf_counter()
        moved = 0
        for j in range(max_len):
            act = [(t, sel[t][j]) for t in range(T) if j < len(sel[t])]
            if not act:
                continue
            t0 = time.perf_counter()
            if use_dj:                              # (d) batched dJ ranking
                rowsl = torch.tensor([t for t, _ in act])
                blkl = torch.tensor([b for _, b in act])
                V5, sfV, l2V, l3V = A.block_values_batched(
                    x3[rowsl, blkl], cands)
                vcur = (RS.v4 * RS.d).reshape(T, nb, 64)[rowsl, blkl]
                d = V5 - vcur.unsqueeze(0)                # (K5, p, 64)
                Mg = RS.M.reshape(T, nb, 64)[rowsl, blkl]
                dJ = torch.empty(V5.shape[0] + 1, len(act))
                dJ[:V5.shape[0]] = (2.0 * (d * Mg.unsqueeze(0)).sum(-1)
                                    + torch.einsum('kpc,pce,kpe->kp', d,
                                                   Gww[blkl], d))
                dJ[V5.shape[0]] = 0.0
                kstar = dJ.argmin(dim=0)
                ks = kstar.clamp_max(V5.shape[0] - 1)
                ar = torch.arange(len(act))
                V = V5[ks, ar]
                sf_p = sfV[ks, ar]
                u_p = A._unit_of(sf_p, l2V[ks, ar], l3V[ks, ar])
                l2_p = l2V[ks, ar]
                l3_p = l3V[ks, ar]
            else:   # measured: kstar is the identity row on 97-100% of
                    # tries -> the swap target is always cand K5-2
                xb = torch.stack([x3[t, b] for t, b in act])
                V, sf_p, u_p = _cand_last(xb)
                l2_p = torch.full((len(act), 8), 2.0)
                l3_p = torch.full((len(act), 8, 2), 2.0)
            _tadd(tm, "cand", t0)

            # ---- the try: one perturbed 64-block per row + short refine --
            t0 = time.perf_counter()
            saveds = {}
            for i, (t, b) in enumerate(act):
                saveds[i] = RS.perturb(t, b, V[i], u_p[i], gwl)
            RS.rounds(ref_sweeps)
            Jt_try = RS.Jrow(B)
            better = Jt_try < Jt - eps
            moved += int(better.sum())
            for i, (t, b) in enumerate(act):
                if bool(better[t]):
                    sf_sel[t, b] = sf_p[i]
                    lv2_sel[t, b] = l2_p[i]
                    lv3_sel[t, b] = l3_p[i]
                else:
                    RS.restore(t, b, saveds[i])
            # rejected rows restore exactly, accepted rows keep Jt_try
            Jt = torch.where(better, Jt_try, Jt)
            _tadd(tm, "try+refine", t0)
        _tadd(tm, f"pass{it}", t0p)
        info["J_hist"].append(float(Jt.sum()))
        info["acc_hist"].append(moved)
        if moved == 0:
            break
    info["dt"] = time.perf_counter() - t_all
    return RS.values(), RS.d * 4.0, sf_sel, lv2_sel, lv3_sel, info


# ---------------------------------------------------------------------------
# scoring / comparison harness (decomp2 groups, ship cal states)
# ---------------------------------------------------------------------------
def load_case(name, t_idx):
    group = S2.build_group(name)
    cc = torch.load(os.path.join(S2.CACHE, f"{name}_ship.pt"),
                    weights_only=True)
    st = cc["cal"]["activation_state"]
    wp = cc["cal"]["weight_params"]
    pair = group["test_activation_list"][t_idx]
    w_ref = H.dequantize_nvfp4(*group["weight"])
    w_std = V.deq(V.quant_alg1(w_ref.float()))
    x_ref = H.dequantize_nvfp4(*pair)
    ref = H.linear_ref(x_ref, w_ref)
    x_std = V.deq(V.quant_alg1(x_ref.float()))
    mse_std = ((H.linear_ref(x_std, w_std) - ref) ** 2).mean().item()
    w_play = H.hif4_dequantize(wp)
    p_ship = SOL.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
    xq_ship = H.hif4_dequantize(p_ship)
    mse_play = ((H.linear_ref(xq_ship, w_play) - ref) ** 2).mean().item()
    torch.manual_seed(0)
    ints = A.dynamic_internals(pair, st)
    return {"pair": pair, "st": st, "mse_std": mse_std, "mse_play": mse_play,
            "w_play": w_play, "ref": ref, "ints": ints}


def tie_audit(x, v_p, st, rows, blocks, sf_p, l2_p, l3_p, sf_f, l2_f, l3_f,
              eq, rel_tol=1e-6):
    """For mismatched pairs: is the two grids' objective a tie (<1e-6 rel)?
    Evaluated in the PROTOTYPE's final state: plain-rounded block values on
    each chosen grid, exact block delta dF = 2*d.M + d gw_sub d."""
    gw = st["gw"].float()
    gwf = st["gwf"].float()
    T, C = x.shape
    nb = C // 64
    B = x @ gwf
    M_p = v_p @ gw - B
    Jrow = (v_p * (M_p - B)).sum(dim=1)
    Mr3 = M_p.reshape(T, nb, 64)
    x3 = x.reshape(T, nb, 8, 2, 4)
    bad = (~eq).nonzero(as_tuple=False).flatten()
    ties = 0
    rels = []
    for i in bad.tolist():
        t = int(rows[i])
        b = int(blocks[i])
        sl = slice(b * 64, (b + 1) * 64)
        G = gw[sl, sl]
        Mg = Mr3[t, b]
        dF = {}
        for tag, sf_s, l2_s, l3_s in ((b"p", sf_p, l2_p, l3_p),
                                      (b"f", sf_f, l2_f, l3_f)):
            u = (sf_s[t, b] * l2_s[t, b].reshape(8, 1)
                 * l3_s[t, b]).reshape(8, 2, 1).expand(8, 2, 4).reshape(16, 4)
            blk = x3[t, b].reshape(16, 4)
            Vg = torch.clamp(torch.round(blk.abs() / u * 4.0) / 4.0,
                             0.0, 1.75) * u * torch.sign(blk)
            dlt = (Vg - v_p[t, sl].reshape(16, 4)).reshape(64)
            dF[tag] = float(2.0 * (dlt * Mg).sum() + (dlt @ G @ dlt))
        rel = abs(dF[b"f"] - dF[b"p"]) / max(abs(float(Jrow[t])), 1e-30)
        rels.append(rel)
        ties += int(rel <= rel_tol)
    return len(bad), ties, rels


def run_check(c_filter, sweeps, accept_kw, passes, use_dj, max_pairs):
    grid = [g for g in S2.iter_grid(c_filter)
            if os.path.exists(os.path.join(S2.CACHE, f"{g[0]}_ship.pt"))
            and g[2] <= 2048]
    print(f"[check] {len(grid)} groups  sweeps={sweeps} passes={passes} "
          f"dj={use_dj} maxp={max_pairs}")
    res = {}
    for name, seed, C, N, spread, outp in grid:
        cs = load_case(name, 0)
        ints = cs["ints"]
        x = ints["x"]
        torch.manual_seed(0)
        t0 = time.perf_counter()
        v_p, _, sf_p, l2_p, l3_p, _ = A.act_rerank(
            x, ints["v1"], ints["unit"], ints["p"], cs["st"], SOL.CAND_GRID,
            passes=3)
        dt_p = time.perf_counter() - t0
        mse_p = ((H.linear_ref(H.hif4_dequantize(
            A.act_params_from(v_p, sf_p, l2_p, l3_p)[0]), cs["w_play"])
            - cs["ref"]) ** 2).mean().item()
        torch.manual_seed(0)
        t0 = time.perf_counter()
        v_f, _, sf_f, l2_f, l3_f, info = rerank_fast(
            x, ints["v1"], ints["unit"], ints["p"], cs["st"],
            passes=passes, ref_sweeps=sweeps, use_dj=use_dj,
            max_pairs=max_pairs)
        dt_f = time.perf_counter() - t0
        mse_f = ((H.linear_ref(H.hif4_dequantize(
            A.act_params_from(v_f, sf_f, l2_f, l3_f)[0]), cs["w_play"])
            - cs["ref"]) ** 2).mean().item()
        rows, blocks = info["pairs"]
        eq = ((sf_p[rows, blocks] == sf_f[rows, blocks])
              & (l2_p[rows, blocks] == l2_f[rows, blocks]).all(-1)
              & (l3_p[rows, blocks] == l3_f[rows, blocks]).flatten(1).all(-1))
        nbad, nties, rels = tie_audit(x, v_p, cs["st"], rows, blocks,
                                      sf_p, l2_p, l3_p, sf_f, l2_f, l3_f, eq)
        res[name] = {"C": C, "outp": outp, "spread": spread,
                     "m": info["m"], "match": float(eq.float().mean()),
                     "n_bad": nbad, "n_tie": nties,
                     "proto_d_pp": (cs["mse_play"] - mse_p) / cs["mse_std"] * 100,
                     "fast_d_pp": (cs["mse_play"] - mse_f) / cs["mse_std"] * 100,
                     "dt_proto": dt_p, "dt_fast": dt_f,
                     "tm": info["tm"], "acc": info["acc_hist"],
                     "speedup": dt_p / max(dt_f, 1e-9)}
        print(f"[check] {name}: m={info['m']:>3} match={res[name]['match']*100:5.1f}% "
              f"bad/tie={nbad}/{nties} d_pp proto {res[name]['proto_d_pp']:+.2f} "
              f"fast {res[name]['fast_d_pp']:+.2f} "
              f"dt {dt_p:.2f}->{dt_f*1000:.0f}ms x{res[name]['speedup']:.0f}")
        sys.stdout.flush()
    with open(RES, "w") as f:
        json.dump(res, f, indent=1)
    summ(res)


def run_time(passes=3):
    out = {}
    for C in (512, 1024, 2048):
        for t_idx, T in ((0, 10), (1, 128)):
            dts = []
            ms = []
            dpps = []
            grid = [g for g in S2.iter_grid({C})
                    if g[5] == 0.002
                    and os.path.exists(os.path.join(S2.CACHE, f"{g[0]}_ship.pt"))]
            for name, *_ in grid:
                cs = load_case(name, t_idx)
                torch.manual_seed(0)
                rep = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    _, _, _, _, _, info = rerank_fast(
                        cs["ints"]["x"], cs["ints"]["v1"], cs["ints"]["unit"],
                        cs["ints"]["p"], cs["st"], passes=passes,
                        ref_sweeps=-1,
                        max_pairs=None if T <= 32 else 256)
                    rep.append(time.perf_counter() - t0)
                rep.sort()
                dts.append(rep[1])
                ms.append(info.get("m_capped", info["m"]))
                # score delta of the deploy config (T=128 behaviour evidence)
                v_f, _, sf_f, l2_f, l3_f, _ = rerank_fast(
                    cs["ints"]["x"], cs["ints"]["v1"], cs["ints"]["unit"],
                    cs["ints"]["p"], cs["st"], passes=passes, ref_sweeps=-1,
                    max_pairs=None if T <= 32 else 256)
                mse_f = ((H.linear_ref(H.hif4_dequantize(
                    A.act_params_from(v_f, sf_f, l2_f, l3_f)[0]), cs["w_play"])
                    - cs["ref"]) ** 2).mean().item()
                dpps.append((cs["mse_play"] - mse_f) / cs["mse_std"] * 100)
            out[f"T{T}_C{C}"] = {"median_dt": sorted(dts)[len(dts) // 2],
                                 "per_group": dts, "m": ms, "n": len(dts),
                                 "d_pp": dpps}
            print(f"[time] T={T} C={C}: median {out[f'T{T}_C{C}']['median_dt']*1000:.0f}ms "
                  f"(per-group {[f'{d*1000:.0f}' for d in dts]}ms, m={ms}, "
                  f"d_pp {['%.2f' % d for d in dpps]})")
            sys.stdout.flush()
    with open(os.path.join(HERE, "results_time.json"), "w") as f:
        json.dump(out, f, indent=1)


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def summ(res):
    out = [n for n in res if res[n]["outp"] > 0]
    cl = [n for n in res if res[n]["outp"] == 0]
    print(f"\ngroups {len(res)} (outlier {len(out)}, clean {len(cl)})")
    print(f"match rate: mean {_mean([res[n]['match'] for n in res])*100:.2f}% "
          f"min {min(res[n]['match'] for n in res)*100:.2f}% "
          f"pairs {sum(res[n]['m'] for n in res)} "
          f"bad {sum(res[n]['n_bad'] for n in res)} "
          f"tie-of-bad {sum(res[n]['n_tie'] for n in res)}")
    print(f"outlier d_pp: proto {_mean([res[n]['proto_d_pp'] for n in out]):+.3f} "
          f"fast {_mean([res[n]['fast_d_pp'] for n in out]):+.3f}")
    print(f"clean   d_pp: proto {_mean([res[n]['proto_d_pp'] for n in cl]):+.3f} "
          f"fast {_mean([res[n]['fast_d_pp'] for n in cl]):+.3f}")
    print(f"dt: proto {_mean([res[n]['dt_proto'] for n in res]):.2f}s "
          f"fast {_mean([res[n]['dt_fast'] for n in res])*1000:.0f}ms "
          f"speedup x{_mean([res[n]['speedup'] for n in res]):.0f}")
    tm = {}
    for n in res:
        for k, val in res[n]["tm"].items():
            if not k.startswith("pass"):
                tm[k] = tm.get(k, 0.0) + val / len(res)
    print("fast sections (ms/group):",
          {k: f"{v*1000:.1f}" for k, v in sorted(tm.items(), key=lambda kv: -kv[1])})


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    args = sys.argv[2:]
    c_filter = None
    sweeps, passes, use_dj, max_pairs = 1, 3, False, None
    for i, a in enumerate(args):
        if a == "--C":
            c_filter = set(int(x) for x in args[i + 1].split(","))
        elif a == "--sweeps":
            sweeps = int(args[i + 1])
        elif a == "--passes":
            passes = int(args[i + 1])
        elif a == "--dj":
            use_dj = True
        elif a == "--maxp":
            max_pairs = int(args[i + 1])
    if mode == "check":
        run_check(c_filter, sweeps, None, passes, use_dj, max_pairs)
    elif mode == "time":
        run_time(passes)
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
