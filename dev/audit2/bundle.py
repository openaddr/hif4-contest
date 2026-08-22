"""Audit-2 savings bundle: textual patch of a LOADED COPY of solution v25.

  b1 (C4) _refine_act_values: hoist loop-invariants (-2*d, (d*d)*col2 --
      both (T,C)-sized, recomputed every round in v25), cache the v4 legality
      bounds and maintain them by (T,1) scatter (bounds change only at the
      flipped column), dirn at (T,1) after gather, in-place/out= kernels.
  b2 (C3) _refine_weight_values: same round optimization, plus the
      sweep/round/chunk nest restructured to chunk/sweep/round -- PROOF:
      rows are independent (a flip touches only its own row of v4/A; Gxx is
      read-only), so each row's 20-flip sequence is unchanged; enables
      per-chunk hoisting. REFINE_W_CHUNK 2048 -> 1024 (chunk-invariance
      proven + measured fastest).
  b3 (C2) _quantize_weighted: candidate-batched _quant_chunk_vec threshold
      R*C >= 4M -> 2M (measured crossover; below 2M plain wins/ties).
  b4 (C5) calibration dedupe: hoist a_big @ w[rows].T out of the alpha loop
      (identical operands -> identical matmul), reuse ref (xh_pick @
      w_final.T) for ref2 (identical operands).

Usage: python dev/audit2/bundle.py ab [configs...] | unit | attn
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

# ---------------------------------------------------------------------------
# patch construction
# ---------------------------------------------------------------------------

_A_REFINE_ACT_OLD = '''def _refine_act_values(x: torch.Tensor, values: torch.Tensor,
                       unit: torch.Tensor, gw: torch.Tensor,
                       gwf: torch.Tensor) -> torch.Tensor:'''

_PREAMBLE = '''_ROUND_INF = float("inf")


def _round_hoisted(M, neg2d, d2col, bpos, bneg, g, up, legal, keep):
    """One greedy top-1 round on a row block, bit-identical to
    g, dirn = _flip_sel(d_blk, M, col2, v4_blk) + argmin/apply: same ops in
    the same order with loop-invariants (-2*d, (d*d)*col2) and the v4 legality
    bounds hoisted/cached (bounds change only at the flipped column, maintained
    by scatter)."""
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
    return idx, dr


def _bound_update(v4c, bpos, bneg, idx):
    nv = v4c.gather(1, idx)
    bpos.scatter_(1, idx, nv < 7.0)
    bneg.scatter_(1, idx, nv > -7.0)


def _rounds_np(M, v4, d, neg2d, d2col, gw, n_sweeps, rounds):
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


def _refine_act_values(x: torch.Tensor, values: torch.Tensor,
                       unit: torch.Tensor, gw: torch.Tensor,
                       gwf: torch.Tensor) -> torch.Tensor:'''

_A_REFINE_ACT_BODY_OLD = '''    v4 = torch.round(values / unit * 4.0)
    d = 0.25 * unit
    col2 = gw.diagonal()
    M = (v4 * d) @ gw - x @ gwf
    # T-adaptive sweep depth (only T <= REFINE_T_MAX reaches here)
    T = values.shape[0]
    # sweep curves (synthetic suite, 0.28x judge transfer): no flattening by
    # 12 at any T bucket; s12 pays +321..455 online for +17-29s. Rounds-only
    # changes are bit-identical no-ops (s10 == s5r40) -- raise sweeps only.
    # v23/v24 timeout postmortem: sweep rounds are MEMORY-BOUND (judge ~2x
    # local, not 4.8x) -> s5->s8 at T=1024 costs ~22s online, s12 ~55s. Value
    # per sweep is T-uniform but cost scales with R: spend depth on small T.
    n_sweeps = 12 if T <= 256 else 8 if T <= 512 else 5
    for _ in range(n_sweeps):
        for _ in range(REFINE_ROUNDS):
            g, dirn = _flip_sel(d, M, col2, v4)
            idx = g.argmin(dim=1, keepdim=True)
            fin = torch.isfinite(g.gather(1, idx))
            dr = dirn.gather(1, idx) * fin.float()
            v4.scatter_add_(1, idx, dr)
            M += (dr * d.gather(1, idx)) * gw[idx[:, 0]]
    return v4 * d'''

_A_REFINE_ACT_BODY_NEW = '''    v4 = torch.round(values / unit * 4.0)
    d = 0.25 * unit
    col2 = gw.diagonal()
    M = (v4 * d) @ gw - x @ gwf
    # T-adaptive sweep depth (only T <= REFINE_T_MAX reaches here)
    T = values.shape[0]
    # sweep curves (synthetic suite, 0.28x judge transfer): no flattening by
    # 12 at any T bucket; s12 pays +321..455 online for +17-29s. Rounds-only
    # changes are bit-identical no-ops (s10 == s5r40) -- raise sweeps only.
    # v23/v24 timeout postmortem: sweep rounds are MEMORY-BOUND (judge ~2x
    # local, not 4.8x) -> s5->s8 at T=1024 costs ~22s online, s12 ~55s. Value
    # per sweep is T-uniform but cost scales with R: spend depth on small T.
    n_sweeps = 12 if T <= 256 else 8 if T <= 512 else 5
    neg2d = -2.0 * d          # (T,C) loop-invariant (v25 recomputed/round)
    d2col = (d * d) * col2    # (T,C) loop-invariant
    if T <= 32:
        # tiny-T: torch round loop is dispatch-bound; numpy twin is ~3x faster
        _rounds_np(M, v4, d, neg2d, d2col, gw, n_sweeps, REFINE_ROUNDS)
        return v4 * d
    bpos = v4 < 7.0           # legality bounds; change only at flipped cols
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
    return v4 * d'''

_W_LOOP_OLD = '''    for _ in range(REFINE_W_SWEEPS):
        for _ in range(REFINE_ROUNDS):
            for i1 in range(0, N, REFINE_W_CHUNK):
                i2 = min(i1 + REFINE_W_CHUNK, N)
                g, dirn = _flip_sel(d[i1:i2], A[i1:i2], colE, v4[i1:i2])
                idx = g.argmin(dim=1, keepdim=True)
                fin = torch.isfinite(g.gather(1, idx))
                dr = dirn.gather(1, idx) * fin.float()
                v4[i1:i2].scatter_add_(1, idx, dr)
                A[i1:i2] += (dr * d[i1:i2].gather(1, idx)) * Gxx[idx[:, 0]]'''

_W_LOOP_NEW = '''    # chunk-outer restructure: rows are independent (a flip touches only its
    # own row of v4/A; Gxx is read-only), so each row's flip sequence is
    # bit-identical under either nesting; enables per-chunk hoisting.
    for i1 in range(0, N, REFINE_W_CHUNK):
        i2 = min(i1 + REFINE_W_CHUNK, N)
        neg2d = -2.0 * d[i1:i2]
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
                Ac += gb'''

_ALPHA_OLD = '''    rows = torch.randperm(R)[: min(R, 256)]
    best_alpha = 0.0
    best_loss = None
    for alpha in ALPHA_GRID:
        s = torch.exp(logm * alpha)
        wp = _quant_weight_fast(w[rows] / s, torch.ones(1, C))'''

_ALPHA_NEW = '''    rows = torch.randperm(R)[: min(R, 256)]
    w_rows = w[rows]                      # identical gather, hoisted
    a_wr = a_big @ w_rows.T               # identical operands -> same matmul
    best_alpha = 0.0
    best_loss = None
    for alpha in ALPHA_GRID:
        s = torch.exp(logm * alpha)
        wp = _quant_weight_fast(w_rows / s, torch.ones(1, C))'''

_LOSS_OLD = '''        loss = ((a_big @ wq.T - a_big @ w[rows].T) ** 2).mean().item()'''
_LOSS_NEW = '''        loss = ((a_big @ wq.T - a_wr) ** 2).mean().item()'''

_WFIN_OLD = '''    w_final = tf_final(w_s)'''
_WFIN_NEW = '''    w_final = tf_final(w_s)
    _ref_shared = None'''
_REF_OLD = '''        q_g = _gptq_quantize_values(w_final, unit, Uw)
        ref = xh_pick @ w_final.T'''
_REF_NEW = '''        q_g = _gptq_quantize_values(w_final, unit, Uw)
        ref = xh_pick @ w_final.T
        _ref_shared = ref                  # reused below (same operands)'''

_REF2_OLD = '''            ref2 = xh_pick @ w_final.T'''
_REF2_NEW = '''            ref2 = (_ref_shared if _ref_shared is not None
                    else xh_pick @ w_final.T)'''

_THR_OLD = '''    fn = _quant_chunk_vec if R * C >= 4_000_000 else _quant_chunk'''
_THR_NEW = '''    fn = _quant_chunk_vec if R * C >= 2_000_000 else _quant_chunk'''

_CHUNK_OLD = '''REFINE_W_CHUNK = 2048       # weight-row chunk for the greedy sweep'''
_CHUNK_NEW = '''REFINE_W_CHUNK = 1024       # weight-row chunk for the greedy sweep
                            # (rows independent -> chunk size bit-inert;
                            # 1024 measured fastest, best cache locality)'''


def build_patch():
    src = harness.src_text()
    subs = [
        (_CHUNK_OLD, _CHUNK_NEW),
        (_WFIN_OLD, _WFIN_NEW),
        (_THR_OLD, _THR_NEW),
        (_ALPHA_OLD, _ALPHA_NEW),
        (_LOSS_OLD, _LOSS_NEW),
        (_REF_OLD, _REF_NEW),
        (_REF2_OLD, _REF2_NEW),
        (_A_REFINE_ACT_OLD, _PREAMBLE),
        (_A_REFINE_ACT_BODY_OLD, _A_REFINE_ACT_BODY_NEW),
        (_W_LOOP_OLD, _W_LOOP_NEW),
    ]
    for old, new in subs:
        assert src.count(old) == 1, f"anchor not unique/found: {old[:60]!r}"
        src = src.replace(old, new)
    return src


# ---------------------------------------------------------------------------
# A/B + bit-identity
# ---------------------------------------------------------------------------

def run_once(sol, g):
    torch.manual_seed(0)
    t0 = time.perf_counter()
    out = sol.hif4_calibration_and_quantize_weight(
        g["weight"][0], g["weight"][1], g["calib_activation_list"])
    t_cal = time.perf_counter() - t0
    st = out["activation_state"]
    t_dyn = 0.0
    ps = []
    for pair in g["test_activation_list"]:
        t0 = time.perf_counter()
        p = sol.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
        t_dyn += time.perf_counter() - t0
        ps.append(p)
    return out, ps, t_cal, t_dyn


def cmd_ab(names):
    base = harness.load_variant()
    combo = harness.load_variant(patch_src=build_patch())
    reps = 3
    tot_b = tot_c = 0.0
    for name in names:
        g = harness.load_group(name)
        cals_b, cals_c, dyns_b, dyns_c = [], [], [], []
        ok = None
        for r in range(reps):
            ob, pb, tcb, tdb = run_once(base, g)
            oc, pc, tcc, tdc = run_once(combo, g)
            cals_b.append(tcb); cals_c.append(tcc)
            dyns_b.append(tdb); dyns_c.append(tdc)
            if r == 0:
                ok = (harness.eq_params(ob["weight_params"], oc["weight_params"])
                      and harness.eq_state(ob["activation_state"],
                                           oc["activation_state"])
                      and all(harness.eq_params(a, b) for a, b in zip(pb, pc)))
        mc_b, mc_c = statistics.median(cals_b), statistics.median(cals_c)
        md_b, md_c = statistics.median(dyns_b), statistics.median(dyns_c)
        tot_b += mc_b + md_b
        tot_c += mc_c + md_c
        print(f"{name}: calib {mc_b:6.2f} -> {mc_c:6.2f} ({mc_b - mc_c:+6.2f}) | "
              f"dyn {md_b:6.2f} -> {md_c:6.2f} ({md_b - md_c:+6.2f}) | "
              f"bit-identical: {ok}")
        sys.stdout.flush()
    print(f"TOTAL saved (these configs): {tot_b - tot_c:+.2f} local s")


def cmd_gate():
    """Bit-identity across all 4 synth configs + 2 extra seeds + real mini."""
    base = harness.load_variant()
    combo = harness.load_variant(patch_src=build_patch())
    ok = True
    for name, _, _, _, _ in harness.CONFIGS:
        g = harness.load_group(name)
        ok = ok and harness.check_linear(base, combo, g, name)
    for seed in (4242, 5151):
        import synth
        g = synth.make_linear_group(seed, 8192, 2048, tokens=(10, 128, 512, 1024),
                                    spread=0.5, outlier_p=0.0, w_spread=0.3)
        ok = ok and harness.check_linear(base, combo, g, f"c2048_n8192 seed{seed}")
    lin = torch.load(os.path.join(harness.MINI, "linear.pt"),
                     weights_only=True)[0]
    ok = ok and harness.check_linear(base, combo, lin, "mini linear (real)")
    # attention (threshold b3 affects _quantize_weighted used by attn paths)
    att = torch.load(os.path.join(harness.MINI, "attn.pt"), weights_only=True)[0]
    qh, kvh, dh = att["q_num_heads"], att["kv_num_heads"], att["head_dim"]
    torch.manual_seed(0)
    ab = base.hif4_calibration_attention(att["calib"], qh, kvh, dh)
    torch.manual_seed(0)
    av = combo.hif4_calibration_attention(att["calib"], qh, kvh, dh)
    oka = all(harness.eq_state(ab[k], av[k]) for k in ("q_state", "k_state"))
    dyn_ok = True
    for smp in att["test"]:
        for mb, mv in ((base, combo),):
            mb._QKV_CARRY.clear(); mv._QKV_CARRY.clear()
            mb._VCOMP.update({"n": 0, "el": 0.0}); mv._VCOMP.update({"n": 0, "el": 0.0})
            pq = mb.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh, dh, ab["q_state"])
            pv = mv.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh, dh, av["q_state"])
            pk = mb.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], kvh, dh, ab["k_state"])
            kv = mv.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], kvh, dh, av["k_state"])
            pb = mb.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], kvh, dh, ab["v_state"])
            pvv = mv.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], kvh, dh, av["v_state"])
            dyn_ok = dyn_ok and (harness.eq_params(pq, pv) and harness.eq_params(pk, kv)
                                 and harness.eq_params(pb, pvv))
    print(f"[bitid] attn mini (states + q/k/v dyn): "
          f"{'PASS' if oka and dyn_ok else 'FAIL'}")
    print("[gate] OVERALL:", "PASS" if ok and oka and dyn_ok else "FAIL")
    return ok and oka and dyn_ok


def cmd_attn_ab(reps=3):
    base = harness.load_variant()
    combo = harness.load_variant(patch_src=build_patch())
    att = torch.load(os.path.join(harness.MINI, "attn.pt"), weights_only=True)[0]
    qh, kvh, dh = att["q_num_heads"], att["kv_num_heads"], att["head_dim"]
    tb, tc = [], []
    for r in range(reps):
        for sol, acc in ((base, tb), (combo, tc)):
            torch.manual_seed(0)
            t0 = time.perf_counter()
            acal = sol.hif4_calibration_attention(att["calib"], qh, kvh, dh)
            t = time.perf_counter() - t0
            for smp in att["test"]:
                sol._QKV_CARRY.clear()
                sol._VCOMP.update({"n": 0, "el": 0.0})
                t0 = time.perf_counter()
                sol.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh, dh, acal["q_state"])
                sol.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], kvh, dh, acal["k_state"])
                sol.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], kvh, dh, acal["v_state"])
                t += time.perf_counter() - t0
            acc.append(t)
    print(f"attn mini: base {statistics.median(tb):.3f}s -> "
          f"combo {statistics.median(tc):.3f}s "
          f"(save {statistics.median(tb) - statistics.median(tc):+.3f}s)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "ab"
    if mode == "ab":
        cmd_ab(sys.argv[2:] or [c[0] for c in harness.CONFIGS])
    elif mode == "gate":
        cmd_gate()
    elif mode == "attn":
        cmd_attn_ab()
    elif mode == "src":
        print(build_patch()[:200])
    else:
        raise SystemExit(__doc__)
