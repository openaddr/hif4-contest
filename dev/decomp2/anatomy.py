"""decomp2/anatomy: residual anatomy WITHIN refined calls (task 3), at T=10
(deep sweep tier: 24 sweeps at study start, 32 after the concurrent v30
bump -- the probe sweeps 96 and checks bit-identity, so the tier value at
run time is recorded via the s96 comparison) on the CURRENT solution, ship
cal states from study2 pop.

Per refined group:
  A. replicate the dynamic call internals (x, unit, v0, v1) and ASSERT the
     replicated refined values reproduce the ship call bit-exactly.
  B. output-space error decomposition: mse_play, mse_act, mse_w, cross.
  C. lattice convergence: best remaining single-flip gain (>=0 == converged);
     96-sweep probe (bit-identical result == greedy stuck at local optimum).
  D. act-side grid re-rank: greedy per-(row,64-block) sf/lv2/lv3 re-selection
     ranked by the EXACT Gram objective J(v)=tr(v gw v^T)-2tr(v gwf^T x^T)
     (the carried bf16 Grams, i.e. the deployable objective), interleaved
     with flip refinement.  6-cand grid (ship act grid) and 16-cand grid
     (upper bound).  Report J reduction + output-space score delta.
  E. weight-side grid re-anchor (task 3iii upper bound): exhaustive per-block
     sf/lv2/lv3 re-search against the exact CALIBRATION output objective
     (Gxx from <=2048 fit rows, hold-out = last calib sample, E3 convention),
     started from the ship q_used (post E3/GPTQ), 2 passes, plain-rounded
     candidate values.  Re-score all 5 test cases with re-anchored weights
     (activations unchanged) -> direct per-case score delta.

Usage: python dev/decomp2/anatomy.py run [--C 512,1024,2048] [--limit k]
       python dev/decomp2/anatomy.py rep
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
sys.path.insert(0, DEV)
sys.path.insert(0, HERE)
import hif4 as H          # noqa: E402
import variants as V      # noqa: E402
import study2 as S2       # noqa: E402

RES_AN = os.path.join(HERE, "results_anatomy.json")
SOL = S2.sol()


# ---------------------------------------------------------------------------
# replication of the ship dynamic path (refined branch)
# ---------------------------------------------------------------------------
def dynamic_internals(pair, st, mod=SOL):
    x = mod.dequantize_nvfp4(pair[0], pair[1]).float()
    R, C = x.shape
    s = st.get("s")
    if isinstance(s, torch.Tensor) and s.numel() == C:
        x = x * s.float()
    if st.get("mode") == 1:
        x = mod._rot_blocks(x)
    p = mod._quantize_weighted(x, torch.ones(1, C, dtype=torch.float32))
    unit = mod._params_unit_flat(p)
    values = None
    if st.get("g") == 1:
        u = st.get("u_act")
        order = st.get("order")
        if isinstance(order, torch.Tensor) and order.numel() == C:
            ol = order.long()
            q = mod._gptq_quantize_values(x[:, ol], unit[:, ol], u.float())
            q0 = torch.empty_like(q)
            q0[:, ol] = q
            values = q0
        else:
            values = mod._gptq_quantize_values(x, unit, u.float())
    v0 = values if values is not None else mod._deq_params(p)
    gw = st["gw"].float()
    gwf = st["gwf"].float()
    v1 = mod._refine_act_values(x, v0, unit, gw, gwf)
    return {"x": x, "p": p, "unit": unit, "v0": v0, "v1": v1, "gw": gw, "gwf": gwf}


def act_J(v, x, gw, gwf):
    return (((v @ gw) * v).sum() - 2.0 * ((x @ gwf) * v).sum()).item()


def flip_min_gains(v, unit, x, gw, gwf, T):
    v4 = torch.round(v / unit * 4.0)
    d = 0.25 * unit
    col2 = gw.diagonal()
    M = (v4 * d) @ gw - x @ gwf
    g, _ = SOL._flip_sel(d, M, col2, v4)
    return g.view(T, -1).min(dim=1).values


# ---------------------------------------------------------------------------
# candidate grid enumeration for one 64-block, many rows (batched over the
# candidate axis; same lv-tree tie semantics as solution._quant_chunk)
# ---------------------------------------------------------------------------
def block_values_batched(blk, cands):
    """blk: (M, 64) signed target values.  Returns V (K, M, 64) plain
    rounding of blk on each candidate grid, sfV (K, M), l2V (K, M, 8),
    l3V (K, M, 8, 2).  K = 5 * len(cands): per sf candidate the lv tree
    plus the 4 uniform lv combos."""
    M = blk.shape[0]
    ab = blk.abs()
    sgn = torch.sign(blk)
    ab4 = ab.reshape(1, M, 1, 8, 2, 4)
    amax = ab.amax(dim=1)
    e0 = torch.floor(torch.log2((amax / 7.0).clamp_min(1e-38)))
    offs = torch.tensor([float(k) for k, _ in cands]).reshape(-1, 1)
    sigs = torch.tensor([float(s) for _, s in cands]).reshape(-1, 1)
    sfV = (torch.exp2(e0.unsqueeze(0) + offs) * sigs).clamp(SOL.SF_MIN, SOL.SF_MAX)
    K = sfV.shape[0]
    sf5 = sfV[:, :, None, None, None, None]             # (K, M, 1,1,1,1)

    def rnd(unit):
        mant = torch.clamp(torch.round(ab4 / unit * 4.0) / 4.0, 0.0, 1.75)
        return mant * unit * sgn.reshape(1, M, 1, 8, 2, 4)

    # lv tree (solution order: lv3=1.0 / lv2=1.0 win ties)
    best_e2 = None
    best_l2 = None
    best_l3 = None
    for lv2_c in (1.0, 2.0):
        e3_list = []
        for lv3_c in (1.0, 2.0):
            e3_list.append(((rnd(sf5 * lv2_c * lv3_c) - ab4) ** 2).sum(dim=5))
        take1 = e3_list[0] <= e3_list[1]
        e3 = torch.where(take1, e3_list[0], e3_list[1])
        lv3 = torch.where(take1, 1.0, 2.0)
        e2 = e3.sum(dim=4)
        if best_e2 is None:
            best_e2 = e2
            best_l2 = torch.full_like(e2, lv2_c)
            best_l3 = lv3
        else:
            take2 = e2 < best_e2
            best_e2 = torch.where(take2, e2, best_e2)
            best_l2 = torch.where(take2, torch.full_like(e2, lv2_c), best_l2)
            best_l3 = torch.where(take2.unsqueeze(-1), lv3, best_l3)
    # best_*: (K, M, 1, 8[, 2])
    unit_tree = (sf5 * best_l2.reshape(K, M, 1, 8, 1, 1)
                 * best_l3.reshape(K, M, 1, 8, 2, 1))
    V_tree = rnd(unit_tree).reshape(K, M, 64)
    l2V_t = best_l2.reshape(K, M, 8)
    l3V_t = best_l3.reshape(K, M, 8, 2)

    V_list = [V_tree]
    sf_list = [sfV]
    l2_list = [l2V_t]
    l3_list = [l3V_t]
    for lv2_c in (1.0, 2.0):
        for lv3_c in (1.0, 2.0):
            V_list.append(rnd(sf5 * float(lv2_c * lv3_c)).reshape(K, M, 64))
            sf_list.append(sfV)
            l2_list.append(torch.full((K, M, 8), lv2_c))
            l3_list.append(torch.full((K, M, 8, 2), lv3_c))
    return (torch.cat(V_list, 0), torch.cat(sf_list, 0),
            torch.cat(l2_list, 0), torch.cat(l3_list, 0))


# ---------------------------------------------------------------------------
# D. act-side grid re-rank (deployable objective: bf16 Grams)
# ---------------------------------------------------------------------------
def act_rerank(x, v_in, unit_in, p_ship, st, cands, passes=3, mod=SOL,
               reflip=True):
    """Greedy per-(row, block) grid re-selection ranked by exact J, then flip
    refinement on the (mixed) grids; iterate.  With reflip=True the
    acceptance test is: apply the per-row best candidates, re-run the flip
    refinement, then keep per ROW only what improved J_t (rows are
    independent in J and in the greedy flips, so per-row accept/revert is
    exact).  Plain mode accepts a block move only if the plain-rounded
    candidate already improves J.  v/unit stay format-legal."""
    gw = st["gw"].float()
    gwf = st["gwf"].float()
    T, C = x.shape
    nb = C // 64
    v = v_in.clone()
    unit = unit_in.clone()
    sf_sel = p_ship["scale_factor"].reshape(T, nb).clone()
    lv2_sel = p_ship["scale_lv2"].reshape(T, nb, 8).clone()
    lv3_sel = p_ship["scale_lv3"].reshape(T, nb, 8, 2).clone()
    Jt = ((v @ gw) * v).sum(dim=1) - 2.0 * ((x @ gwf) * v).sum(dim=1)  # (T,)
    J0 = float(Jt.sum())
    hist = [J0]
    rows = torch.arange(T)
    B = x @ gwf                      # constant linear term of the objective
    moved = -1
    for it in range(passes):
        M2 = v @ gw
        moved = 0
        for b in range(nb):
            sl = slice(b * 64, (b + 1) * 64)
            V, sfV, l2V, l3V = block_values_batched(x[:, sl], cands)
            K = V.shape[0] + 1
            d = V - v[:, sl].unsqueeze(0)                  # (K-1, T, 64)
            dJ = torch.zeros(K, T)
            dJ[:K - 1] = 2.0 * torch.einsum('ktc,tc->kt', d, M2[:, sl]) \
                - 2.0 * torch.einsum('ktc,tc->kt', d, B[:, sl]) \
                + torch.einsum('ktc,dc,ktd->kt', d, gw[sl, sl], d)
            kstar = dJ.argmin(dim=0)
            acc_plain = dJ[kstar, rows] < -1e-9
            if not reflip:
                if not acc_plain.any():
                    continue
                moved += int(acc_plain.sum())
                idx = rows[acc_plain]
                ks = kstar[acc_plain].clamp_max(K - 2)
                v[:, sl] = v[:, sl].masked_scatter(
                    acc_plain.unsqueeze(1), V[ks, idx, :])
                unit[:, sl] = unit[:, sl].masked_scatter(
                    acc_plain.unsqueeze(1),
                    _unit_of(sfV[ks, idx], l2V[ks, idx], l3V[ks, idx]))
                sf_sel[:, b] = sf_sel[:, b].masked_scatter(acc_plain, sfV[ks, idx])
                lv2_sel[:, b] = lv2_sel[:, b].masked_scatter(
                    acc_plain.unsqueeze(1), l2V[ks, idx])
                lv3_sel[:, b] = lv3_sel[:, b].masked_scatter(
                    acc_plain.unsqueeze(1).unsqueeze(1), l3V[ks, idx])
                M2 = v @ gw
            else:
                # perturb every row to its per-row best candidate, re-flip,
                # keep per-row improvements only (exact: rows independent)
                v_try = v.clone()
                unit_try = unit.clone()
                sf_t = sf_sel.clone()
                lv2_t = lv2_sel.clone()
                lv3_t = lv3_sel.clone()
                ks = kstar.clamp_max(K - 2)
                v_try[:, sl] = V[ks, rows, :]
                unit_try[:, sl] = _unit_of(sfV[ks, rows], l2V[ks, rows],
                                           l3V[ks, rows])
                sf_t[:, b] = sfV[ks, rows]
                lv2_t[:, b] = l2V[ks, rows]
                lv3_t[:, b] = l3V[ks, rows]
                v_try = mod._refine_act_values(x, v_try, unit_try, gw, gwf)
                Jt_try = ((v_try @ gw) * v_try).sum(dim=1) \
                    - 2.0 * ((x @ gwf) * v_try).sum(dim=1)
                better = Jt_try < Jt - 1e-9
                if better.any():
                    moved += int(better.sum())
                    Jt = torch.where(better, Jt_try, Jt)
                    keep = better.unsqueeze(1)
                    v = torch.where(keep, v_try, v)
                    unit = torch.where(keep, unit_try, unit)
                    sf_sel = torch.where(keep, sf_t, sf_sel)
                    lv2_sel = torch.where(keep.unsqueeze(1), lv2_t, lv2_sel)
                    lv3_sel = torch.where(keep.unsqueeze(1).unsqueeze(1),
                                          lv3_t, lv3_sel)
                    M2 = v @ gw
        hist.append(act_J(v, x, gw, gwf))
        if not reflip:
            v = mod._refine_act_values(x, v, unit, gw, gwf)
            hist.append(act_J(v, x, gw, gwf))
        if moved == 0:
            break
    return v, unit, sf_sel, lv2_sel, lv3_sel, {"J_hist": hist, "moved_last": moved}


def _unit_of(sf, l2, l3):
    """Rebuild the per-element unit (n, 64) from selected candidate meta."""
    n = sf.shape[0]
    unit = (sf.reshape(n, 1, 1) * l2.reshape(n, 8, 1) * l3.reshape(n, 8, 2))
    return unit.reshape(n, 16, 1).expand(n, 16, 4).reshape(n, 64)


def act_params_from(v, sf_sel, lv2_sel, lv3_sel):
    T, C = v.shape
    nb = C // 64
    sf = sf_sel.reshape(T, nb, 1, 1, 1).float()
    lv2 = lv2_sel.reshape(T, nb, 8, 1, 1)
    lv3 = lv3_sel.reshape(T, nb, 8, 2, 1)
    unit = (sf * lv2 * lv3).expand(T, nb, 8, 2, 4).reshape(T, C)
    mant = (torch.round(v.abs() / unit * 4.0)).clamp_(0.0, 7.0) * 0.25
    p = {"scale_factor": sf.contiguous(), "scale_lv2": lv2.contiguous(),
         "scale_lv3": lv3.contiguous(),
         "sign": torch.sign(v).reshape(T, nb, 8, 2, 4).contiguous(),
         "mant": mant.reshape(T, nb, 8, 2, 4).contiguous()}
    return p, unit


# ---------------------------------------------------------------------------
# E. weight-side grid re-anchor (calibration objective, upper bound)
# ---------------------------------------------------------------------------
def build_gxx(calib_list, st, mod=SOL):
    acts_s = [mod.dequantize_nvfp4(aq, as_).float() * st["s"].float()
              for aq, as_ in calib_list]
    rows = 0
    Gxx = None
    for a in acts_s[:-1]:
        if rows >= mod.REFINE_W_ROWS:
            break
        at = mod._rot_blocks(a) if st.get("mode") == 1 else a
        at = at[: mod.REFINE_W_ROWS - rows]
        g = at.T @ at
        Gxx = g if Gxx is None else Gxx + g
        rows += at.shape[0]
    xh = acts_s[-1]
    if xh.shape[0] > mod.REFINE_W_HOLD_ROWS:
        stride = max(1, (xh.shape[0] + mod.REFINE_W_HOLD_ROWS - 1)
                     // mod.REFINE_W_HOLD_ROWS)
        xh = xh[::stride]
    return Gxx, xh.contiguous()


def weight_reanchor(w_final, q_used, wp_ship, Gxx, passes=2, chunk=2048):
    """Greedy per-block sf/lv2/lv3 re-selection ranked by the exact fit
    objective J_w = tr((q-w) Gxx (q-w)^T).  Plain-rounded candidate values on
    each grid; identity always available.  Vectorized in row chunks."""
    N, C = q_used.shape
    nb = C // 64
    q = q_used.clone()
    E = q - w_final
    hist = [((E @ Gxx) * E).sum().item()]
    sf_sel = wp_ship["scale_factor"].reshape(N, nb).clone()
    lv2_sel = wp_ship["scale_lv2"].reshape(N, nb, 8).clone()
    lv3_sel = wp_ship["scale_lv3"].reshape(N, nb, 8, 2).clone()
    moved = -1
    for it in range(passes):
        moved = 0
        for r0 in range(0, N, chunk):
            r2 = min(r0 + chunk, N)
            M2 = E[r0:r2] @ Gxx
            for b in range(nb):
                sl = slice(b * 64, (b + 1) * 64)
                V, sfV, l2V, l3V = block_values_batched(w_final[r0:r2, sl],
                                                        SOL.CAND_GRID_W)
                K = V.shape[0] + 1
                nc = r2 - r0
                d = V - q[r0:r2, sl].unsqueeze(0)          # (K-1, nc, 64)
                dJ = torch.zeros(K, nc)
                dJ[:K - 1] = 2.0 * torch.einsum('knc,nc->kn', d, M2[:, sl]) \
                    + torch.einsum('knc,dc,knd->kn', d, Gxx[sl, sl], d)
                kstar = dJ.argmin(dim=0)
                rows = torch.arange(nc)
                acc = dJ[kstar, rows] < -1e-9
                if acc.any():
                    moved += int(acc.sum())
                    idx = rows[acc]
                    ks = kstar[acc].clamp_max(K - 2)
                    q[r0:r2, sl] = q[r0:r2, sl].masked_scatter(
                        acc.unsqueeze(1), V[ks, idx, :])
                    sf_sel[r0:r2, b] = sf_sel[r0:r2, b].masked_scatter(
                        acc, sfV[ks, idx])
                    lv2_sel[r0:r2, b] = lv2_sel[r0:r2, b].masked_scatter(
                        acc.unsqueeze(1), l2V[ks, idx])
                    lv3_sel[r0:r2, b] = lv3_sel[r0:r2, b].masked_scatter(
                        acc.unsqueeze(1).unsqueeze(1), l3V[ks, idx])
                    dE = q[r0:r2, sl] - w_final[r0:r2, sl] - E[r0:r2, sl]
                    M2 += dE @ Gxx[sl, :]
                    E[r0:r2, sl] += dE
        E = q - w_final
        hist.append(((E @ Gxx) * E).sum().item())
        if moved == 0:
            break
    return q, sf_sel, lv2_sel, lv3_sel, {"J_hist": hist, "moved_last": moved}


def weight_params_from(q, sf_sel, lv2_sel, lv3_sel):
    N, C = q.shape
    nb = C // 64
    sf = sf_sel.reshape(N, nb, 1, 1, 1).float()
    lv2 = lv2_sel.reshape(N, nb, 8, 1, 1)
    lv3 = lv3_sel.reshape(N, nb, 8, 2, 1)
    unit = (sf * lv2 * lv3).expand(N, nb, 8, 2, 4).reshape(N, C)
    mant = (torch.round(q.abs() / unit * 4.0)).clamp_(0.0, 7.0) * 0.25
    p = {"scale_factor": sf.contiguous(), "scale_lv2": lv2.contiguous(),
         "scale_lv3": lv3.contiguous(),
         "sign": torch.sign(q).reshape(N, nb, 8, 2, 4).contiguous(),
         "mant": mant.reshape(N, nb, 8, 2, 4).contiguous()}
    return p


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def anatomy_group(name):
    group = S2.build_group(name)
    w_ref = H.dequantize_nvfp4(*group["weight"])
    w_std = V.deq(V.quant_alg1(w_ref.float()))
    cc = torch.load(os.path.join(S2.CACHE, f"{name}_ship.pt"),
                    weights_only=True)
    cal = cc["cal"]
    st = cal["activation_state"]
    wp = cal["weight_params"]
    w_play = H.hif4_dequantize(wp)
    out = {"C": next(g[2] for g in S2.iter_grid() if g[0] == name),
           "N": int(group["weight"][0].shape[0])}

    cases = []
    for pair in group["test_activation_list"]:
        x_ref = H.dequantize_nvfp4(*pair)
        ref = H.linear_ref(x_ref, w_ref)
        x_std = V.deq(V.quant_alg1(x_ref.float()))
        mse_std = ((H.linear_ref(x_std, w_std) - ref) ** 2).mean().item()
        p = SOL.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
        x_play = H.hif4_dequantize(p)
        mse_play = ((H.linear_ref(x_play, w_play) - ref) ** 2).mean().item()
        mse_act = ((H.linear_ref(x_play, w_ref) - ref) ** 2).mean().item()
        mse_w = ((H.linear_ref(x_ref, w_play) - ref) ** 2).mean().item()
        cases.append({"T": int(pair[0].shape[0]), "mse_std": mse_std,
                      "mse_play": mse_play, "mse_act": mse_act, "mse_w": mse_w,
                      "score": (mse_std - mse_play) / mse_std,
                      "_x_play": x_play, "_ref": ref})
    out["cases"] = [{k: v for k, v in c.items() if not k.startswith("_")}
                    for c in cases]

    # ---- T=10 anatomy ----
    t10 = cases[0]
    assert t10["T"] == 10
    pair = group["test_activation_list"][0]
    p_ship = SOL.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
    xq_ship = H.hif4_dequantize(p_ship)
    ints = dynamic_internals(pair, st)
    v1 = ints["v1"]
    xq_repl = H.hif4_dequantize(
        SOL._values_to_params(v1.contiguous(), ints["p"]))
    out["replication_ok"] = bool(torch.equal(xq_repl, xq_ship))

    x, unit, gw, gwf = ints["x"], ints["unit"], ints["gw"], ints["gwf"]
    J_pre = act_J(v1, x, gw, gwf)
    per_row_min = flip_min_gains(v1, unit, x, gw, gwf, 10)
    conv = {"J_pre": J_pre,
            "rows_no_improving_flip": int((per_row_min >= 0).sum()),
            "min_gain": float(per_row_min.min())}
    m96 = S2.load_patched(sweeps=96)
    p96 = m96.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
    xq96 = H.hif4_dequantize(p96)
    conv["s96_bit_identical"] = bool(torch.equal(xq96, xq_ship))
    mse96 = ((H.linear_ref(xq96, w_play) - t10["_ref"]) ** 2).mean().item()
    conv["s96_dmsT10_pp"] = (t10["mse_play"] - mse96) / t10["mse_std"] * 100
    out["convergence"] = conv

    # act-side re-rank (deployable Gram objective)
    rer = {}
    for tag, cands in (("g6", SOL.CAND_GRID), ("g16", SOL.CAND_GRID_W)):
        t0 = time.perf_counter()
        v_r, unit_r, sf_s, lv2_s, lv3_s, info = act_rerank(
            x, v1, unit, ints["p"], st, cands, passes=3)
        p_r, _ = act_params_from(v_r, sf_s, lv2_s, lv3_s)
        xq_r = H.hif4_dequantize(p_r)
        mse_r = ((H.linear_ref(xq_r, w_play) - t10["_ref"]) ** 2).mean().item()
        rer[tag] = {"J_hist": info["J_hist"],
                    "dt": time.perf_counter() - t0,
                    "dmsT10_pp": (t10["mse_play"] - mse_r) / t10["mse_std"] * 100}
    out["act_rerank"] = rer

    # ---- weight-side re-anchor ----
    w = SOL.dequantize_nvfp4(*group["weight"]).float()
    w_final = w / st["s"].float()
    if st.get("mode") == 1:
        w_final = SOL._rot_blocks(w_final)
    q_used = w_play.clone()
    Gxx, xh = build_gxx(group["calib_activation_list"], st)
    ref_h = xh @ w_final.T
    hold0 = ((xh @ q_used.T - ref_h) ** 2).mean().item()
    t0 = time.perf_counter()
    q_new, sf_s, lv2_s, lv3_s, winfo = weight_reanchor(w_final, q_used, wp,
                                                       Gxx, passes=2)
    wp_new = weight_params_from(q_new, sf_s, lv2_s, lv3_s)
    w_play_new = H.hif4_dequantize(wp_new)
    hold1 = ((xh @ w_play_new.T - ref_h) ** 2).mean().item()
    wout = {"J_hist": winfo["J_hist"], "hold0": hold0, "hold1": hold1,
            "dt": time.perf_counter() - t0,
            "J_drop_frac": 1.0 - winfo["J_hist"][-1] / max(winfo["J_hist"][0], 1e-30),
            "roundtrip_rel": float((w_play_new - q_new).abs().max()
                                   / q_new.abs().mean().clamp_min(1e-30)),
            "cases": []}
    for c in cases:
        mse_new = ((H.linear_ref(c["_x_play"], w_play_new)
                    - c["_ref"]) ** 2).mean().item()
        wout["cases"].append({"T": c["T"],
                              "d_pp": (c["mse_play"] - mse_new) / c["mse_std"] * 100})
    out["weight_reanchor"] = wout
    return out


def jload_(p):
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def jsave_(p, obj):
    with open(p, "w") as f:
        json.dump(obj, f, indent=1)


def run(c_filter, limit):
    res = jload_(RES_AN)
    grid = [g for g in S2.iter_grid(c_filter)
            if os.path.exists(os.path.join(S2.CACHE, f"{g[0]}_ship.pt"))]
    grid = [g for g in grid if g[2] <= 2048]      # always-refined populations
    if limit:
        grid = grid[:limit]
    print(f"[anat] {len(grid)} groups")
    for name, seed, C, N, spread, outp in grid:
        if name in res:
            print(f"[anat] {name}: cached, skip")
            continue
        t0 = time.perf_counter()
        res[name] = anatomy_group(name)
        jsave_(RES_AN, res)
        r = res[name]
        print(f"[anat] {name}: {time.perf_counter()-t0:.1f}s "
              f"repl={r['replication_ok']} "
              f"conv={r['convergence']['rows_no_improving_flip']}/10 "
              f"s96={r['convergence']['s96_bit_identical']} "
              f"actR g6 {r['act_rerank']['g6']['dmsT10_pp']:+.2f} "
              f"g16 {r['act_rerank']['g16']['dmsT10_pp']:+.2f}pp "
              f"wR Jdrop {r['weight_reanchor']['J_drop_frac']*100:.1f}% "
              f"hold {r['weight_reanchor']['hold0']:.3e}"
              f"->{r['weight_reanchor']['hold1']:.3e}")
        sys.stdout.flush()
    print("[anat] complete")


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def rep():
    res = jload_(RES_AN)
    names = sorted(res.keys())
    print(f"groups: {len(names)}")
    print("\n=== T=10 residual shares (of mse_play) ===")
    aw, ww, cw = [], [], []
    for n in names:
        c = res[n]["cases"][0]
        aw.append(c["mse_act"] / c["mse_play"])
        ww.append(c["mse_w"] / c["mse_play"])
        cw.append((c["mse_play"] - c["mse_act"] - c["mse_w"]) / c["mse_play"])
    print(f"act share mean {_mean(aw):.3f} | w share mean {_mean(ww):.3f} "
          f"| cross share mean {_mean(cw):.3f}")
    print("\n=== convergence (T=10, after 24 sweeps) ===")
    print(f"replication ok: {sum(res[n]['replication_ok'] for n in names)}/{len(names)}")
    print(f"rows with no improving flip: "
          f"{sum(res[n]['convergence']['rows_no_improving_flip'] for n in names)}"
          f"/{10 * len(names)}")
    print(f"s96 bit-identical: "
          f"{sum(res[n]['convergence']['s96_bit_identical'] for n in names)}"
          f"/{len(names)}; s96 residual dmsT10 pp mean "
          f"{_mean([res[n]['convergence']['s96_dmsT10_pp'] for n in names]):+.3f}")
    print("\n=== act grid re-rank ===")
    for tag in ("g6", "g16"):
        J0 = [res[n]["act_rerank"][tag]["J_hist"][0] for n in names]
        J1 = [res[n]["act_rerank"][tag]["J_hist"][-1] for n in names]
        dp = [res[n]["act_rerank"][tag]["dmsT10_pp"] for n in names]
        dt = [res[n]["act_rerank"][tag]["dt"] for n in names]
        print(f"{tag}: J mean {sum(J0)/len(J0):.3e} -> {sum(J1)/len(J1):.3e} "
              f"| dmsT10 {_mean(dp):+.2f}pp (min {min(dp):+.2f}) | dt {_mean(dt):.2f}s")
    print("\n=== weight grid re-anchor ===")
    jd = [res[n]["weight_reanchor"]["J_drop_frac"] for n in names]
    ho = [res[n]["weight_reanchor"]["hold1"] / max(res[n]["weight_reanchor"]["hold0"], 1e-30)
          for n in names]
    print(f"J drop mean {_mean(jd) * 100:.1f}% | hold1/hold0 mean {_mean(ho):.3f} "
          f"(n better {sum(h < 1 for h in ho)}) | dt "
          f"{_mean([res[n]['weight_reanchor']['dt'] for n in names]):.1f}s")
    for T in (10, 128, 512, 1024):
        d = [c["d_pp"] for n in names for c in res[n]["weight_reanchor"]["cases"]
             if c["T"] == T]
        print(f"  T={T}: d_pp mean {_mean(d):+.2f} (n={len(d)})")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "rep"
    c_filter = None
    limit = None
    args = sys.argv[2:]
    for i, a in enumerate(args):
        if a == "--C":
            c_filter = set(int(x) for x in args[i + 1].split(","))
        elif a == "--limit":
            limit = int(args[i + 1])
    if mode == "run":
        run(c_filter, limit)
    else:
        rep()


if __name__ == "__main__":
    main()
