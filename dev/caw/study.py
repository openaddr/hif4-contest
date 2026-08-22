"""CAW: CANCELLATION-AWARE weight quantization for the lattice-refinement
pipeline.  solution.py is NEVER modified; a patched module is exec'd from its
source with a surgical wrap of the weight-GPTQ stage.

TASK 1 - algebra (pipeline notation)
------------------------------------
Per dynamic call (Linear, transformed space: smoothed and, if mode==1,
rotated).  x: (T,C) exact input; v: refined activation values on the lattice;
w = w_final, q = q_used: (N,C).  Output error

  D = v q^T - x w^T = (v - x) q^T + x (q - w)^T = Dq*q^T + E_w,

  Dq := v - x   (act-side correction),   E_w := x (q - w)^T.

`_refine_act_values` moves v on the flip lattice: v4 in [-7,7] integer, step
0.25*unit per element, so per element (r,c)

  Dq_rc = (v4_rc - v4_init,rc) * d_rc,  d = 0.25*unit,
  continuous relaxation:  Dq_rc in [-A_rc, +A_rc],
  A_rc = min(k, 7 - |v4_rtn,rc|) * d_rc      (k ~ 1-2 effective quanta).

The refinement greedily minimizes the EXACT objective
  J(v) = ||v q^T - x w^T||_F^2 = sum_r [v_r Gw v_r^T - 2 v_r Gwf x_r^T] + const,
Gw = q^T q, Gwf = w^T q (the carried Grams).  First order it cancels the
component of E_w reachable by lattice moves, i.e. the projection of E_w onto
the zonotope image Z = {Dq q^T : |Dq_rc| <= A_rc}.  The weight quantizer
should therefore target the UNCANCELABLE residual

  J_u(q) = min_{|Dq| <= A} || Dq q^T + x (q-w)^T ||_F^2          (rows indep.)

(a) diag approx: Gram-diagonalize Q^T Q ~= diag(g_c), g_c = ||q_c||^2.
    Row r's error along column direction q_c is <e_r, q_c> = (X (Q-W)^T Q)_{rc};
    element (r,c) nulls it up to A_rc*g_c (in <.,q_c> units):
      J_u^diag = sum_{r,c} max(0, |(X (Q-W)^T Q)_{rc}| - A_rc g_c)^2 / g_c.
    Wired into GPTQ (the shipped variant): at column i the error-feedback
    target e = w_i - q_i is replaced by the uncancelable-projected error
      e_u = e - beta_i alpha_i q_i,  alpha_i = <e, q_i>/g_i,
      beta_i = sum_r min(|x_ri alpha_i|, A_ri)^2 / sum_r (x_ri alpha_i)^2
             = cancelable fraction of the q_i-aligned output-error energy,
    everything else in GPTQ (grid, blocks, Hessian U, act side, E3, Grams)
    identical.
(b) box-relaxed: solve the continuous-box projection with FISTA projected
    gradient (offline / guard-side only; too slow inside the column loop).
(c) empirical oracle: run the ACTUAL `_refine_act_values` on held rows.

Guard (task 4, E3 convention): fit rows from calib[:-1], guard on calib[-1]
(xh_pick, 128-row subsample); acceptance = post-refinement holdout MSE
(the shipped refinement, run on <= guard_rows of the holdout rows) must beat
the ship choice (RTN vs GPTQ by the ship's own plain-holdout rule).

Usage:
  python dev/caw/study.py selftest
  python dev/caw/study.py fit
  python dev/caw/study.py suite [--k 1.0]
  python dev/caw/study.py mini  [--k 1.0]
  python dev/caw/study.py rep
"""
from __future__ import annotations

import json
import os
import sys
import time
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.dirname(HERE)
ROOT = os.path.dirname(DEV)
sys.path.insert(0, DEV)
import hif4 as H          # noqa: E402
import synth              # noqa: E402
import variants as V      # noqa: E402

CACHE = os.path.join(HERE, "cache")
RES_FIT = os.path.join(HERE, "results_fit.json")
RES_SUITE = os.path.join(HERE, "results_suite.json")
RES_MINI = os.path.join(HERE, "results_mini.json")
SOL_PATH = os.path.join(ROOT, "example", "solution", "solution.py")

CALIB_T = (10, 128, 512, 1024)
TEST_T = (10, 128, 512, 1024, 1024)
CS_FULL = (512, 1024, 2048, 4096, 8192)     # study2 enumeration order
CS_SUITE = (1024, 2048, 4096)
NS = (1024, 8192)
SPREADS = (0.5, 0.9)
OUTLIERS = (0.0, 0.002)

FIT_GROUPS = ((1024, 1024, 0.5, 0.0), (1024, 8192, 0.9, 0.002),
              (2048, 8192, 0.9, 0.0), (2048, 1024, 0.5, 0.002))

# ---------------------------------------------------------------------------
# patched-module construction (solution source + surgical wrap, study2 style)
# ---------------------------------------------------------------------------
_QG_LINE = "        q_g = _gptq_quantize_values(w_final, unit, Uw)\n"
_ACC_OLD = (
    "        mse_r = ((xh_pick @ q_used.T - ref) ** 2).mean().item()\n"
    "        mse_g = ((xh_pick @ q_g.T - ref) ** 2).mean().item()\n"
    "        if mse_g < mse_r:\n"
    "            weight_params = _values_to_params(q_g, weight_params)\n"
    "            q_used = q_g.contiguous()\n")
_ACC_NEW = _ACC_OLD + (
    "        if _caw_cand is not None:\n"
    "            _pick = _caw_guard_pick(xh_pick, q_used, _caw_cand, w_final)\n"
    "            if _pick is not None:\n"
    "                weight_params = _values_to_params(_pick, weight_params)\n"
    "                q_used = _pick.contiguous()\n")

_APPEND = '''

# === dev/caw: cancellation-aware weight GPTQ (prototype; not shipped) ===
import time as _caw_time

_CAW_CFG = {"on": False, "k": 1.0, "ks": (), "fit_rows": 650,
            "guard": True, "guard_rows": 24}
_CAW_LAST = {}


def _caw_grid(X):
    """Per-row flip lattice geometry: d = 0.25*unit and |v4| of the RTN point
    (transformed space; unit exactly as the dynamic path computes it)."""
    p = _quantize_weighted(X, torch.ones(1, X.shape[1], dtype=torch.float32))
    unit = _params_unit_flat(p)
    v4 = (torch.round(X.abs() / unit * 4.0)).clamp_(0.0, 7.0)
    return (0.25 * unit), v4


def _caw_col_caps(dv, sgn_a, k):
    """Directional per-column caps (numpy, (R,) in): the lattice box is
    v4 in [-7,7]; the required correction at row r is delta* = x_r*alpha, so
    a saturated element (|v4|=7) is blocked only when sign(alpha) pushes it
    further from zero:  headroom_quanta = 7 - |v4|*sign(alpha)."""
    d_col, v4_col = dv
    h = np.maximum(7.0 - v4_col * sgn_a, 0.0)
    return np.minimum(float(k), h) * d_col


def _caw_gptq_np(w_final, unit, hinv, Xf, dgrid, k):
    """numpy twin of _gptq_quantize_values_np with cancellation-aware error
    feedback: at column i the feedback error e = w - q is replaced by
    e_u = e - beta*alpha*q (uncancelable-projected).  alpha = <e,q>/||q||^2
    is the q_i-aligned relative error; beta = cancelable fraction of the
    aligned output-error energy under the fit rows (diag approx, directional
    lattice box)."""
    R, C = w_final.shape
    W = w_final.clone()
    Q = torch.empty_like(W)
    unp = (unit if unit.is_contiguous() else unit.contiguous()).numpy()
    hnp = hinv.contiguous().numpy()
    Xn = Xf.numpy() if torch.is_tensor(Xf) else Xf
    dn_ = dgrid[0].contiguous().numpy()
    vn_ = dgrid[1].contiguous().numpy()
    npr_, npa_, npw_, npc_ = np.round, np.abs, np.where, np.clip
    one, mone = np.float32(1.0), np.float32(-1.0)
    betas = np.zeros(C, dtype=np.float32)
    for i1 in range(0, C, GPTQ_BLOCK):
        i2 = min(i1 + GPTQ_BLOCK, C)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        w1, q1, e1 = W1.numpy(), Q1.numpy(), E1.numpy()
        Hi = hnp[i1:i2, i1:i2]
        u = unp[:, i1:i2]
        last = i2 - i1 - 1
        for i in range(i2 - i1):
            g = i1 + i
            w = w1[:, i]
            ui = u[:, i]
            m = npr_(npa_(w) / ui * 4.0)
            npc_(m, 0.0, 7.0, out=m)
            m *= 0.25
            s = npw_(w >= 0, one, mone)
            q = s * m * ui
            q1[:, i] = q
            d = Hi[i, i]
            if d < 1e-30:
                d = np.float32(1e-30)
            e = w - q
            gg = float(q @ q)
            beta = 0.0
            if gg > 1e-30:
                al = float(e @ q) / gg
                if al != 0.0:
                    xa = Xn[:, g] * al
                    den = float(xa @ xa)
                    if den > 1e-30:
                        ac = _caw_col_caps((dn_[:, g], vn_[:, g]),
                                           1.0 if al > 0.0 else -1.0, k)
                        num = np.minimum(npa_(xa), ac)
                        beta = min(1.0, float(num @ num) / den)
                        e = e - np.float32(beta * al) * q
            betas[g] = beta
            e1[:, i] = e / d
            if i < last:
                w1[:, i + 1:] -= e1[:, i][:, None] * Hi[i, i + 1:]
        Q[:, i1:i2] = Q1
        if i2 < C:
            W[:, i2:] -= E1 @ hinv[i1:i2, i2:]
            W[:, i1:i2] = W1
    return Q, betas


def _caw_ref_mse(xh, w_final, q):
    """Post-refinement holdout MSE of weights q on rows xh (fp32 Grams)."""
    p = _quantize_weighted(xh, torch.ones(1, xh.shape[1], dtype=torch.float32))
    unit = _params_unit_flat(p)
    v0 = _deq_params(p)
    gw = q.T @ q
    gwf = w_final.T @ q
    v1 = _refine_act_values(xh, v0, unit, gw, gwf)
    return ((v1 @ q.T - xh @ w_final.T) ** 2).mean().item()


def _caw_guard_pick(xh_pick, q_ship, q_caw, w_final):
    """E3-convention guard: accept q_caw only if the POST-REFINEMENT holdout
    MSE beats the ship choice.  Returns q_caw or None.  _CAW_CFG["force"]
    bypasses the guard (diagnostic: what unguarded caw would ship)."""
    if _CAW_CFG.get("force"):
        _CAW_LAST["guard"] = {"forced": True}
        return q_caw
    if not _CAW_CFG.get("guard", True):
        return None
    n = xh_pick.shape[0]
    gr = int(_CAW_CFG.get("guard_rows", 24))
    stride = max(1, (n + gr - 1) // gr)
    xh = xh_pick[::stride].contiguous()
    t0 = _caw_time.perf_counter()
    m_s = _caw_ref_mse(xh, w_final, q_ship)
    m_c = _caw_ref_mse(xh, w_final, q_caw)
    _CAW_LAST["guard"] = {"m_ship": m_s, "m_caw": m_c,
                          "rows": int(xh.shape[0]),
                          "dt": _caw_time.perf_counter() - t0}
    return q_caw if m_c < m_s else None


def _caw_stage(w_final, unit, Uw, acts_s, tf_final, weight_params,
               xh_pick, q_rtn):
    """Wraps the ship weight GPTQ call.  Returns (q_g, cand_or_None, info);
    with the stage off it is a pure pass-through (bit-identical ship path)."""
    q_g = _gptq_quantize_values(w_final, unit, Uw)
    _CAW_LAST.clear()
    _CAW_LAST.update({"on": False, "C": int(w_final.shape[1])})
    if (not _CAW_CFG.get("on") or xh_pick is None
            or w_final.shape[1] > 4096 or len(acts_s) < 2):
        return q_g, None, {}
    t0 = _caw_time.perf_counter()
    cap = int(_CAW_CFG.get("fit_rows", 650))
    rows = []
    nrow = 0
    for a in acts_s[:-1]:
        if nrow >= cap:
            break
        at = tf_final(a)
        at = at[: cap - nrow]
        rows.append(at)
        nrow += at.shape[0]
    Xf = torch.cat(rows).contiguous()
    ks = tuple(_CAW_CFG.get("ks") or (_CAW_CFG.get("k", 1.0),))
    kmain = float(_CAW_CFG.get("k", 1.0))
    if kmain not in ks:
        ks = ks + (kmain,)
    cands = {}
    gptq_s = {}
    beta_mean = {}
    dgrid = None
    for k in ks:
        if dgrid is None:
            dgrid = _caw_grid(Xf)
        t1 = _caw_time.perf_counter()
        qc, betas = _caw_gptq_np(w_final, unit, Uw, Xf, dgrid, k)
        gptq_s[k] = _caw_time.perf_counter() - t1
        cands[k] = qc
        beta_mean[k] = float(betas.mean())
    info = {"fit_rows": int(Xf.shape[0]), "ks": [float(k) for k in ks],
            "gptq_s": {float(k): v for k, v in gptq_s.items()},
            "beta_mean": beta_mean,
            "dt_stage": _caw_time.perf_counter() - t0}
    cand = cands[kmain] if _CAW_CFG.get("guard", True) else None
    _CAW_LAST.update({"on": True, "info": info, "Xf": Xf, "xh_pick": xh_pick,
                      "w_final": w_final, "q_rtn": q_rtn, "q_g": q_g,
                      "cands": dict(cands), "unit": unit, "Uw": Uw})
    return q_g, cand, info
'''


def load_sol(cfg=None):
    """cfg None -> unpatched ship module; else patched with _CAW_CFG=cfg."""
    with open(SOL_PATH, encoding="utf-8") as f:
        src = f.read()
    if cfg is not None:
        if src.count(_QG_LINE) != 1:
            raise RuntimeError("q_g patch target not unique")
        new_qg = ("        q_g, _caw_cand, _caw_info = _caw_stage(\n"
                  "            w_final, unit, Uw, acts_s, tf_final,\n"
                  "            weight_params, xh_pick, q_used)\n")
        src = src.replace(_QG_LINE, new_qg)
        if src.count(_ACC_OLD) != 1:
            raise RuntimeError("acceptance patch target not unique")
        src = src.replace(_ACC_OLD, _ACC_NEW)
        cfg_txt = ", ".join(f"{k!r}: {v!r}" for k, v in cfg.items())
        src += _APPEND.replace(
            '_CAW_CFG = {"on": False, "k": 1.0, "ks": (), "fit_rows": 650,\n'
            '            "guard": True, "guard_rows": 24}',
            f"_CAW_CFG = {{{cfg_txt}}}")
    mod = types.ModuleType("_caw_sol")
    mod.__file__ = SOL_PATH
    exec(compile(src, SOL_PATH, "exec"), mod.__dict__)
    return mod


# ---------------------------------------------------------------------------
# group construction (seeds identical to study2.iter_grid enumeration)
# ---------------------------------------------------------------------------
def iter_grid():
    out = []
    i = 0
    for C in CS_FULL:
        for N in NS:
            for spread in SPREADS:
                for outp in OUTLIERS:
                    out.append((f"c{C}_n{N}_s{spread}_o{outp}",
                                4200 + 13 * i, C, N, spread, outp))
                    i += 1
    return out


def suite_grid():
    """CS_SUITE x spreads x outliers (12 combos); N alternates 1024/8192."""
    out = []
    full = {g[0]: g for g in iter_grid()}
    j = 0
    for C in CS_SUITE:
        for spread in SPREADS:
            for outp in OUTLIERS:
                N = (1024, 8192)[j % 2]
                name = f"c{C}_n{N}_s{spread}_o{outp}"
                out.append(full[name])
                j += 1
    return out


def make_group(seed, C, N, spread, outlier_p):
    g = synth.make_linear_group(seed, N, C, tokens=CALIB_T + TEST_T,
                                spread=spread, outlier_p=outlier_p)
    nc = len(CALIB_T)
    return {"weight": g["weight"],
            "calib_activation_list": g["calib_activation_list"][:nc],
            "test_activation_list": g["test_activation_list"][nc:]}


def calibrate(mod, group, tag, name, use_cache=True):
    cpath = os.path.join(CACHE, f"{name}_{tag}.pt")
    if use_cache and os.path.exists(cpath):
        return torch.load(cpath, weights_only=True)
    torch.manual_seed(0)
    t0 = time.perf_counter()
    cal = mod.hif4_calibration_and_quantize_weight(
        group["weight"][0], group["weight"][1],
        group["calib_activation_list"])
    cal_s = time.perf_counter() - t0
    out = {"cal": cal, "cal_s": cal_s}
    if use_cache:
        torch.save(out, cpath)
    return out


def score_case(mod, pair, w_ref, w_std, weight_params, st):
    x_ref = H.dequantize_nvfp4(*pair)
    ref = H.linear_ref(x_ref, w_ref)
    x_std = V.deq(V.quant_alg1(x_ref.float()))
    mse_std = ((H.linear_ref(x_std, w_std) - ref) ** 2).mean().item()
    t0 = time.perf_counter()
    p = mod.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
    dt = time.perf_counter() - t0
    x_play = H.hif4_dequantize(p)
    w_play = H.hif4_dequantize(weight_params)
    mse_play = ((H.linear_ref(x_play, w_play) - ref) ** 2).mean().item()
    mse_act = ((H.linear_ref(x_play, w_ref) - ref) ** 2).mean().item()
    mse_w = ((H.linear_ref(x_ref, w_play) - ref) ** 2).mean().item()
    return {"T": int(pair[0].shape[0]), "dt": dt, "mse_std": mse_std,
            "mse_play": mse_play, "mse_act": mse_act, "mse_w": mse_w,
            "score": (mse_std - mse_play) / mse_std}


# ---------------------------------------------------------------------------
# fidelity predictors (torch, offline)
# ---------------------------------------------------------------------------
def caw_A(mod, X, W, Q, k):
    """Directional per-element caps A for eval rows X under candidate Q."""
    d, v4 = mod._caw_grid(X)
    g = Q.pow(2.0).sum(dim=0).clamp_min(1e-30)
    al = ((W - Q).T @ Q).diagonal() / g
    sgn = torch.where(al >= 0, 1.0, -1.0)
    h = (7.0 - v4 * sgn.unsqueeze(0)).clamp_min(0.0)
    return torch.minimum(torch.full_like(h, float(k)), h) * d
def ju_diag(X, W, Q, A):
    """(a) diag-approx uncancelable energy / (R*N) -> mse units."""
    Ge = (Q - W).T @ Q
    g = Q.pow(2.0).sum(dim=0).clamp_min(1e-20)
    XGe = X @ Ge
    z = (XGe.abs() - A * g.unsqueeze(0)).clamp_min(0.0)
    return ((z * z) / g.unsqueeze(0)).sum().item() / (X.shape[0] * Q.shape[0])


def ju_box(X, W, Q, A, iters=120):
    """(b) box-relaxed projection via FISTA projected gradient -> mse."""
    Ge = (Q - W).T @ Q
    Gq = Q.T @ Q
    B = X @ Ge
    v = B.sum(dim=0)
    for _ in range(12):
        v = Gq @ v
        n = float(v.norm())
        if n <= 1e-30:
            break
        v = v / n
    L = float(v @ (Gq @ v))
    eta = 1.0 / max(L, 1e-30)
    DEL = torch.zeros_like(X)
    Y = DEL.clone()
    tk = 1.0
    for _ in range(iters):
        Dn = torch.clamp(Y - eta * (Y @ Gq + B), -A, A)
        tn = 0.5 * (1.0 + (1.0 + 4.0 * tk * tk) ** 0.5)
        Y = Dn + ((tk - 1.0) / tn) * (Dn - DEL)
        DEL, tk = Dn, tn
    D = DEL @ Q.T + X @ (Q - W).T
    return (D * D).mean().item()


# ---------------------------------------------------------------------------
# selftest: patched module with the stage OFF is bit-identical to ship
# ---------------------------------------------------------------------------
def run_selftest():
    ship = load_sol()
    patched = load_sol({"on": False, "k": 1.0})
    name, seed, C, N, spread, outp = suite_grid()[0]
    group = make_group(seed, C, N, spread, outp)
    c0 = calibrate(ship, group, "ship", name, use_cache=False)
    c1 = calibrate(patched, group, "shipoff", name, use_cache=False)
    ok = all(torch.equal(c0["cal"]["weight_params"][k],
                         c1["cal"]["weight_params"][k])
             for k in c0["cal"]["weight_params"])
    s0 = [c["score"] for c in (score_case(ship, p, H.dequantize_nvfp4(*group["weight"]),
                                          V.deq(V.quant_alg1(
                                              H.dequantize_nvfp4(*group["weight"]).float())),
                                          c0["cal"]["weight_params"],
                                          c0["cal"]["activation_state"])
                               for p in group["test_activation_list"][:2])]
    print(f"[selftest] C={C} N={N} bit-identical(off)={ok} scores "
          f"{['%.4f' % s for s in s0]}")
    assert ok, "patched-off module diverged from ship"


# ---------------------------------------------------------------------------
# fidelity: (a) diag vs (b) box-relaxed vs (c) actual refinement oracle
# ---------------------------------------------------------------------------
def run_fit():
    res = {}
    if os.path.exists(RES_FIT):
        with open(RES_FIT) as f:
            res = json.load(f)
    full = {g[0]: g for g in iter_grid()}
    ks = (0.5, 1.0, 2.0)
    cfg = {"on": True, "k": 1.0, "ks": ks, "guard": False,
           "fit_rows": 650, "guard_rows": 24}
    mod = load_sol(cfg)
    for C, N, spread, outp in FIT_GROUPS:
        name = f"c{C}_n{N}_s{spread}_o{outp}"
        if name in res:
            print(f"[fit] {name}: cached, skip")
            continue
        t0 = time.perf_counter()
        group = make_group(*full[name][1:6])
        cc = calibrate(mod, group, "fitoff", name, use_cache=False)
        L = mod._CAW_LAST
        assert L.get("on"), f"{name}: stage did not run"
        xh = L["xh_pick"].contiguous()
        W = L["w_final"]
        variants_q = {"rtn": L["q_rtn"], "gptq": L["q_g"]}
        for k in ks:
            variants_q[f"caw{k}"] = L["cands"][k]
        entry = {"C": C, "N": N, "spread": spread, "outlier_p": outp,
                 "info": {k: float(v) for k, v in L["info"]["beta_mean"].items()},
                 "gptq_s": L["info"]["gptq_s"], "rows": int(xh.shape[0]),
                 "variants": {}}
        for tag, q in variants_q.items():
            mse_plain = ((xh @ q.T - xh @ W.T) ** 2).mean().item()
            mse_or = mod._caw_ref_mse(xh, W, q)
            Aq = caw_A(mod, xh, W, q, 1.0)
            mse_box = ju_box(xh, W, q, Aq)
            mse_diag = ju_diag(xh, W, q, Aq)
            entry["variants"][tag] = {"mse_plain": mse_plain,
                                      "mse_oracle": mse_or,
                                      "mse_box": mse_box,
                                      "mse_diag": mse_diag}
            print(f"[fit] {name} {tag}: plain {mse_plain:.3e} oracle {mse_or:.3e} "
                  f"box {mse_box:.3e} diag {mse_diag:.3e}")
        floor = mod._caw_ref_mse(xh, W, W)
        entry["act_floor_oracle"] = floor
        res[name] = entry
        with open(RES_FIT, "w") as f:
            json.dump(res, f, indent=1)
        print(f"[fit] {name}: done {time.perf_counter()-t0:.1f}s "
              f"floor {floor:.3e} beta {entry['info']}")
        sys.stdout.flush()
    print("[fit] complete")


# ---------------------------------------------------------------------------
# end-to-end suite vs ship
# ---------------------------------------------------------------------------
def run_suite(k, limit=None, force=False):
    tag2 = "cawf" if force else "caw"
    res = {}
    if os.path.exists(RES_SUITE):
        with open(RES_SUITE) as f:
            res = json.load(f)
    res = {n: e for n, e in res.items()
           if e.get("k") == k}          # k change invalidates cache
    ship = load_sol()
    cfg = {"on": True, "k": k, "ks": (k,), "guard": True,
           "fit_rows": 650, "guard_rows": 24}
    if force:
        cfg["force"] = True
    caw = load_sol(cfg)
    grid = suite_grid()
    if limit:
        grid = grid[:limit]
    print(f"[suite] {len(grid)} groups, k={k}, {tag2}")
    for name, seed, C, N, spread, outp in grid:
        if name in res and tag2 in res[name]:
            print(f"[suite] {name}: cached, skip")
            continue
        t0 = time.perf_counter()
        group = make_group(seed, C, N, spread, outp)
        w_ref = H.dequantize_nvfp4(*group["weight"])
        w_std = V.deq(V.quant_alg1(w_ref.float()))
        entry = res.get(name, {"C": C, "N": N, "spread": spread,
                               "outlier_p": outp, "k": k})
        if "ship" not in entry:
            cc = calibrate(ship, group, "ship", name)
            st = cc["cal"]["activation_state"]
            wp = cc["cal"]["weight_params"]
            entry["ship"] = {"cal_s": cc["cal_s"],
                             "cases": [score_case(ship, p, w_ref, w_std, wp, st)
                                       for p in group["test_activation_list"]]}
            sc = [c["score"] * 100 for c in entry["ship"]["cases"]]
            print(f"[suite] {name} ship: cal {cc['cal_s']:.1f}s pp "
                  f"{['%.1f' % s for s in sc]}")
        cc = calibrate(caw, group, tag2, name)
        st = cc["cal"]["activation_state"]
        wp = cc["cal"]["weight_params"]
        entry[tag2] = {"cal_s": cc["cal_s"],
                       "cases": [score_case(caw, p, w_ref, w_std, wp, st)
                                 for p in group["test_activation_list"]]}
        L = caw._CAW_LAST
        entry["stagef" if force else "stage"] = {
            "info": L.get("info"), "guard": L.get("guard"),
            "grams_carried": st["gw"] is not None}
        sc = [c["score"] * 100 for c in entry[tag2]["cases"]]
        print(f"[suite] {name} {tag2}: cal {cc['cal_s']:.1f}s pp "
              f"{['%.1f' % s for s in sc]}")
        res[name] = entry
        with open(RES_SUITE, "w") as f:
            json.dump(res, f, indent=1)
        print(f"[suite] {name}: done {time.perf_counter()-t0:.1f}s")
        sys.stdout.flush()
    print("[suite] complete")


# ---------------------------------------------------------------------------
# mini_sample (eval.py conventions: quant_norm7 / hif4_quantize_standard)
# ---------------------------------------------------------------------------
def run_mini(k):
    res = {}
    if os.path.exists(RES_MINI):
        with open(RES_MINI) as f:
            res = json.load(f)
    if res.get("k") == k and "cases" in res.get("caw", {}):
        print("[mini] cached, skip")
        return
    ship = load_sol()
    caw = load_sol({"on": True, "k": k, "ks": (k,), "guard": True,
                    "fit_rows": 650, "guard_rows": 24})
    linear = torch.load(os.path.join(ROOT, "example", "mini_sample",
                                     "linear.pt"), weights_only=True)
    res = {"k": k, "groups": []}
    for gi, g in enumerate(linear):
        w_ref = H.dequantize_nvfp4(*g["weight"])
        w_std = H.hif4_dequantize(H.hif4_quantize_standard(w_ref.float()))
        gentry = {"cases": {}}
        for tag, mod in (("ship", ship), ("caw", caw)):
            torch.manual_seed(0)
            t0 = time.perf_counter()
            cal = mod.hif4_calibration_and_quantize_weight(
                *g["weight"], g["calib_activation_list"])
            cal_s = time.perf_counter() - t0
            st = cal["activation_state"]
            wp = cal["weight_params"]
            w_play = H.hif4_dequantize(wp)
            cases = []
            for pair in g["test_activation_list"]:
                x_ref = H.dequantize_nvfp4(*pair)
                ref = H.linear_ref(x_ref, w_ref)
                x_std = V.deq(V.quant_norm7(x_ref.float()))
                mse_std = ((H.linear_ref(x_std, w_std) - ref) ** 2).mean().item()
                p = mod.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
                x_play = H.hif4_dequantize(p)
                # dev/eval.py convention (weight-error metric): exact x_ref
                mse_play = ((H.linear_ref(x_ref, w_play) - ref) ** 2).mean().item()
                # decomp-study convention (joint error): quantized x_play
                mse_joint = ((H.linear_ref(x_play, w_play) - ref) ** 2).mean().item()
                cases.append({"T": int(pair[0].shape[0]),
                              "mse_std": mse_std, "mse_play": mse_play,
                              "mse_joint": mse_joint,
                              "score": (mse_std - mse_play) / mse_std,
                              "score_joint": (mse_std - mse_joint) / mse_std})
            gentry["cases"][tag] = {"cal_s": cal_s, "cases": cases}
            if tag == "caw":
                gentry["stage"] = {"info": mod._CAW_LAST.get("info"),
                                   "guard": mod._CAW_LAST.get("guard")}
            sc = [c["score"] * 100 for c in cases]
            print(f"[mini] g{gi} {tag}: cal {cal_s:.1f}s pp "
                  f"{['%.1f' % s for s in sc]}")
        res["groups"].append(gentry)
    with open(RES_MINI, "w") as f:
        json.dump(res, f, indent=1)
    print("[mini] complete")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def rep():
    # fidelity table
    if os.path.exists(RES_FIT):
        with open(RES_FIT) as f:
            fit = json.load(f)
        print("=== fidelity (holdout rows; mse units) ===")
        print(f"{'group':>22} {'variant':>7} {'plain':>10} {'oracle':>10} "
              f"{'box':>10} {'diag':>10} {'beta1':>6}")
        for n, e in sorted(fit.items()):
            for tag, v in e["variants"].items():
                print(f"{n:>22} {tag:>7} {v['mse_plain']:>10.3e} "
                      f"{v['mse_oracle']:>10.3e} {v['mse_box']:>10.3e} "
                      f"{v['mse_diag']:>10.3e} "
                      f"{e['info'].get('1.0', float('nan')):>6.2f}")
        # rank agreement of box/diag vs oracle across variants within group
        import math
        ranks = {"box": [], "diag": [], "plain": []}
        ratios = {"box": [], "diag": []}
        for n, e in fit.items():
            vs = list(e["variants"].items())
            orv = [v["mse_oracle"] for _, v in vs]
            if len(set(orv)) < 2:
                continue
            def rank_corr(pred):
                pr = sorted(range(len(vs)), key=lambda i: pred[i])
                ora = sorted(range(len(vs)), key=lambda i: orv[i])
                rp = {j: r for r, j in enumerate(pr)}
                ro = {j: r for r, j in enumerate(ora)}
                num = sum((rp[i]-ro[i])**2 for i in range(len(vs)))
                m = len(vs)
                return 1.0 - 6.0*num/(m*(m*m-1))
            ranks["box"].append(rank_corr([v["mse_box"] for _, v in vs]))
            ranks["diag"].append(rank_corr([v["mse_diag"] for _, v in vs]))
            ranks["plain"].append(rank_corr([v["mse_plain"] for _, v in vs]))
            fl = e["act_floor_oracle"]
            for key, pk in (("box", "mse_box"), ("diag", "mse_diag")):
                rr = [(v[pk]-fl)/max(v["mse_oracle"]-fl, 1e-30)
                      for _, v in vs if v["mse_oracle"] > fl]
                ratios[key] += rr
        print(f"rank-corr vs oracle (mean over groups): "
              f"box {_mean(ranks['box']):.2f} diag {_mean(ranks['diag']):.2f} "
              f"plain {_mean(ranks['plain']):.2f}")
        print(f"(pred-floor)/(oracle-floor) ratio: box {_mean(ratios['box']):.2f} "
              f"diag {_mean(ratios['diag']):.2f}")

    # suite table
    if os.path.exists(RES_SUITE):
        with open(RES_SUITE) as f:
            res = json.load(f)
        names = sorted(res)
        print(f"\n=== suite: {len(names)} groups, k={res[names[0]]['k']} ===")
        acc2 = [1 if res[n]["stage"]["guard"]["m_caw"]
                < res[n]["stage"]["guard"]["m_ship"] else 0
                for n in names if res[n].get("stage", {}).get("guard")]
        print(f"guard accept: {sum(acc2)}/{len(acc2)}")
        Ts = (10, 128, 512, 1024)
        print(f"{'C':>6} {'n':>3} | " + " | ".join(
            f"T{T}: ship/caw/d" for T in Ts) + " || all: ship/caw/d | dcal")
        for C in CS_SUITE:
            sub = [n for n in names if res[n]["C"] == C]
            if not sub:
                continue
            cells = []
            for T in Ts:
                s0 = [c["score"]*100 for n in sub for c in res[n]["ship"]["cases"]
                      if c["T"] == T]
                s1 = [c["score"]*100 for n in sub for c in res[n]["caw"]["cases"]
                      if c["T"] == T]
                cells.append(f"{_mean(s0):6.1f}/{_mean(s1):6.1f}/"
                             f"{_mean(s1)-_mean(s0):+5.1f}")
            a0 = [c["score"]*100 for n in sub for c in res[n]["ship"]["cases"]]
            a1 = [c["score"]*100 for n in sub for c in res[n]["caw"]["cases"]]
            dcal = _mean([res[n]["caw"]["cal_s"] - res[n]["ship"]["cal_s"]
                          for n in sub])
            print(f"{C:>6} {len(sub):>3} | " + " | ".join(cells)
                  + f" || {_mean(a0):6.1f}/{_mean(a1):6.1f}/"
                  f"{_mean(a1)-_mean(a0):+5.1f} | {dcal:+5.1f}s")
        byT = {T: (_mean([c["score"]*100 for n in names
                          for c in res[n]["ship"]["cases"] if c["T"] == T]),
                   _mean([c["score"]*100 for n in names
                          for c in res[n]["caw"]["cases"] if c["T"] == T]))
               for T in Ts}
        print("by T:", {T: f"{a:.1f}->{b:.1f} ({b-a:+.1f})"
                        for T, (a, b) in byT.items()})
        stg = [res[n]["stage"]["info"]["dt_stage"] for n in names
               if res[n].get("stage", {}).get("info")]
        gdt = [res[n]["stage"]["guard"]["dt"] for n in names
               if res[n].get("stage", {}).get("guard")]
        print(f"stage dt mean {_mean(stg):.2f}s (gptq "
              f"{_mean([res[n]['stage']['info']['gptq_s'][str(res[n]['k'])] for n in names if res[n].get('stage', {}).get('info')]):.2f}s) "
              f"| guard dt mean {_mean(gdt):.2f}s")
        if all("cawf" in res[n] for n in names):
            print("forced (unguarded) caw delta vs ship, pp:")
            for C in CS_SUITE:
                sub = [n for n in names if res[n]["C"] == C]
                d = [c2["score"] * 100 - c1["score"] * 100
                     for n in sub for c1, c2 in
                     zip(res[n]["ship"]["cases"], res[n]["cawf"]["cases"])]
                print(f"  C={C}: mean {_mean(d):+.2f} worst {min(d):+.2f} "
                      f"(n={len(d)})")

    if os.path.exists(RES_MINI):
        with open(RES_MINI) as f:
            mini = json.load(f)
        print(f"\n=== mini linear (k={mini['k']}) ===")
        for gi, g in enumerate(mini["groups"]):
            for tag in ("ship", "caw"):
                sc = [c["score"]*100 for c in g["cases"][tag]["cases"]]
                print(f"g{gi} {tag}: cal {g['cases'][tag]['cal_s']:.1f}s pp "
                      f"{['%.1f' % s for s in sc]} mean {_mean(sc):.1f}")
            gr = g.get("stage", {}).get("guard")
            if gr:
                print(f"g{gi} guard: ship {gr['m_ship']:.3e} "
                      f"caw {gr['m_caw']:.3e} -> "
                      f"{'ACCEPT' if gr['m_caw'] < gr['m_ship'] else 'reject'}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "rep"
    k = 1.0
    limit = None
    args = sys.argv[2:]
    for i, a in enumerate(args):
        if a == "--k":
            k = float(args[i + 1])
        elif a == "--limit":
            limit = int(args[i + 1])
    os.makedirs(CACHE, exist_ok=True)
    if mode == "selftest":
        run_selftest()
    elif mode == "fit":
        run_fit()
    elif mode == "suite":
        run_suite(k, limit)
    elif mode == "suitef":
        run_suite(k, limit, force=True)
    elif mode == "mini":
        run_mini(k)
    else:
        rep()


if __name__ == "__main__":
    main()
