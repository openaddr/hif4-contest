"""Coordinate-descent refinement on top of the v9 linear pipeline (mini sample).

Prototype experiments ONLY -- example/solution/solution.py is untouched.

  E1  activation refinement, exact-W objective
        strategies : "all"    (flip every element whose best-step Delta < 0)
                     "greedy" (top-1 flip per row, 20 rounds per sweep)
        objectives : "task"   res = (xq - x) @ W^T          (task-book formula)
                     "true"   res = xq @ W^T - x @ w_final^T (true output error)
  E2  activation refinement, rank-64 projected objective (eigh of W^T W):
        W_k = U_k S_k V_k^T;  projected column p_c = s_k * V_k[c]; the flip
        Delta lives entirely in the 64-d coordinates: res_k = err @ P,
        Delta = 2*delta*<res_k, p_c> + delta^2*||p_c||^2.
  E3  weight refinement on calib rows (first 4 samples, <= 2048 rows),
      hold-out (calib[-1]) gated, per-element Delta over the calib objective
        sum_r ||x_r q^T - x_r w_final^T||^2.
  C   weight refinement + activation refinement combined.

Residual bookkeeping: res (T,N) is never materialized. We track its image
M = res @ W (T,C) via Gram updates  M += D @ (W^T W)  -- mathematically
identical to "res += delta * W[:,i]", one (T,C)x(C,C) matmul per sweep.

Scoring convention identical to dev/exp_common.py: baseline =
dev.variants.quant_alg1 on both weight and activation; score per test
= (mse_std - mse_play) / mse_std.  Base 5-test: +0.8209 +0.8134 +0.8540
+0.8470 +0.8570 (mean 0.83846).

Usage: python dev/refine.py [base|e1|e2|e3|combo|all]
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "example", "solution"))
sys.path.insert(0, ROOT)

INF = float("inf")


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load_mod(os.path.join(ROOT, "..", "example", "solution", "solution.py"), "sol")
import hif4  # noqa: E402
import variants as V  # noqa: E402

torch.manual_seed(0)
LIN = torch.load(os.path.join(ROOT, "..", "example", "mini_sample", "linear.pt"),
                 weights_only=True, map_location="cpu")[0]
TESTS = LIN["test_activation_list"]
CALIB = LIN["calib_activation_list"]
W_REF = hif4.dequantize_nvfp4(*LIN["weight"])
W_F = W_REF.float()
BASE_SCORES = (0.8209, 0.8134, 0.8540, 0.8470, 0.8570)
BASE_MEAN = sum(BASE_SCORES) / len(BASE_SCORES)

# ---- standard (alg1) baseline MSEs + true references, harness-faithful ----
_w_std = V.deq(V.quant_alg1(W_F))
REFS, STD_MSES = [], []
for _pair in TESTS:
    _xr = hif4.dequantize_nvfp4(*_pair)
    _ref = hif4.linear_ref(_xr, W_REF)
    REFS.append(_ref)
    _xs = V.deq(V.quant_alg1(_xr.float()))
    STD_MSES.append(((hif4.linear_ref(_xs, _w_std) - _ref) ** 2).mean().item())
del _w_std

# ---- pipeline state (filled by setup()) ----
T_CAL = None
WP = STATE = None
S_V = MODE = None
Q_USED = W_FINAL = None
ACTS: list[dict] = []
LOCAL_BASE = None  # measured scores of the CURRENT pipeline (v14, damp 0.05)


def setup():
    global T_CAL, WP, STATE, S_V, MODE, Q_USED, W_FINAL, ACTS, LOCAL_BASE
    torch.manual_seed(0)
    t0 = time.perf_counter()
    out = S.hif4_calibration_and_quantize_weight(LIN["weight"][0],
                                                 LIN["weight"][1], CALIB)
    T_CAL = time.perf_counter() - t0
    WP, STATE = out["weight_params"], out["activation_state"]
    S_V, MODE = STATE["s"], STATE["mode"]
    Q_USED = S._deq_params(WP).contiguous()
    W_FINAL = _tf_act_bare(W_F / S_V).contiguous()
    ACTS = []
    for pair, ref in zip(TESTS, REFS):
        xr = S.dequantize_nvfp4(*pair).float()
        x = _tf_act_bare(xr * S_V) if MODE == 1 else xr * S_V
        t0 = time.perf_counter()
        p = S.hif4_dynamic_quantize_activation(pair[0], pair[1], STATE)
        dt = time.perf_counter() - t0
        xq = S._deq_params(p)
        unit = S._params_unit_flat(p)
        ACTS.append({"x": x, "p": p, "xq": xq, "unit": unit,
                     "v4": torch.round(xq / unit * 4.0), "ref": ref,
                     "dt_dyn": dt})
    print(f"[setup] calibration {T_CAL:.2f}s  mode={MODE} g={STATE['g']} "
          f"order={'yes' if STATE['order'] is not None else 'no'}  "
          f"dyn s/test: [{' '.join(f'{a['dt_dyn']:.3f}' for a in ACTS)}]")
    LOCAL_BASE = scores_of(
        [((a["xq"] @ Q_USED.T - a["ref"]) ** 2).mean().item() for a in ACTS])
    print(f"[setup] local base scores: " + " ".join(f"{s:+.4f}" for s in LOCAL_BASE)
          + f" | mean {sum(LOCAL_BASE) / len(LOCAL_BASE):+.4f}")


def _tf_act_bare(x):
    return S._rot_blocks(x) if MODE == 1 else x


def scores_of(mses):
    return [(a - b) / a for a, b in zip(STD_MSES, mses)]


def report(tag, mses, extra=""):
    sc = scores_of(mses)
    m = sum(sc) / len(sc)
    ref_m = sum(LOCAL_BASE) / len(LOCAL_BASE) if LOCAL_BASE is not None else BASE_MEAN
    print(f"[{tag}] " + " ".join(f"{s:+.4f}" for s in sc)
          + f" | mean {m:+.4f} ({(m - BASE_MEAN) * 100:+.2f}pp vs recorded base,"
          f" {(m - ref_m) * 100:+.2f}pp vs local base) {extra}")
    return m


def verify_values(xq_new, p, W, ref, tag):
    """Round-trip refined values through _values_to_params and score exactly."""
    p2 = S._values_to_params(xq_new.contiguous(), p)
    xq2 = S._deq_params(p2)
    dev = (xq2 - xq_new).abs().max().item() / max(xq_new.abs().max().item(), 1e-30)
    mant_ok = p2["mant"].min().item() >= 0.0 and p2["mant"].max().item() <= 1.75
    mse = ((xq2 @ W.T - ref) ** 2).mean().item()
    print(f"[{tag}] verify roundtrip: rel dev {dev:.1e}  mant-in-[0,1.75] {mant_ok}"
          f"  exact mse {mse:.4e}")
    return mse, bool(mant_ok and dev < 1e-4)


def _selftest():
    """Validate the flip-Delta formula, the Gram-maintained M, and greedy
    monotonicity on a small random problem."""
    gen = torch.Generator().manual_seed(7)
    T, C, N = 6, 64, 32
    W = torch.randn(N, C, generator=gen)
    x = torch.randn(T, C, generator=gen)
    unit = torch.full((T, C), 0.5)
    d = 0.25 * unit
    v4 = torch.randint(-7, 8, (T, C), generator=gen).float()
    Gw = W.T @ W
    col2 = Gw.diagonal()
    M = (v4 * d - x) @ Gw
    j0 = (((v4 * d - x) @ W.T) ** 2).sum().item()
    g, dirn = _sel(d, M, col2, v4)
    r, c = 2, 17
    v4n = v4.clone()
    v4n[r, c] += dirn[r, c]
    j1 = (((v4n * d - x) @ W.T) ** 2).sum().item()
    pred = g[r, c].item()
    rel = abs((j1 - j0) - pred) / max(abs(j1 - j0), 1e-12)
    print(f"[selftest] single-flip dJ exact {j1 - j0:+.6e} formula {pred:+.6e}"
          f" rel err {rel:.1e}")
    assert rel < 1e-3, rel  # fp32 cancellation noise floor
    # Gram-maintained M after one greedy round == recomputed M
    idx = g.argmin(dim=1, keepdim=True)
    fin = torch.isfinite(g.gather(1, idx))
    dr = dirn.gather(1, idx) * fin.float()
    v4.scatter_add_(1, idx, dr)
    M2 = M.clone()
    M2 += (dr * d.gather(1, idx)) * Gw[idx[:, 0]]
    Mref = (v4 * d - x) @ Gw
    dev = (M2 - Mref).abs().max().item() / Mref.abs().max().item()
    print(f"[selftest] greedy Gram-maintained M vs recompute: rel dev {dev:.1e}")
    assert dev < 1e-3
    # greedy monotone on the toy: 50 rounds of exact CD must not increase J
    j_prev = (((v4 * d - x) @ W.T) ** 2).sum().item()
    for _ in range(50):
        g, dirn = _sel(d, M2, col2, v4)
        idx = g.argmin(dim=1, keepdim=True)
        fin = torch.isfinite(g.gather(1, idx))
        dr = dirn.gather(1, idx) * fin.float()
        v4.scatter_add_(1, idx, dr)
        M2 += (dr * d.gather(1, idx)) * Gw[idx[:, 0]]
        j_new = (((v4 * d - x) @ W.T) ** 2).sum().item()
        assert j_new <= j_prev + 1e-6 * abs(j_prev), (j_prev, j_new)
        j_prev = j_new
    print(f"[selftest] greedy 50 rounds monotone: J {j0:.4f} -> {j_prev:.4f} OK")


def verify_weight(W_new, tag):
    """Round-trip refined weight values through _values_to_params."""
    p2 = S._values_to_params(W_new.contiguous(), WP)
    w2 = S._deq_params(p2)
    dev = (w2 - W_new).abs().max().item() / max(W_new.abs().max().item(), 1e-30)
    mant_ok = p2["mant"].min().item() >= 0.0 and p2["mant"].max().item() <= 1.75
    mse = ((ACTS[2]["xq"] @ w2.T - REFS[2]) ** 2).mean().item()
    print(f"[{tag}] verify roundtrip: rel dev {dev:.1e}  mant-in-[0,1.75] {mant_ok}"
          f"  mse(test2, unrefined acts) {mse:.4e}")
    return mse, bool(mant_ok and dev < 1e-4)


def _sel(d, M, col2, v4, eps=0.0):
    """Best single-grid-step flip gain per element (negative = improves).

    Delta(s) = 2*s*d*M + d^2*col2 for step direction s in {+1,-1}; the optimum
    is s* = -sign(M) with Delta* = -2*d*|M| + d^2*col2.  If s* is illegal
    (mant at the grid edge) the other direction has Delta > 0, so no flip.
    Returns (g, dirn) with g = INF where no improving legal flip exists.
    """
    g = -2.0 * d * M.abs() + (d * d) * col2
    up = M < 0.0
    legal = torch.where(up, v4 < 7.0, v4 > -7.0)
    g = torch.where(legal & (g < -eps), g, torch.full_like(g, INF))
    dirn = torch.where(up, 1.0, -1.0)
    return g, dirn


# =============================================================================
# E1: activation refinement, exact W
# =============================================================================

def refine_act_exact(a, W, obj, strat, Gw, Gwf, sweeps=3, greedy_steps=20, eps=0.0):
    """M = res @ W maintained via Gram updates; v4 = xq/unit*4 grid indices."""
    x, unit, ref = a["x"], a["unit"], a["ref"]
    v4 = a["v4"].clone()
    d = 0.25 * unit
    col2 = Gw.diagonal()
    xq = v4 * d
    t0 = time.perf_counter()
    if obj == "task":
        M = (xq - x) @ Gw
    else:  # res = xq@W^T - x@w_final^T  ->  M = xq@(W^T W) - x@(w_final^T W)
        M = xq @ Gw - x @ Gwf
    init_t = time.perf_counter() - t0
    mses, times, nfs = [], [], []
    for sw in range(sweeps):
        ts = time.perf_counter()
        if strat == "all":
            g, dirn = _sel(d, M, col2, v4, eps)
            mask = torch.isfinite(g)
            dirn = torch.where(mask, dirn, 0.0)
            nfs.append(int(mask.sum()))
            v4 += dirn
            M += (d * dirn) @ Gw
        else:
            nf = 0
            for _ in range(greedy_steps):
                g, dirn = _sel(d, M, col2, v4, eps)
                idx = g.argmin(dim=1, keepdim=True)
                fin = torch.isfinite(g.gather(1, idx))
                dr = dirn.gather(1, idx) * fin.float()
                nf += int(fin.sum())
                v4.scatter_add_(1, idx, dr)
                M += (dr * d.gather(1, idx)) * Gw[idx[:, 0]]
            nfs.append(nf)
        times.append(time.perf_counter() - ts)
        xq = v4 * d
        mses.append(((xq @ W.T - ref) ** 2).mean().item())  # offline scoring
    return {"v4": v4, "xq": xq, "mse": mses, "t": times, "nf": nfs,
            "init_t": init_t}


def e1():
    print("\n===== E1: activation refinement, exact objective =====")
    W = Q_USED
    t0 = time.perf_counter()
    Gw = (W.T @ W).contiguous()            # calib-side precompute, shared
    Gwf = (W_FINAL.T @ W).contiguous()     # for the "true" objective
    print(f"[E1] Gram precomputes (calib-side): {time.perf_counter() - t0:.2f}s")
    results = {}
    for obj in ("task", "true"):
        for strat in ("all", "greedy"):
            tag = f"E1 {strat:6s}/{obj}"
            per = [refine_act_exact(a, W, obj, strat, Gw, Gwf) for a in ACTS]
            results[(obj, strat)] = per
            for sw in range(len(per[0]["mse"])):
                mses = [r["mse"][sw] for r in per]
                tl = " ".join(f"{r['t'][sw]:.2f}" for r in per)
                nf = sum(r["nf"][sw] for r in per)
                report(f"{tag} sweep{sw + 1}", mses,
                       f"| refine s/test [{tl}] flips {nf}")
            print(f"[{tag}] init s/test: [{' '.join(f'{r['init_t']:.3f}' for r in per)}]")
    return results


# =============================================================================
# E2: activation refinement, rank-64 projected objective
# =============================================================================

def refine_act_r64(a, W, P, GfU, obj, strat, sweeps=3, greedy_steps=20, eps=0.0):
    x, unit, ref = a["x"], a["unit"], a["ref"]
    v4 = a["v4"].clone()
    d = 0.25 * unit
    pn = (P * P).sum(1)  # ||p_c||^2 per input column
    xq = v4 * d
    t0 = time.perf_counter()
    if obj == "task":
        resk = (xq - x) @ P
    else:  # res_k = xq@P - (x @ w_final^T) @ U_k = xq@P - x@(w_final^T U_k)
        resk = xq @ P - x @ GfU
    init_t = time.perf_counter() - t0
    mses, times, nfs = [], [], []
    for sw in range(sweeps):
        ts = time.perf_counter()
        if strat == "all":
            Mk = resk @ P.T
            g, dirn = _sel(d, Mk, pn, v4, eps)
            mask = torch.isfinite(g)
            dirn = torch.where(mask, dirn, 0.0)
            nfs.append(int(mask.sum()))
            v4 += dirn
            resk += (d * dirn) @ P
        else:
            nf = 0
            for _ in range(greedy_steps):
                Mk = resk @ P.T
                g, dirn = _sel(d, Mk, pn, v4, eps)
                idx = g.argmin(dim=1, keepdim=True)
                fin = torch.isfinite(g.gather(1, idx))
                dr = dirn.gather(1, idx) * fin.float()
                nf += int(fin.sum())
                v4.scatter_add_(1, idx, dr)
                resk += (dr * d.gather(1, idx)) * P[idx[:, 0]]
            nfs.append(nf)
        times.append(time.perf_counter() - ts)
        xq = v4 * d
        mses.append(((xq @ W.T - ref) ** 2).mean().item())  # offline scoring
    return {"v4": v4, "xq": xq, "mse": mses, "t": times, "nf": nfs,
            "init_t": init_t}


def rank64_factors(Gw, W, k=64):
    """P = V_k * s_k (C,k), U_k = W V_k / s_k (N,k) from eigh of W^T W."""
    lam, Vc = torch.linalg.eigh(Gw)          # ascending
    lam = lam.flip(0)[:k].clamp_min(1e-12)
    Vk = Vc.flip(1)[:, :k].contiguous()
    sk = lam.sqrt()
    P = (Vk * sk).contiguous()
    Uk = (W @ (Vk / sk)).contiguous()
    return P, Uk


def e2():
    print("\n===== E2: activation refinement, rank-64 projected objective =====")
    W = Q_USED
    t0 = time.perf_counter()
    Gw = (W.T @ W).contiguous()
    P, Uk = rank64_factors(Gw, W)
    GfU = (W_FINAL.T @ Uk).contiguous()
    t_svd = time.perf_counter() - t0
    ev = (Gw.diagonal().sum().item(), (P ** 2).sum().item())
    print(f"[E2] Gram+eigh+U_k precomputes (calib-side): {t_svd:.2f}s  "
          f"rank-64 energy fraction {ev[1] / ev[0]:.3f}")
    results = {}
    for obj in ("task", "true"):
        for strat in ("all", "greedy"):
            tag = f"E2 {strat:6s}/{obj}"
            per = [refine_act_r64(a, W, P, GfU, obj, strat) for a in ACTS]
            results[(obj, strat)] = per
            for sw in range(len(per[0]["mse"])):
                mses = [r["mse"][sw] for r in per]
                tl = " ".join(f"{r['t'][sw]:.2f}" for r in per)
                nf = sum(r["nf"][sw] for r in per)
                report(f"{tag} sweep{sw + 1}", mses,
                       f"| refine s/test [{tl}] flips {nf}")
            print(f"[{tag}] init s/test: [{' '.join(f'{r['init_t']:.3f}' for r in per)}]")
    return results


# =============================================================================
# E3: weight refinement (calib rows, hold-out gated)
# =============================================================================

def refine_weight(strat="greedy", sweeps=3, greedy_steps=20, eps=0.0, chunk=2048):
    unit_w = S._params_unit_flat(WP)
    v4w0 = torch.round(Q_USED / unit_w * 4.0)
    xs = [_tf_act_bare(S.dequantize_nvfp4(*c).float() * S_V) if MODE == 1
          else S.dequantize_nvfp4(*c).float() * S_V for c in CALIB[:-1]]
    x_cal = torch.cat(xs)[:2048].contiguous()
    x_hold = (_tf_act_bare(S.dequantize_nvfp4(*CALIB[-1]).float() * S_V) if MODE == 1
              else S.dequantize_nvfp4(*CALIB[-1]).float() * S_V).contiguous()
    t0 = time.perf_counter()
    Gxx = (x_cal.T @ x_cal).contiguous()
    colE = Gxx.diagonal()
    A = (Q_USED - W_FINAL) @ Gxx          # A = res^T @ x_cal, res = x(q^T - w_f^T)
    ref_hold = x_hold @ W_FINAL.T
    t_init = time.perf_counter() - t0
    hold0 = ((x_hold @ Q_USED.T - ref_hold) ** 2).mean().item()
    print(f"[E3] calib rows {tuple(x_cal.shape)}  holdout rows {tuple(x_hold.shape)}"
          f"  init {t_init:.2f}s (Gxx + A + ref_hold)")
    print(f"[E3] holdout mse before: {hold0:.4e}")
    d = 0.25 * unit_w
    v4 = v4w0.clone()
    N = v4.shape[0]
    mses_hist, hold_hist, times, nfs = [], [], [], []
    for sw in range(sweeps):
        ts = time.perf_counter()
        nf = 0
        if strat == "all":
            for i1 in range(0, N, chunk):
                i2 = min(i1 + chunk, N)
                g, dirn = _sel(d[i1:i2], A[i1:i2], colE, v4[i1:i2], eps)
                mask = torch.isfinite(g)
                dirn = torch.where(mask, dirn, 0.0)
                nf += int(mask.sum())
                v4[i1:i2] += dirn
                A[i1:i2] += (d[i1:i2] * dirn) @ Gxx
        else:
            # weight rows are independent (J = sum_i ||x(q_i - w_i)^T||^2), so
            # top-1 flip per weight row is exact CD, batched over rows
            for _ in range(greedy_steps):
                for i1 in range(0, N, chunk):
                    i2 = min(i1 + chunk, N)
                    g, dirn = _sel(d[i1:i2], A[i1:i2], colE, v4[i1:i2], eps)
                    idx = g.argmin(dim=1, keepdim=True)
                    fin = torch.isfinite(g.gather(1, idx))
                    dr = dirn.gather(1, idx) * fin.float()
                    nf += int(fin.sum())
                    v4[i1:i2].scatter_add_(1, idx, dr)
                    A[i1:i2] += (dr * d[i1:i2].gather(1, idx)) * Gxx[idx[:, 0]]
        Wn = v4 * d
        hold = ((x_hold @ Wn.T - ref_hold) ** 2).mean().item()  # gate (real cost)
        times.append(time.perf_counter() - ts)
        hold_hist.append(hold)
        mses = [((ACTS[i]["xq"] @ Wn.T - REFS[i]) ** 2).mean().item()
                for i in range(len(ACTS))]                       # offline scoring
        mses_hist.append(mses)
        report(f"E3 {strat} sweep{sw + 1}", mses,
               f"| calib refine {times[-1]:.2f}s  flips {nf}  holdout {hold:.4e}")
    keep = hold_hist[-1] < hold0
    print(f"[E3] holdout after: {hold_hist[-1]:.4e}  -> "
          f"{'KEEP refinement' if keep else 'REJECT (revert)'}")
    return {"v4w": v4, "W": (v4 * d).contiguous(), "keep": keep,
            "mses": mses_hist, "hold": hold_hist, "t": times}


# =============================================================================
# combination + verdict
# =============================================================================

def combo(wres):
    print("\n===== C: weight refinement + activation refinement =====")
    if wres is not None and wres["keep"]:
        W = wres["W"]
        print("[C] using E3-refined weight (hold-out accepted)")
        verify_weight(W, "C weight")
    else:
        W = Q_USED
        print("[C] E3 rejected/absent -> original pipeline weight")
    t0 = time.perf_counter()
    Gw = (W.T @ W).contiguous()
    Gwf = (W_FINAL.T @ W).contiguous()
    P, Uk = rank64_factors(Gw, W)
    GfU = (W_FINAL.T @ Uk).contiguous()
    print(f"[C] precomputes (calib-side): {time.perf_counter() - t0:.2f}s")
    out = {}
    for name, fn in (("exact", lambda a: refine_act_exact(a, W, "true", "all", Gw, Gwf)),
                     ("exact-greedy", lambda a: refine_act_exact(a, W, "true", "greedy", Gw, Gwf)),
                     ("rank64", lambda a: refine_act_r64(a, W, P, GfU, "true", "all")),
                     ("rank64-greedy", lambda a: refine_act_r64(a, W, P, GfU, "true", "greedy"))):
        per = [fn(a) for a in ACTS]
        out[name] = per
        for sw in range(len(per[0]["mse"])):
            mses = [r["mse"][sw] for r in per]
            tt = [r["t"][sw] for r in per]
            report(f"C {name:13s} sweep{sw + 1}", mses,
                   f"| refine s/test [{' '.join(f'{t:.2f}' for t in tt)}]")
        tt = [r["init_t"] + sum(r["t"]) for r in per]
        print(f"[C {name:13s}] total incl init s/test: "
              + " ".join(f"{t:.2f}" for t in tt))
    verify_values(out["exact-greedy"][2]["xq"], ACTS[2]["p"], W, REFS[2],
                  "C exact-greedy test2")
    return out, W


def diag():
    """Sweep-depth curve for the winning config (greedy/true, exact W)."""
    print("\n===== diag: greedy/true sweep depth (6 sweeps, exact W) =====")
    W = Q_USED
    Gw = (W.T @ W).contiguous()
    Gwf = (W_FINAL.T @ W).contiguous()
    per = [refine_act_exact(a, W, "true", "greedy", Gw, Gwf, sweeps=6) for a in ACTS]
    for sw in range(6):
        mses = [r["mse"][sw] for r in per]
        tl = " ".join(f"{r['t'][sw]:.2f}" for r in per)
        report(f"diag sweep{sw + 1}", mses, f"| s/test [{tl}]")


def base():
    print("===== base: reproduce pipeline scores =====")
    mses = [((a["xq"] @ Q_USED.T - a["ref"]) ** 2).mean().item() for a in ACTS]
    report("base pipeline", mses)
    print(f"[base] recorded BASELINE   "
          + " ".join(f"{s:+.4f}" for s in BASE_SCORES)
          + f" | mean {BASE_MEAN:+.4f}"
          + "  (v13-era; current solution.py is v14 GPTQ_DAMP=0.05, +0.37pp)")


def main():
    stages = sys.argv[1:] or ["all"]
    if "all" in stages:
        stages = ["base", "e1", "e2", "e3", "diag", "combo"]
    t00 = time.perf_counter()
    _selftest()
    setup()
    if "base" in stages:
        base()
    e1r = e2r = wres = None
    if "e1" in stages:
        e1r = e1()
    if "e2" in stages:
        e2r = e2()
    if "e3" in stages:
        print("\n===== E3: weight refinement =====")
        wres = refine_weight()
    if "diag" in stages:
        diag()
    if "combo" in stages:
        cr, Wc = combo(wres)
        # ---- verdict ----
        print("\n===== verdict =====")
        lb = LOCAL_BASE_MEAN = sum(LOCAL_BASE) / len(LOCAL_BASE)
        best_name, best_mean, best_t = None, -INF, None
        for name, per in cr.items():
            sw = len(per[0]["mse"]) - 1
            m = sum(scores_of([r["mse"][sw] for r in per])) / len(ACTS)
            tt = [r["init_t"] + sum(r["t"]) for r in per]
            mt = sum(tt) / len(tt)
            print(f"[verdict] {name:13s} mean {m:+.4f} ({(m - lb) * 100:+.2f}pp vs local"
                  f" base {lb:+.4f})  dyn refine {mt:.3f}s/test (mean,"
                  f" worst {max(tt):.3f}s)")
            if m > best_mean:
                best_name, best_mean, best_t = name, m, mt
        best_tt = [r["init_t"] + sum(r["t"]) for r in cr[best_name]]
        gain = (best_mean - LOCAL_BASE_MEAN) * 100
        ok_score = best_mean - LOCAL_BASE_MEAN >= 0.005
        ok_time = best_t <= 0.5
        ok_time_worst = max(best_tt) <= 0.5
        print(f"[verdict] best: {best_name}  gain {gain:+.2f}pp  "
              f"dyn increment mean {best_t:.3f}s/test (worst {max(best_tt):.3f}s,"
              f" T=1024)  score-gain>=+0.5pp: {ok_score}  time<=0.5s mean: {ok_time}"
              f" worst: {ok_time_worst}  -> "
              + ("BUILD IT (建议装机)" if (ok_score and ok_time) else "NOT READY (不建议装机)"))
    print(f"\n[total elapsed {time.perf_counter() - t00:.1f}s]")


if __name__ == "__main__":
    main()
