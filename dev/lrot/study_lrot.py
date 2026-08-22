"""Data-fitted orthogonal rotation (mode-2 prototype) vs fixed Hadamard (mode-1).

Question: the linear pipeline's only exact-invariance transform today is the
fixed block-diagonal random Hadamard (_rot_blocks), which is optimal for
UNSTRUCTURED spectra.  The judge's linear data is strongly structured, so a
calib-fitted orthogonal R may beat it by aligning quantization error to
low-impact directions / equalizing block variance.

Candidates (fit ON smoothed calibration activations, all but the LAST calib
sample, which the pipeline itself holds out for its guards; exactly orthogonal,
sign-fixed, det +1):
  none     mode-0 baseline (identity rotation, everything else identical)
  hadamard the shipped block-diagonal H*D (mode-1 baseline)
  Ra       PCA rotation: eigh of centered calib covariance, eigenvector columns
           sorted by eigenvalue DESC (new coords = principal components)
  Rb       PCA-Hadamard hybrid: Ra @ blkdiag(H*D) -- PCA sorts the spectrum
           into near-equal-variance 64-blocks, Hadamard gaussianizes inside
  Rc       variance-equalizing: Ra columns round-robin permuted so every
           64-block gets equal total variance (block b gets eigs b, b+nb, ...)
  Rd       GPTQ-informed: eigenvectors (desc) of the uncentered act-Hessian
           Hs = sum_cal a^T a -- the same statistic the pipeline's GPTQ
           Hessians are built from

Harness: FULL current pipeline (ship config, REFINE_MAX_C untouched) with
S._rot_blocks monkeypatched to the candidate's dense fp32 R (mode-1 slot).
The pipeline's own holdout mode-choice proxy then decides per group whether
the candidate actually gets used (identical treatment for every variant).

Scoring (dev/decomp/study.py conventions): score = (mse_std - mse_play) /
mse_std with mse_std from the exact paper Alg.1 (variants.quant_alg1).
Data: example/mini_sample/linear.pt + synthetic groups from dev/synth.py
(C in {1024,2048,4096} x spread {0.5,0.9} x outlier {0,0.002}, N alternating
1024/8192 by grid index, seeds 4200+13*i, calib T=(10,128,512,1024),
test T=(10,128,512,1024,1024) -- 12 groups).

Usage:
  /c/App/env/Python/python.exe dev/lrot/study_lrot.py run [--quick]
  /c/App/env/Python/python.exe dev/lrot/study_lrot.py bench
  /c/App/env/Python/python.exe dev/lrot/study_lrot.py rep
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.dirname(HERE)
ROOT = os.path.dirname(DEV)
sys.path.insert(0, DEV)
import hif4 as H        # noqa: E402
import synth            # noqa: E402
import variants as V    # noqa: E402

RES = os.path.join(HERE, "results.json")
BENCH = os.path.join(HERE, "bench.json")

CALIB_T = (10, 128, 512, 1024)
TEST_T = (10, 128, 512, 1024, 1024)
CS = (1024, 2048, 4096)
SPREADS = (0.5, 0.9)
OUTLIERS = (0.0, 0.002)

VARIANTS = ("none", "hadamard", "Ra", "Rb", "Rc", "Rd")


def load_sol():
    spec = importlib.util.spec_from_file_location(
        "_lrot_sol", os.path.join(ROOT, "example", "solution", "solution.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load_sol()
ROT_ORIG = S._rot_blocks


# ---------------------------------------------------------------------------
# smoothing replica (verbatim ops from hif4_calibration_and_quantize_weight;
# no RNG is consumed before the randperm, so this reproduces s bitwise)
# ---------------------------------------------------------------------------
def smoothing_s(weight_quant, weight_scale, calib_activation_list):
    torch.manual_seed(0)
    w = S.dequantize_nvfp4(weight_quant, weight_scale).float()
    Rn, C = w.shape
    acts_raw = [S.dequantize_nvfp4(aq, as_).float() for aq, as_ in calib_activation_list]
    abs_sum = torch.zeros(C, dtype=torch.float32)
    n_tok = 0
    a_big = None
    for a in acts_raw:
        abs_sum += a.abs().sum(dim=0)
        n_tok += a.shape[0]
        if a_big is None or a.shape[0] > a_big.shape[0]:
            a_big = a
    m = (abs_sum / max(n_tok, 1)).clamp_min(1e-12)
    logm = m.log()
    logm = logm - logm.mean()
    rows = torch.randperm(Rn)[: min(Rn, 256)]
    w_rows = w[rows]
    a_wr = a_big @ w_rows.T
    best_alpha = 0.0
    best_loss = None
    for alpha in S.ALPHA_GRID:
        s = torch.exp(logm * alpha)
        wp = S._quant_weight_fast(w_rows / s, torch.ones(1, C))
        wq = (wp["sign"] * wp["mant"] * wp["scale_lv3"] * wp["scale_lv2"]
              * wp["scale_factor"]).flatten(-4, -1) * s
        loss = ((a_big @ wq.T - a_wr) ** 2).mean().item()
        if best_loss is None or loss < best_loss:
            best_loss, best_alpha = loss, alpha
    s = torch.exp(logm * best_alpha)
    return s, [a * s for a in acts_raw]


# ---------------------------------------------------------------------------
# candidate R construction (exactly orthogonal, sign-fixed, det +1)
# ---------------------------------------------------------------------------
def _sign_fix(M):
    idx = M.abs().argmax(dim=0)
    sgn = torch.sign(M[idx, torch.arange(M.shape[1], dtype=torch.long)])
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    return M * sgn.unsqueeze(0)


def _det_sign(M):
    """LAPACK det() underflows to 0.0 for C>=1024 on this torch build; slogdet
    (LU + log-sum) is exact enough."""
    return torch.linalg.slogdet(M.double())[0].item()


def _det_fix(M):
    M = M.contiguous()
    if _det_sign(M) < 0:
        M = M.clone()
        M[:, -1] *= -1
    return M


def _eigh_desc(Mat):
    ev, Vec = torch.linalg.eigh(Mat)
    return ev.flip(-1), Vec.flip(-1).contiguous()


def _blk_hadamard(C):
    """block-diagonal H64*D_b with the solution's per-block sign seeds."""
    nb = C // 64
    H = torch.tensor([[1.0]])
    while H.shape[0] < 64:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    H64 = H / 8.0
    d = torch.empty(nb, 64)
    for b in range(nb):
        g = torch.Generator().manual_seed(777 + b)
        d[b] = (torch.rand(64, generator=g) < 0.5).float() * 2 - 1
    Rm = H64.unsqueeze(0) * d.unsqueeze(1)          # (nb, 64, 64)
    blk = torch.zeros(C, C, dtype=torch.float32)
    for b in range(nb):
        blk[b * 64:(b + 1) * 64, b * 64:(b + 1) * 64] = Rm[b]
    return blk


def fit_R(kind, acts_s_holdout_free, C, basis_cache=None):
    """acts_s_holdout_free: list of smoothed calib tensors EXCLUDING the last.
    basis_cache: dict reused across kinds so the (expensive) eigendecomposition
    runs once per group per matrix."""
    nb = C // 64
    kind = kind[-1].lower()          # "Ra" -> "a"
    if basis_cache is None:
        basis_cache = {}
    X = None
    if kind == "d":
        if "hess" not in basis_cache:
            X = torch.cat(acts_s_holdout_free, dim=0).contiguous()
            _, basis_cache["hess"] = _eigh_desc(X.T @ X)
        Vd = _sign_fix(basis_cache["hess"])
        return _det_fix(Vd)
    if "cov" not in basis_cache:
        X = torch.cat(acts_s_holdout_free, dim=0).contiguous()
        mu = X.mean(dim=0, keepdim=True)
        Sig = ((X - mu).T @ (X - mu)) / X.shape[0]
        _, basis_cache["cov"] = _eigh_desc(Sig)
    Vp = _sign_fix(basis_cache["cov"])
    if kind == "a":
        return _det_fix(Vp)
    if kind == "b":
        R = Vp @ _blk_hadamard(C)
        return _det_fix(R)
    if kind == "c":
        # round-robin: new column j (block b=j//64, slot k=j%64) takes
        # eigenvector k*nb + b -> every block draws across the whole spectrum
        src = torch.empty(C, dtype=torch.long)
        for j in range(C):
            b, k = divmod(j, 64)
            src[j] = k * nb + b
        return _det_fix(Vp[:, src])
    raise ValueError(kind)


def check_orth(R):
    I = torch.eye(R.shape[0], dtype=torch.float32)
    err = (R @ R.T - I).abs().max().item()
    det = _det_sign(R) * float(torch.exp(torch.linalg.slogdet(R.double())[1]))
    ok = torch.allclose(R @ R.T, I, atol=1e-4) and abs(det - 1.0) < 1e-2
    return ok, err, det


def mk_rot(R):
    def f(t):
        return t @ R
    return f


IDENT = lambda t: t      # noqa: E731


# ---------------------------------------------------------------------------
# groups
# ---------------------------------------------------------------------------
def synth_groups():
    out = []
    i = 0
    for C in CS:
        for spread in SPREADS:
            for outp in OUTLIERS:
                N = 1024 if i % 2 == 0 else 8192
                seed = 4200 + 13 * i
                g = synth.make_linear_group(seed, N, C, tokens=CALIB_T + TEST_T,
                                            spread=spread, outlier_p=outp)
                nc = len(CALIB_T)
                out.append((f"c{C}_n{N}_s{spread}_o{outp}", {
                    "weight": g["weight"],
                    "calib_activation_list": g["calib_activation_list"][:nc],
                    "test_activation_list": g["test_activation_list"][nc:],
                }))
                i += 1
    return out


def mini_group():
    lin = torch.load(os.path.join(ROOT, "example", "mini_sample", "linear.pt"),
                     weights_only=True, map_location="cpu")[0]
    return ("mini", lin)


def all_groups():
    return [mini_group()] + synth_groups()


# ---------------------------------------------------------------------------
# scoring (dev/decomp/study.py score_case, ship refine config untouched)
# ---------------------------------------------------------------------------
def score_case(pair, w_ref, w_std, weight_params, st):
    x_ref = H.dequantize_nvfp4(*pair)
    ref = H.linear_ref(x_ref, w_ref)
    x_std = V.deq(V.quant_alg1(x_ref.float()))
    mse_std = ((H.linear_ref(x_std, w_std) - ref) ** 2).mean().item()
    t0 = time.perf_counter()
    p = S.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
    dt = time.perf_counter() - t0
    x_play = H.hif4_dequantize(p)
    w_play = H.hif4_dequantize(weight_params)
    mse_play = ((H.linear_ref(x_play, w_play) - ref) ** 2).mean().item()
    return {
        "T": int(pair[0].shape[0]),
        "mse_std": mse_std,
        "mse_play": mse_play,
        "score": (mse_std - mse_play) / mse_std,
        "dyn_s": dt,
    }


def state_bytes(st, R=None):
    tot = 0
    for v in st.values():
        if isinstance(v, torch.Tensor):
            tot += v.numel() * v.element_size()
    extra = {}
    if R is not None:
        extra["R_fp32"] = R.numel() * 4
        extra["R_bf16"] = R.numel() * 2
    return tot, extra


# ---------------------------------------------------------------------------
# main run
# ---------------------------------------------------------------------------
def jload(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def jsave(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def run(quick):
    res = jload(RES)
    groups = all_groups()
    if quick:
        groups = [groups[0], groups[1], groups[4]]
    for gname, group in groups:
        w_ref = H.dequantize_nvfp4(*group["weight"]).float()
        w_std = V.deq(V.quant_alg1(w_ref.float()))
        s = None
        acts_s = None
        basis_cache = {}
        for vn in VARIANTS:
            key = f"{gname}|{vn}"
            if key in res:
                print(f"[run] {key}: cached", flush=True)
                continue
            t_fit = 0.0
            Rm = None
            if vn in ("Ra", "Rb", "Rc", "Rd"):
                if s is None:
                    t0 = time.perf_counter()
                    s, acts_s = smoothing_s(group["weight"][0], group["weight"][1],
                                            group["calib_activation_list"])
                    t_fit += time.perf_counter() - t0
                if vn not in basis_cache:
                    t0 = time.perf_counter()
                    Rm = fit_R(vn, acts_s[:-1], w_ref.shape[1], basis_cache)
                    t_fit += time.perf_counter() - t0
                else:
                    Rm = fit_R(vn, acts_s[:-1], w_ref.shape[1], basis_cache)
                ok, oerr, det = check_orth(Rm)
                if not ok:
                    print(f"[run] {key}: NOT ORTHOGONAL err={oerr:.2e} det={det:.3f} SKIP",
                          flush=True)
                    res[key] = {"orth_ok": False}
                    continue
            else:
                ok, oerr, det = True, 0.0, 1.0
            rot = (IDENT if vn == "none"
                   else ROT_ORIG if vn == "hadamard" else mk_rot(Rm))
            S._rot_blocks = rot
            try:
                torch.manual_seed(0)
                t0 = time.perf_counter()
                cal = S.hif4_calibration_and_quantize_weight(
                    group["weight"][0], group["weight"][1],
                    group["calib_activation_list"])
                t_cal = time.perf_counter() - t0
                st = cal["activation_state"]
                wp = cal["weight_params"]
                cases = [score_case(p, w_ref, w_std, wp, st)
                         for p in group["test_activation_list"]]
                # holdout guard: the LAST calib sample, never used for fitting
                hold = score_case(group["calib_activation_list"][-1], w_ref, w_std, wp, st)
                sb, extra = state_bytes(st, Rm)
            finally:
                S._rot_blocks = ROT_ORIG
            res[key] = {
                "orth_ok": True, "orth_err": oerr, "det": det,
                "t_fit_s": t_fit, "t_cal_s": t_cal,
                "mode": int(st["mode"]), "g": int(st["g"]),
                "tmax": int(st.get("tmax") or 0),
                "state_bytes": sb, "state_extra": extra,
                "cases": cases, "hold": hold,
            }
            jsave(RES, res)
            sc = [c["score"] * 100 for c in cases]
            print(f"[run] {key}: mode={st['mode']} cal={t_cal:.1f}s fit={t_fit:.1f}s "
                  f"hold={hold['score'] * 100:+.2f}pp | "
                  f"{['%.1f' % x for x in sc]}", flush=True)
    print("[run] complete", flush=True)


# ---------------------------------------------------------------------------
# bench: SVD/eigh fit cost + dynamic per-call rotation cost
# ---------------------------------------------------------------------------
def bench():
    out = {}
    torch.manual_seed(7)
    for C in CS:
        n = 1674  # mini/synth calib rows excluding holdout
        X = torch.randn(n, C, dtype=torch.float32)
        Sig = X.T @ X / n
        for name, fn in (("eigh_cov", lambda: _eigh_desc(Sig)),
                         ("svd_X", lambda: torch.linalg.svd(X, full_matrices=False))):
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t0)
            out[f"fit_{name}_C{C}"] = sorted(ts)[1]
        R = torch.linalg.qr(torch.randn(C, C, dtype=torch.float32))[0]
        Hb = _blk_hadamard(C)
        for T in (10, 128, 512, 1024):
            x = torch.randn(T, C, dtype=torch.float32)
            for name, fn in (("hadamard", lambda: ROT_ORIG(x)),
                             ("fullR", lambda: x @ R)):
                fn()
                ts = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    fn()
                    ts.append(time.perf_counter() - t0)
                out[f"apply_{name}_C{C}_T{T}"] = sorted(ts)[2]
        print(f"[bench] C={C} done", flush=True)
    jsave(BENCH, out)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def rep():
    res = jload(RES)
    benchd = jload(BENCH)
    groups = [k.split("|")[0] for k in res if "|" in k]
    gnames = sorted(set(groups), key=lambda s: (s != "mini", s))
    print(f"{'case':>26} " + "".join(f"{v:>9}" for v in VARIANTS))
    means = {v: [] for v in VARIANTS}
    holds = {v: [] for v in VARIANTS}
    modes = {v: [] for v in VARIANTS}
    for gn in gnames:
        row = []
        for v in VARIANTS:
            e = res.get(f"{gn}|{v}")
            if e is None or not e.get("orth_ok", False):
                row.append("     ---")
                continue
            m = sum(c["score"] for c in e["cases"]) / len(e["cases"]) * 100
            means[v].append(m)
            holds[v].append(e["hold"]["score"] * 100)
            modes[v].append(e["mode"])
            row.append(f"{m:+9.2f}")
        print(f"{gn:>26} " + "".join(row))
    print(f"{'MEAN':>26} " + "".join(
        f"{sum(means[v]) / len(means[v]):+9.2f}" if means[v] else "     ---"
        for v in VARIANTS))
    print(f"{'HOLDOUT mean':>26} " + "".join(
        f"{sum(holds[v]) / len(holds[v]):+9.2f}" if holds[v] else "     ---"
        for v in VARIANTS))
    print(f"{'mode=1 frac':>26} " + "".join(
        f"{sum(modes[v]) / len(modes[v]):9.2f}" if modes[v] else "     ---"
        for v in VARIANTS))
    # delta vs hadamard per group
    print("\ndelta vs hadamard (pp/case, mean over each group's 5 tests):")
    for v in VARIANTS:
        if v == "hadamard" or not means[v]:
            continue
        ds = []
        for gn in gnames:
            a = res.get(f"{gn}|{v}")
            b = res.get(f"{gn}|hadamard")
            if a is None or b is None or not a.get("orth_ok", False):
                continue
            ma = sum(c["score"] for c in a["cases"]) / len(a["cases"])
            mb = sum(c["score"] for c in b["cases"]) / len(b["cases"])
            ds.append((ma - mb) * 100)
        if ds:
            print(f"  {v:>9}: mean {sum(ds) / len(ds):+.3f}  min {min(ds):+.3f}  "
                  f"max {max(ds):+.3f}  wins {sum(d > 0 for d in ds)}/{len(ds)}")
    # holdout guard: R vs hadamard on the holdout sample
    print("\nholdout guard (R fit on calib[:-1], scored on calib[-1]):")
    for v in VARIANTS:
        if v == "hadamard" or not holds[v]:
            continue
        ds = []
        for gn in gnames:
            a = res.get(f"{gn}|{v}")
            b = res.get(f"{gn}|hadamard")
            if a is None or b is None or not a.get("orth_ok", False):
                continue
            ds.append((a["hold"]["score"] - b["hold"]["score"]) * 100)
        if ds:
            print(f"  {v:>9}: mean {sum(ds) / len(ds):+.3f}pp  "
                  f"wins {sum(d > 0 for d in ds)}/{len(ds)}")
    if benchd:
        print("\nbench:")
        for k in sorted(benchd):
            print(f"  {k:>28}: {benchd[k] * 1000:8.2f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("run", "bench", "rep"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.mode == "run":
        run(args.quick)
    elif args.mode == "bench":
        bench()
    else:
        rep()


if __name__ == "__main__":
    main()
