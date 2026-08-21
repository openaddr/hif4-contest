"""Decomposition study of the remaining linear error (v21) + small-T sf-prior
prototype.  All work in-memory / cached here; example/solution/solution.py is
never modified.

Task A (shape decomposition):
  - synthetic linear groups: C in {512,1024,2048,4096}, N in {1024,8192},
    spread in {0.5,0.9}, outlier_p in {0.0,0.002}; calib T=(10,128,512,1024),
    test T=(10,128,512,1024,1024).  One make_linear_group call per group so
    calib and test share the channel-gain structure (as on the judge).
  - variants: refined (REFINE_MAX_C=1e9: lattice grams + act refine at every
    C, i.e. the hash-even branch forced) vs unrefined (no weight lattice
    refine, grams stripped: the v14 path).
  - per-case score mirrors dev/diag3.py: score = (mse_std - mse_play)/mse_std
    with mse_std from variants.quant_alg1 (exact paper Alg.1) on x_ref/w_ref.

Task B (sf-prior prototype):
  - dynamic-time anchor max(test_absmax, beta * calib_block_absmax_prior),
    beta in {0.5, 0.7, 1.0}; prior = per-64-block absmax max over all calib
    rows, in the transformed space (after s / rotation), fp16-carried.
    Implemented by swapping sol._quant_chunk/_quant_chunk_vec for anchored
    twins (identical code except the anchor line) around the untouched
    sol.hif4_dynamic_quantize_activation.

Usage:
  python dev/decomp/study.py A --C 512 [--N 1024,8192] [--limit k]
  python dev/decomp/study.py B --C 512 [--betas 0.5,0.7,1.0]
  python dev/decomp/study.py rep
"""
from __future__ import annotations

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
import hif4 as H          # noqa: E402
import synth              # noqa: E402
import variants as V      # noqa: E402

CACHE = os.path.join(HERE, "cache")
RES_A = os.path.join(HERE, "results_A.json")
RES_B = os.path.join(HERE, "results_B.json")

CALIB_T = (10, 128, 512, 1024)
TEST_T = (10, 128, 512, 1024, 1024)
CS = (512, 1024, 2048, 4096)
NS = (1024, 8192)
SPREADS = (0.5, 0.9)
OUTLIERS = (0.0, 0.002)
BETAS = (0.5, 0.7, 1.0)


def load_sol():
    spec = importlib.util.spec_from_file_location(
        "_decomp_sol", os.path.join(ROOT, "example", "solution", "solution.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_SOL = load_sol()
SHIP_REFINE_MAX_C = _SOL.REFINE_MAX_C


# ---------------------------------------------------------------------------
# anchored quantizer twins (solution code copied verbatim except the anchor)
# ---------------------------------------------------------------------------
_ANCHOR = {"beta": 0.0, "prior": None}


def _anchored(amax):
    """amax: (r, nb) per-row per-64-block absmax -> anchored absmax."""
    beta = _ANCHOR["beta"]
    prior = _ANCHOR["prior"]
    if prior is None or beta <= 0.0:
        return amax
    return torch.maximum(amax, (beta * prior).clamp_min(0.0))


def _quant_chunk_a(xb, wblk, grid=None):
    if grid is None:
        grid = _SOL.CAND_GRID
    ab = xb.abs()
    amax = ab.amax(dim=(2, 3, 4))                       # (r, nb)
    t = (_anchored(amax) / 7.0).clamp_min(1e-38)
    e0 = torch.floor(torch.log2(t))                     # (r, nb)

    err_best = None
    sf_best = None
    lv2_best = None
    lv3_best = None
    for k_off, sig in grid:
        sf = (torch.exp2(e0 + k_off) * sig).clamp(_SOL.SF_MIN, _SOL.SF_MAX)
        sf5 = sf[..., None, None, None]
        best_e2 = None
        best_l2 = None
        best_l3 = None
        for lv2_c in (1.0, 2.0):
            e3_list = []
            for lv3_c in (1.0, 2.0):
                unit = sf5 * lv2_c * lv3_c
                mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
                e3_list.append(((mant * unit - ab) ** 2 * wblk).sum(dim=4))  # (r,nb,8,2)
            take1 = e3_list[0] <= e3_list[1]
            e3 = torch.where(take1, e3_list[0], e3_list[1])
            lv3 = torch.where(take1, 1.0, 2.0)
            e2 = e3.sum(dim=3)                                              # (r,nb,8)
            if best_e2 is None:
                best_e2, best_l2, best_l3 = e2, lv2_c, lv3
            else:
                take2 = e2 < best_e2
                best_e2 = torch.where(take2, e2, best_e2)
                best_l2 = torch.where(take2, lv2_c, best_l2)
                best_l3 = torch.where(take2.unsqueeze(-1), lv3, best_l3)
        err = best_e2.sum(dim=2)                                            # (r,nb)
        if err_best is None:
            err_best, sf_best = err, sf
            lv2_best, lv3_best = best_l2, best_l3
        else:
            take = err < err_best
            take2 = take.unsqueeze(-1)
            take3 = take2.unsqueeze(-1)
            err_best = torch.where(take, err, err_best)
            sf_best = torch.where(take, sf, sf_best)
            lv2_best = torch.where(take2, best_l2, lv2_best)
            lv3_best = torch.where(take3, best_l3, lv3_best)

    sf = sf_best[..., None, None, None]
    lv2 = lv2_best[..., None, None]
    lv3 = lv3_best[..., None]
    unit = sf * lv2 * lv3
    mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
    return {
        "scale_factor": sf,
        "scale_lv2": lv2,
        "scale_lv3": lv3,
        "sign": torch.sign(xb),
        "mant": mant,
    }


def _quant_chunk_vec_a(xb, wblk, grid):
    KB = 2
    ab = xb.abs()
    amax = ab.amax(dim=(2, 3, 4))                       # (r, nb)
    t = (_anchored(amax) / 7.0).clamp_min(1e-38)
    e0 = torch.floor(torch.log2(t))                     # (r, nb)
    K = len(grid)
    offs = torch.tensor([float(k) for k, _ in grid])
    sigs = torch.tensor([float(s) for _, s in grid])
    sf_all = (torch.exp2(e0.unsqueeze(-1) + offs) * sigs).clamp(_SOL.SF_MIN, _SOL.SF_MAX)
    abB = ab.unsqueeze(2)                    # (r,nb,1,8,2,4) view
    wbB = (wblk.unsqueeze(2) if wblk.dim() == 5
           else wblk.unsqueeze(0).unsqueeze(2))  # broadcastable (r?,nb,1,8,2,4)
    r, nb = e0.shape
    tmp = torch.empty((r, nb, KB, 8, 2, 4), dtype=torch.float32)

    def run_batch(sf):
        kB = sf.shape[2]
        best_e2 = best_l2 = best_l3 = None
        for lv2_c in (1.0, 2.0):
            e3_list = []
            for lv3_c in (1.0, 2.0):
                unit = (sf.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                        * lv2_c * lv3_c)                     # (r,nb,kB,1,1,1)
                tgt = tmp[:, :, :kB] if kB < KB else tmp
                torch.div(abB, unit, out=tgt)
                tgt.mul_(4.0)
                tgt.round_()
                tgt.mul_(0.25)
                tgt.clamp_(0.0, 1.75)                        # mant
                tgt.mul_(unit)
                tgt.sub_(abB)
                tgt.pow_(2)
                tgt.mul_(wbB)
                e3_list.append(tgt.sum(dim=5))               # (r,nb,kB,8,2)
            take1 = e3_list[0] <= e3_list[1]                 # lv3=1.0 wins ties
            e3 = torch.where(take1, e3_list[0], e3_list[1])
            lv3c = torch.where(take1, 1.0, 2.0)              # (r,nb,kB,8,2)
            e2 = e3.sum(dim=4)                               # (r,nb,kB,8)
            if best_e2 is None:
                best_e2 = e2
                best_l2 = torch.full_like(e2, lv2_c)
                best_l3 = lv3c
            else:
                take2 = e2 < best_e2                         # earlier lv2 wins ties
                best_e2 = torch.where(take2, e2, best_e2)
                best_l2 = torch.where(take2, torch.full_like(e2, lv2_c), best_l2)
                best_l3 = torch.where(take2.unsqueeze(-1), lv3c, best_l3)
        return best_e2.sum(dim=3), best_l2, best_l3          # (r,nb,kB)

    err_best = sf_best = lv2_best = lv3_best = None
    for k0 in range(0, K, KB):
        sf = sf_all[:, :, k0:k0 + KB]
        err, l2, l3 = run_batch(sf)
        for kk in range(err.shape[2]):
            err_k = err[:, :, kk]
            if err_best is None:
                err_best = err_k
                sf_best = sf[:, :, kk]
                lv2_best = l2[:, :, kk]
                lv3_best = l3[:, :, kk]
            else:
                take = err_k < err_best                      # earlier cand wins ties
                take2 = take.unsqueeze(-1)
                take3 = take.unsqueeze(-1).unsqueeze(-1)
                err_best = torch.where(take, err_k, err_best)
                sf_best = torch.where(take, sf[:, :, kk], sf_best)
                lv2_best = torch.where(take2, l2[:, :, kk], lv2_best)
                lv3_best = torch.where(take3, l3[:, :, kk], lv3_best)

    sf = sf_best[..., None, None, None]
    lv2 = lv2_best[..., None, None]
    lv3 = lv3_best[..., None]
    unit = sf * lv2 * lv3
    mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
    return {
        "scale_factor": sf,
        "scale_lv2": lv2,
        "scale_lv3": lv3,
        "sign": torch.sign(xb),
        "mant": mant,
    }


class _Anchored:
    """Swap sol._quant_chunk/_quant_chunk_vec for the anchored twins."""

    def __init__(self, prior, beta):
        self.prior = prior
        self.beta = beta

    def __enter__(self):
        self._oc, self._ov = _SOL._quant_chunk, _SOL._quant_chunk_vec
        _SOL._quant_chunk = _quant_chunk_a
        _SOL._quant_chunk_vec = _quant_chunk_vec_a
        _ANCHOR["prior"] = self.prior
        _ANCHOR["beta"] = self.beta
        return self

    def __exit__(self, *exc):
        _SOL._quant_chunk, _SOL._quant_chunk_vec = self._oc, self._ov
        _ANCHOR["prior"] = None
        _ANCHOR["beta"] = 0.0
        return False


# ---------------------------------------------------------------------------
# group construction / calibration / scoring
# ---------------------------------------------------------------------------
def iter_grid(c_filter, n_filter, limit=None):
    # seeds are indexed over the FULL grid so a filtered run reproduces the
    # same groups as the full run
    out = []
    i = 0
    for C in CS:
        for N in NS:
            for spread in SPREADS:
                for outp in OUTLIERS:
                    name = f"c{C}_n{N}_s{spread}_o{outp}"
                    if (c_filter is None or C in c_filter) and \
                       (n_filter is None or N in n_filter):
                        out.append((name, 4200 + 13 * i, C, N, spread, outp))
                    i += 1
    return out[:limit] if limit else out


def make_group(seed, C, N, spread, outlier_p):
    g = synth.make_linear_group(seed, N, C, tokens=CALIB_T + TEST_T,
                                spread=spread, outlier_p=outlier_p)
    nc = len(CALIB_T)
    return {
        "weight": g["weight"],
        "calib_activation_list": g["calib_activation_list"][:nc],
        "test_activation_list": g["test_activation_list"][nc:],
    }


def calibrate(group, variant):
    """variant: 'refined' (forced lattice, any C) or 'unrefined' (v14 path)."""
    orig_refw = _SOL._refine_weight_values
    try:
        if variant == "refined":
            _SOL.REFINE_MAX_C = 10 ** 9
        else:
            _SOL.REFINE_MAX_C = -1
            _SOL._refine_weight_values = (
                lambda w_final, q_used, weight_params, calib_sm, tf:
                (weight_params, q_used))
        torch.manual_seed(0)
        t0 = time.perf_counter()
        cal = _SOL.hif4_calibration_and_quantize_weight(
            group["weight"][0], group["weight"][1],
            group["calib_activation_list"])
        dt = time.perf_counter() - t0
    finally:
        _SOL._refine_weight_values = orig_refw
        _SOL.REFINE_MAX_C = SHIP_REFINE_MAX_C
    if variant == "unrefined":
        st = cal["activation_state"]
        st["gw"] = None
        st["gwf"] = None
    return cal, dt


def transform_x(pair_or_t, st):
    if isinstance(pair_or_t, tuple):
        x = _SOL.dequantize_nvfp4(pair_or_t[0], pair_or_t[1]).float()
    else:
        x = pair_or_t.float()
    s = st.get("s")
    if isinstance(s, torch.Tensor) and s.numel() == x.shape[1]:
        x = x * s.float()
    if st.get("mode") == 1:
        x = _SOL._rot_blocks(x)
    return x


def block_absmax(x):
    T, C = x.shape
    return x.reshape(T, C // 64, 64).abs().amax(dim=-1)      # (T, nb)


def compute_prior(group, st):
    prior = None
    for pair in group["calib_activation_list"]:
        a = block_absmax(transform_x(pair, st)).amax(dim=0)
        prior = a if prior is None else torch.maximum(prior, a)
    p16 = prior.to(torch.float16).float()
    if bool(torch.isfinite(p16).all()):
        return p16
    return prior


def score_case(pair, w_ref, w_std, weight_params, st, refine_max_c):
    x_ref = H.dequantize_nvfp4(*pair)
    ref = H.linear_ref(x_ref, w_ref)
    x_std = V.deq(V.quant_alg1(x_ref.float()))
    mse_std = ((H.linear_ref(x_std, w_std) - ref) ** 2).mean().item()
    _SOL.REFINE_MAX_C = refine_max_c
    try:
        p = _SOL.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
    finally:
        _SOL.REFINE_MAX_C = SHIP_REFINE_MAX_C
    x_play = H.hif4_dequantize(p)
    w_play = H.hif4_dequantize(weight_params)
    mse_play = ((H.linear_ref(x_play, w_play) - ref) ** 2).mean().item()
    mse_act = ((H.linear_ref(x_play, w_ref) - ref) ** 2).mean().item()
    mse_w = ((H.linear_ref(x_ref, w_play) - ref) ** 2).mean().item()
    return {
        "T": int(pair[0].shape[0]),
        "mse_std": mse_std,
        "mse_play": mse_play,
        "mse_act": mse_act,
        "mse_w": mse_w,
        "score": (mse_std - mse_play) / mse_std,
    }


def jload(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def jsave(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


# ---------------------------------------------------------------------------
# Task A
# ---------------------------------------------------------------------------
def run_A(c_filter, n_filter, limit):
    res = jload(RES_A)
    grid = iter_grid(c_filter, n_filter, limit)
    print(f"[A] {len(grid)} groups")
    for gi, (name, seed, C, N, spread, outp) in enumerate(grid):
        if name in res and "unrefined" in res.get(name, {}).get("variants", {}):
            print(f"[A] {name}: cached, skip")
            continue
        t0 = time.perf_counter()
        group = make_group(seed, C, N, spread, outp)
        w_ref = H.dequantize_nvfp4(*group["weight"])
        parity = int(w_ref.float().double().abs().sum().item() * 1e3) % 2
        w_std = V.deq(V.quant_alg1(w_ref.float()))
        entry = res.get(name, {})
        entry.update({"C": C, "N": N, "spread": spread, "outlier_p": outp,
                      "ship_parity_even": parity == 0, "variants": {}})
        for variant in ("refined", "unrefined"):
            cpath = os.path.join(CACHE, f"{name}_{variant}.pt")
            cal, dt = calibrate(group, variant)
            torch.save({"cal": cal, "cal_s": dt}, cpath)
            st = cal["activation_state"]
            wp = cal["weight_params"]
            rmc = 10 ** 9 if variant == "refined" else SHIP_REFINE_MAX_C
            cases = [score_case(p, w_ref, w_std, wp, st, rmc)
                     for p in group["test_activation_list"]]
            entry["variants"][variant] = {"cal_s": dt, "cases": cases}
            sc = [c["score"] * 100 for c in cases]
            print(f"[A] {name} {variant}: cal {dt:.1f}s "
                  f"score pp {['%.1f' % s for s in sc]}")
        # T=10 anchor stats (transformed space, refined state)
        st = torch.load(os.path.join(CACHE, f"{name}_refined.pt"),
                        weights_only=True)["cal"]["activation_state"]
        prior = compute_prior(group, st)
        am10 = block_absmax(transform_x(group["test_activation_list"][0], st))
        ratio = (am10.amax(dim=0) / prior.clamp_min(1e-30)).tolist()
        moved = {str(b): ((b * prior).unsqueeze(0) > am10).float().mean().item()
                 for b in BETAS}
        entry["anchor_T10"] = {"ratio": ratio, "moved_frac": moved}
        res[name] = entry
        jsave(RES_A, res)
        print(f"[A] {name}: done {time.perf_counter() - t0:.1f}s "
              f"(ship parity {'even' if parity == 0 else 'odd'})")
        sys.stdout.flush()
    print("[A] complete")


# ---------------------------------------------------------------------------
# Task B
# ---------------------------------------------------------------------------
def run_B(c_filter, n_filter, betas, limit):
    resA = jload(RES_A)
    res = jload(RES_B)
    grid = iter_grid(c_filter, n_filter, limit)
    print(f"[B] {len(grid)} groups, betas {betas}")
    for gi, (name, seed, C, N, spread, outp) in enumerate(grid):
        if name in res and len(res[name].get("betas", {})) >= len(betas):
            print(f"[B] {name}: cached, skip")
            continue
        cpath = os.path.join(CACHE, f"{name}_refined.pt")
        if not os.path.exists(cpath):
            print(f"[B] {name}: no refined cache, run task A first, SKIP")
            continue
        t0 = time.perf_counter()
        group = make_group(seed, C, N, spread, outp)
        w_ref = H.dequantize_nvfp4(*group["weight"])
        w_std = V.deq(V.quant_alg1(w_ref.float()))
        cal = torch.load(cpath, weights_only=True)["cal"]
        st = cal["activation_state"]
        wp = cal["weight_params"]
        prior = compute_prior(group, st)
        entry = res.get(name, {"betas": {}})
        entry["betas"] = entry.get("betas", {})
        for beta in betas:
            if str(beta) in entry["betas"]:
                continue
            with _Anchored(prior, beta):
                cases = [score_case(p, w_ref, w_std, wp, st, 10 ** 9)
                         for p in group["test_activation_list"]]
            entry["betas"][str(beta)] = cases
            sc = [c["score"] * 100 for c in cases]
            print(f"[B] {name} beta={beta}: score pp "
                  f"{['%.1f' % s for s in sc]} ({time.perf_counter() - t0:.1f}s)")
            sys.stdout.flush()
        # control parity check on the first group of the first run
        if "beta0_check" not in entry:
            with _Anchored(prior, 0.0):
                ctrl = score_case(group["test_activation_list"][0], w_ref, w_std,
                                  wp, st, 10 ** 9)
            ref_case = resA[name]["variants"]["refined"]["cases"][0]
            rel = abs(ctrl["mse_play"] - ref_case["mse_play"]) / max(ref_case["mse_play"], 1e-30)
            entry["beta0_check"] = {"rel_mse_diff": rel, "ok": rel < 1e-9}
            print(f"[B] {name}: beta=0 parity check rel={rel:.2e} "
                  f"{'OK' if rel < 1e-9 else 'MISMATCH'}")
        res[name] = entry
        jsave(RES_B, res)
        print(f"[B] {name}: done {time.perf_counter() - t0:.1f}s")
        sys.stdout.flush()
    print("[B] complete")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _pct(xs, q):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def report():
    resA = jload(RES_A)
    resB = jload(RES_B)
    names = sorted(resA.keys())
    print(f"groups: {len(names)}")

    print("\n=== Table i: per-case score (pp) by test-T bucket ===")
    print(f"{'T':>6} {'refined':>9} {'unrefined':>10} {'n':>4} | "
          f"{'ref act/w mse rel':>18}")
    for T in (10, 128, 512, 1024):
        row = {"refined": [], "unrefined": []}
        actw = []
        for n in names:
            for v in ("refined", "unrefined"):
                for c in resA[n]["variants"][v]["cases"]:
                    if c["T"] == T:
                        row[v].append(c["score"] * 100)
                        if v == "refined":
                            actw.append(c["mse_act"] / max(c["mse_w"], 1e-30))
        print(f"{T:>6} {_mean(row['refined']):>9.2f} {_mean(row['unrefined']):>10.2f} "
              f"{len(row['refined']):>4} | act/w={_mean(actw):>8.2f}")

    print("\n=== Table ii: per-case score (pp) by (C, refined?) ===")
    print(f"{'C':>6} {'variant':>10} {'T10':>7} {'T128':>7} {'T512':>7} "
          f"{'T1024':>7} {'all':>7} {'cal_s':>6}")
    for C in CS:
        for v in ("refined", "unrefined"):
            per_T = {T: [] for T in (10, 128, 512, 1024)}
            cals = []
            for n in names:
                if resA[n]["C"] != C:
                    continue
                e = resA[n]["variants"][v]
                cals.append(e["cal_s"])
                for c in e["cases"]:
                    per_T[c["T"]].append(c["score"] * 100)
            allv = [s for ts in per_T.values() for s in ts]
            print(f"{C:>6} {v:>10} " + " ".join(f"{_mean(per_T[T]):>7.2f}" for T in per_T)
                  + f" {_mean(allv):>7.2f} {_mean(cals):>6.1f}")

    print("\n=== T=10 anchor stats (max(test_absmax)/calib_block_max, "
          "transformed space) ===")
    for C in CS:
        ratios = []
        for n in names:
            if resA[n]["C"] != C:
                continue
            ratios += resA[n]["anchor_T10"]["ratio"]
        moved = {b: _mean([resA[n]["anchor_T10"]["moved_frac"][b]
                           for n in names if resA[n]["C"] == C]) for b in map(str, BETAS)}
        print(f"C={C:>5}: mean {_mean(ratios):.3f} p10 {_pct(ratios, 0.1):.3f} "
              f"p50 {_pct(ratios, 0.5):.3f} p90 {_pct(ratios, 0.9):.3f} "
              f"frac<0.5 {sum(r < 0.5 for r in ratios) / len(ratios):.2f} "
              f"moved(b=.5/.7/1) {moved['0.5']:.2f}/{moved['0.7']:.2f}/{moved['1.0']:.2f}")

    if resB:
        print("\n=== Beta sweep: score delta (pp) vs refined control ===")
        print(f"{'beta':>6} {'dT10':>8} {'dT128':>8} {'dT512':>8} {'dT1024':>8} "
              f"{'worst':>8}")
        for b in sorted({k for n in resB for k in resB[n]["betas"]}, key=float):
            dTs = {T: [] for T in (10, 128, 512, 1024)}
            for n in names:
                if n not in resB or b not in resB[n]["betas"]:
                    continue
                ctrl = resA[n]["variants"]["refined"]["cases"]
                for c0, cb in zip(ctrl, resB[n]["betas"][b]):
                    dTs[c0["T"]].append((cb["score"] - c0["score"]) * 100)
            worst = min([d for ts in dTs.values() for d in ts] or [float("nan")])
            print(f"{b:>6} " + " ".join(f"{_mean(dTs[T]):>8.2f}" for T in dTs)
                  + f" {worst:>8.2f}")
        print("\nper-C dT10:")
        for C in CS:
            line = [f"C={C}:"]
            for b in sorted({k for n in resB for k in resB[n]["betas"]}, key=float):
                ds = []
                for n in names:
                    if n not in resB or resA[n]["C"] != C or b not in resB[n]["betas"]:
                        continue
                    ctrl = resA[n]["variants"]["refined"]["cases"]
                    ds += [(cb["score"] - c0["score"]) * 100
                           for c0, cb in zip(ctrl, resB[n]["betas"][b]) if c0["T"] == 10]
                line.append(f"b{b}={_mean(ds):+.2f}")
            print("  " + " ".join(line))


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "rep"
    kw = {"c_filter": None, "n_filter": None, "limit": None}
    betas = BETAS
    args = sys.argv[2:]
    for i, a in enumerate(args):
        if a == "--C":
            kw["c_filter"] = set(int(x) for x in args[i + 1].split(","))
        elif a == "--N":
            kw["n_filter"] = set(int(x) for x in args[i + 1].split(","))
        elif a == "--limit":
            kw["limit"] = int(args[i + 1])
        elif a == "--betas":
            betas = tuple(float(x) for x in args[i + 1].split(","))
    if mode == "A":
        run_A(kw["c_filter"], kw["n_filter"], kw["limit"])
    elif mode == "B":
        run_B(kw["c_filter"], kw["n_filter"], betas, kw["limit"])
    else:
        report()


if __name__ == "__main__":
    main()
