"""Sweep/rounds curves for the lattice refinement + E3 weight-sweep curve (v22).

Tasks (solution.py is read-only; a string-patched copy is exec'd per config,
the probe_sweeps.py pattern generalized):

  act  - re-score the cached 'refined' cal states (cache/*_refined.pt, valid:
         the v21->v22 diff touched only the dynamic-time n_sweeps line) at
         configs s5 / s8 / s12 / s5r40 for C in {512,1024,2048} (8 groups per
         C: N{1024,8192} x spread{0.5,0.9} x outlier{0,0.002}), all 5 test
         cases (T=10,128,512,1024,1024).  Per-call dynamic wall time recorded
         (this IS the online per-call price; judge = local/4.8).
         NOTE: in _refine_act_values the sweep and round loops run the SAME
         body, so only the product sweeps*rounds matters; s5r40 (200 iters)
         sits between s8 (160) and s12 (240) on the total-iterations curve.
  w    - full calibration with REFINE_W_SWEEPS in {1,2,3} at C in {512,1024}
         (E3 runs at calibration inside `if _e4:`; C<=2048 => always on).
         torch.manual_seed(0) per calibration, cal_s recorded.  w1 doubles as
         a parity check against results_A (T<=512 buckets, v22 == v21 there).

Usage:
  python dev/decomp/sweep2.py act --C 512 [--configs 5,8,12,5r40] [--limit k]
  python dev/decomp/sweep2.py w   --C 512 [--wsweeps 1,2,3] [--limit k]
  python dev/decomp/sweep2.py rep
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.dirname(HERE)
ROOT = os.path.dirname(DEV)
sys.path.insert(0, DEV)
import hif4 as H          # noqa: E402
import variants as V      # noqa: E402
import study as S         # noqa: E402  (iter_grid/make_group/jsave reuse)

RES_C = os.path.join(HERE, "results_C.json")
RES_D = os.path.join(HERE, "results_D.json")
SOL_PATH = os.path.join(ROOT, "example", "solution", "solution.py")

SHIP_SWEEPS_LINE = "n_sweeps = 5 if T <= 1024 else 0"
SHIP_ROUNDS_LINE = "REFINE_ROUNDS = 20"
SHIP_WSWEPS_LINE = "REFINE_W_SWEEPS = 1"

_MODS: dict[str, types.ModuleType] = {}


def load_patched(sweeps=None, rounds=None, w_sweeps=None):
    """Exec a string-patched copy of solution.py into a fresh module."""
    with open(SOL_PATH, encoding="utf-8") as f:
        src = f.read()
    subs = []
    if sweeps is not None:
        subs.append((SHIP_SWEEPS_LINE, f"n_sweeps = {int(sweeps)}"))
    if rounds is not None:
        subs.append((SHIP_ROUNDS_LINE, f"REFINE_ROUNDS = {int(rounds)}"))
    if w_sweeps is not None:
        subs.append((SHIP_WSWEPS_LINE, f"REFINE_W_SWEEPS = {int(w_sweeps)}"))
    for old, new in subs:
        if src.count(old) != 1:
            raise RuntimeError(f"patch target not unique: {old!r}")
        src = src.replace(old, new)
    mod = types.ModuleType("_sw2_sol")
    mod.__file__ = SOL_PATH
    exec(compile(src, SOL_PATH, "exec"), mod.__dict__)
    return mod


def patched_act(cfg: str):
    """cfg: '5', '8', '12' or '5r40' -> sweeps(, rounds)."""
    m = re.fullmatch(r"(\d+)(?:r(\d+))?", cfg)
    if not m:
        raise ValueError(cfg)
    key = f"s{cfg}"
    if key not in _MODS:
        _MODS[key] = load_patched(
            sweeps=int(m.group(1)),
            rounds=int(m.group(2)) if m.group(2) else None)
    return _MODS[key]


def patched_w(ws: int):
    key = f"w{ws}"
    if key not in _MODS:
        _MODS[key] = load_patched(w_sweeps=ws)
    return _MODS[key]


def score_case_mod(mod, pair, w_ref, w_std, weight_params, st,
                   refine_max_c=None):
    """study.score_case against a patched module; dt = dynamic-call wall time."""
    x_ref = H.dequantize_nvfp4(*pair)
    ref = H.linear_ref(x_ref, w_ref)
    x_std = V.deq(V.quant_alg1(x_ref.float()))
    mse_std = ((H.linear_ref(x_std, w_std) - ref) ** 2).mean().item()
    orig = mod.REFINE_MAX_C
    if refine_max_c is not None:
        mod.REFINE_MAX_C = refine_max_c
    try:
        t0 = time.perf_counter()
        p = mod.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
        dt = time.perf_counter() - t0
    finally:
        mod.REFINE_MAX_C = orig
    x_play = H.hif4_dequantize(p)
    w_play = H.hif4_dequantize(weight_params)
    mse_play = ((H.linear_ref(x_play, w_play) - ref) ** 2).mean().item()
    mse_act = ((H.linear_ref(x_play, w_ref) - ref) ** 2).mean().item()
    mse_w = ((H.linear_ref(x_ref, w_play) - ref) ** 2).mean().item()
    return {"T": int(pair[0].shape[0]), "dt": dt, "mse_std": mse_std,
            "mse_play": mse_play, "mse_act": mse_act, "mse_w": mse_w,
            "score": (mse_std - mse_play) / mse_std}


def build_group(name):
    gi = next(g for g in S.iter_grid(None, None) if g[0] == name)
    _, seed, C, N, spread, outp = gi
    group = S.make_group(seed, C, N, spread, outp)
    w_ref = H.dequantize_nvfp4(*group["weight"])
    w_std = V.deq(V.quant_alg1(w_ref.float()))
    return group, w_ref, w_std


# ---------------------------------------------------------------------------
def run_act(c_filter, configs, limit):
    res = S.jload(RES_C)
    res["act"] = res.get("act", {})
    grid = [g for g in S.iter_grid(c_filter, None, limit)
            if g[2] in (512, 1024, 2048)]
    print(f"[act] {len(grid)} groups, configs {configs}")
    for name, seed, C, N, spread, outp in grid:
        cpath = os.path.join(S.CACHE, f"{name}_refined.pt")
        if not os.path.exists(cpath):
            print(f"[act] {name}: no refined cache, SKIP")
            continue
        entry = res["act"].get(name, {})
        todo = [c for c in configs if c not in entry]
        if not todo:
            print(f"[act] {name}: cached, skip")
            continue
        t0 = time.perf_counter()
        group, w_ref, w_std = build_group(name)
        cal = torch.load(cpath, weights_only=True)["cal"]
        st, wp = cal["activation_state"], cal["weight_params"]
        for cfg in todo:
            mod = patched_act(cfg)
            cases = [score_case_mod(mod, p, w_ref, w_std, wp, st, 10 ** 9)
                     for p in group["test_activation_list"]]
            entry[cfg] = cases
            sc = [c["score"] * 100 for c in cases]
            dt = [c["dt"] for c in cases]
            print(f"[act] {name} s{cfg}: pp {['%.1f' % s for s in sc]} "
                  f"dt {['%.2f' % d for d in dt]} "
                  f"({time.perf_counter() - t0:.1f}s)")
            sys.stdout.flush()
        res["act"][name] = entry
        S.jsave(RES_C, res)
    S.jsave(RES_C, res)
    print("[act] complete")


# ---------------------------------------------------------------------------
def run_w(c_filter, wsweeps, limit):
    res = S.jload(RES_D)
    resA = S.jload(S.RES_A)
    grid = [g for g in S.iter_grid(c_filter, None, limit)]
    print(f"[w] {len(grid)} groups, w_sweeps {wsweeps}")
    for name, seed, C, N, spread, outp in grid:
        entry = res.get(name, {})
        todo = [w for w in wsweeps if f"w{w}" not in entry]
        if not todo:
            print(f"[w] {name}: cached, skip")
            continue
        t0 = time.perf_counter()
        group, w_ref, w_std = build_group(name)
        for ws in todo:
            mod = patched_w(ws)
            torch.manual_seed(0)
            tc = time.perf_counter()
            cal = mod.hif4_calibration_and_quantize_weight(
                group["weight"][0], group["weight"][1],
                group["calib_activation_list"])
            cal_s = time.perf_counter() - tc
            st, wp = cal["activation_state"], cal["weight_params"]
            # ship REFINE_MAX_C (=2048): C<=2048 gate == the 1e9 forced one
            cases = [score_case_mod(mod, p, w_ref, w_std, wp, st)
                     for p in group["test_activation_list"]]
            entry[f"w{ws}"] = {"cal_s": cal_s, "cases": cases}
            sc = [c["score"] * 100 for c in cases]
            print(f"[w] {name} w{ws}: cal {cal_s:.1f}s pp "
                  f"{['%.1f' % s for s in sc]} ({time.perf_counter() - t0:.1f}s)")
            sys.stdout.flush()
        # parity: w1 vs results_A refined (v21 file scored T<=512 with the
        # same dynamic path; T=1024 differs by the v22 sweep equalization)
        if "w1" in entry and "parity_checked" not in entry \
                and name in resA and "refined" in resA[name].get("variants", {}):
            ref = resA[name]["variants"]["refined"]["cases"]
            rel = 0.0
            for c0, c1 in zip(ref, entry["w1"]["cases"]):
                if c0["T"] <= 512:
                    rel = max(rel, abs(c1["mse_play"] - c0["mse_play"])
                              / max(c0["mse_play"], 1e-30))
            entry["parity_checked"] = {"max_rel_T<=512": rel, "ok": rel < 1e-9}
            print(f"[w] {name}: w1 vs results_A parity rel={rel:.2e} "
                  f"{'OK' if rel < 1e-9 else 'MISMATCH'}")
        res[name] = entry
        S.jsave(RES_D, res)
        sys.stdout.flush()
    print("[w] complete")


# ---------------------------------------------------------------------------
def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _fmt(x):
    return f"{x:>8.2f}" if x == x else "     ---"


def rep():
    resC = S.jload(RES_C)
    resD = S.jload(RES_D)
    act = resC.get("act", {})
    names = sorted(act.keys())
    cfgs = [c for c in ("5", "8", "5r40", "12") if any(c in act[n] for n in names)]
    if names:
        print(f"=== ACT sweep curve: score pp by (C, T) x config ===")
        print(f"{'C':>6} {'T':>5} {'n':>3} " + " ".join(f"{'s'+c:>8}" for c in cfgs)
              + "   " + " ".join(f"{m:>10}" for m in
                                 ("d(5->8)/3", "d(8->12)/4", "d5->12")))
        for C in (512, 1024, 2048):
            for T in (10, 128, 512, 1024):
                row = []
                for c in cfgs:
                    row.append([x["score"] * 100 for n in names
                                if _C_of(n, act) == C
                                for x in act[n].get(c, []) if x["T"] == T])
                line = f"{C:>6} {T:>5} {len(row[0]) if row else 0:>3} "
                line += " ".join(_fmt(_mean(r)) for r in row)
                m = []
                for a, b, div in (("5", "8", 3), ("8", "12", 4)):
                    ia = cfgs.index(a) if a in cfgs else None
                    ib = cfgs.index(b) if b in cfgs else None
                    m.append((_mean(row[ib]) - _mean(row[ia])) / div
                             if ia is not None and ib is not None else float("nan"))
                m.append(_mean(row[cfgs.index("12")]) - _mean(row[cfgs.index("5")])
                         if "5" in cfgs and "12" in cfgs else float("nan"))
                line += "   " + " ".join(f"{x:>+10.3f}" if x == x else "       ---"
                                         for x in m)
                print(line)
        print("\n=== ACT per-call dynamic wall time (s), mean over cases ===")
        print(f"{'C':>6} {'T':>5} " + " ".join(f"{'s'+c:>8}" for c in cfgs))
        for C in (512, 1024, 2048):
            for T in (10, 128, 512, 1024):
                row = []
                for c in cfgs:
                    row.append([x["dt"] for n in names
                                for x in act[n].get(c, [])
                                if x["T"] == T and _C_of(n, act) == C])
                if any(row):
                    print(f"{C:>6} {T:>5} " + " ".join(f"{_mean(r):>8.2f}" for r in row))
    if resD:
        names_w = sorted(resD.keys())
        print("\n=== W sweep curve: score pp by (C, T) x w_sweeps ===")
        print(f"{'C':>6} {'T':>5} " + " ".join(f"{('w'+w):>8}" for w in ("1", "2", "3")
                                               if any(f"w{w}" in resD[n] for n in names_w)))
        for C in (512, 1024):
            for T in (10, 128, 512, 1024):
                row = []
                for w in ("1", "2", "3"):
                    row.append([x["score"] * 100 for n in names_w
                                for x in resD[n].get(f"w{w}", {}).get("cases", [])
                                if x["T"] == T and _C_of(n, resD) == C])
                if any(row):
                    print(f"{C:>6} {T:>5} " + " ".join(_fmt(_mean(r)) for r in row))
            allr = []
            for w in ("1", "2", "3"):
                allr.append([x["score"] * 100 for n in names_w
                             for x in resD[n].get(f"w{w}", {}).get("cases", [])
                             if _C_of(n, resD) == C])
            if any(allr):
                print(f"{C:>6} {'all':>5} " + " ".join(_fmt(_mean(r)) for r in allr))
        print("\ncal_s mean by (C, w):")
        for C in (512, 1024):
            line = [f"C={C}:"]
            for w in ("1", "2", "3"):
                cs = [resD[n][f"w{w}"]["cal_s"] for n in names_w
                      if f"w{w}" in resD[n] and _C_of(n, resD) == C]
                line.append(f"w{w}={_mean(cs):.1f}s" if cs else f"w{w}=---")
            print("  " + " ".join(line))
        pars = [resD[n].get("parity_checked", {}).get("max_rel_T<=512")
                for n in names_w if resD[n].get("parity_checked")]
        if pars:
            print(f"w1 vs results_A parity (T<=512 mse): max rel diff "
                  f"{max(pars):.2e}")


def _C_of(name, res):
    if "C" in res.get(name, {}):
        return res[name]["C"]
    gi = next(g for g in S.iter_grid(None, None) if g[0] == name)
    return gi[2]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "rep"
    c_filter = None
    limit = None
    configs = ("5", "8", "12", "5r40")
    wsweeps = (1, 2, 3)
    args = sys.argv[2:]
    for i, a in enumerate(args):
        if a == "--C":
            c_filter = set(int(x) for x in args[i + 1].split(","))
        elif a == "--limit":
            limit = int(args[i + 1])
        elif a == "--configs":
            configs = tuple(args[i + 1].split(","))
        elif a == "--wsweeps":
            wsweeps = tuple(int(x) for x in args[i + 1].split(","))
    if mode == "act":
        run_act(c_filter, configs, limit)
    elif mode == "w":
        run_w(c_filter, wsweeps, limit)
    else:
        rep()


if __name__ == "__main__":
    main()
