"""decomp2: fresh residual decomposition on the CURRENT solution (v29, tiers
24/12/5, hash-gated grams C<=4096, REFINE_T_MAX=1024, E3 weight refine).

solution.py is NEVER modified.  Conventions copied from dev/decomp/study.py:
same 32-group grid (seeds via iter_grid order), same scoring (exact paper
Alg.1 baseline via variants.quant_alg1), same mse_act/mse_w attribution.

Subcommands:
  pop    - task 1 + 2a + 2c: ship-config calibration + scoring of the 32
           groups, plus forced-_e4 variants at C=4096 (nohash / all4096),
           plus 8 extra C=8192 groups (never refined on ship).
  t2048  - task 2b: T=2048 test calls scored with the ship gate (R<=1024 =>
           no refinement) and a REFINE_T_MAX=4096 patched module, against
           the paired T=1024 call.  Reuses cached ship cal states; the group
           generator appends the 2048 draw after the 9 standard draws, so
           calib + first 5 test cases are bit-identical to `pop`.
  rep    - tables.

Usage: python dev/decomp2/study2.py pop [--C 512,1024] [--limit k]
       python dev/decomp2/study2.py t2048 [--C 512,1024,2048]
       python dev/decomp2/study2.py rep
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
RES = os.path.join(HERE, "results_pop.json")
RES_T2K = os.path.join(HERE, "results_t2048.json")
SOL_PATH = os.path.join(ROOT, "example", "solution", "solution.py")

CALIB_T = (10, 128, 512, 1024)
TEST_T = (10, 128, 512, 1024, 1024)
TEST_T2 = (10, 128, 512, 1024, 1024, 2048)
CS2 = (512, 1024, 2048, 4096, 8192)
NS = (1024, 8192)
SPREADS = (0.5, 0.9)
OUTLIERS = (0.0, 0.002)

# exact ship lines (verify count==1 before replace)
_E4_SHIP = ("    _e4 = (C <= REFINE_MAX_C\n"
            "           or (C <= 4096 and int(w.double().abs().sum().item() * 1e3) % 2 == 0))")
_TMAX_SHIP = "REFINE_T_MAX = 1024"
_SWEEPS_SHIP = "n_sweeps = 24 if T <= 256 else 12 if T <= 512 else 5"


def load_patched(e4=None, t_max=None, sweeps=None):
    with open(SOL_PATH, encoding="utf-8") as f:
        src = f.read()
    subs = []
    if e4 == "nohash":
        subs.append((_E4_SHIP, "    _e4 = (C <= REFINE_MAX_C)"))
    elif e4 == "all4096":
        subs.append((_E4_SHIP, "    _e4 = (C <= REFINE_MAX_C or C <= 4096)"))
    elif e4:
        raise ValueError(e4)
    if t_max is not None:
        subs.append((_TMAX_SHIP, f"REFINE_T_MAX = {int(t_max)}"))
    if sweeps is not None:
        subs.append((_SWEEPS_SHIP,
                     f"n_sweeps = {int(sweeps)} if T <= 256 else 12 if T <= 512 else 5"))
    for old, new in subs:
        if src.count(old) != 1:
            raise RuntimeError(f"patch target not unique ({src.count(old)}): {old!r}")
        src = src.replace(old, new)
    mod = types.ModuleType("_d2_sol")
    mod.__file__ = SOL_PATH
    exec(compile(src, SOL_PATH, "exec"), mod.__dict__)
    return mod


_SOL = None


def sol():
    global _SOL
    if _SOL is None:
        _SOL = load_patched()
    return _SOL


_MODS: dict[str, types.ModuleType] = {}


def mod_for(tag):
    if tag not in _MODS:
        if tag == "ship":
            _MODS[tag] = sol()
        elif tag == "nohash":
            _MODS[tag] = load_patched(e4="nohash")
        elif tag == "all4096":
            _MODS[tag] = load_patched(e4="all4096")
        elif tag == "tmax4096":
            _MODS[tag] = load_patched(t_max=4096)
        else:
            raise ValueError(tag)
    return _MODS[tag]


# ---------------------------------------------------------------------------
# group construction (seeds identical to dev/decomp for the first 32 groups;
# C=8192 continues the same enumeration -> seeds 4200+13*i)
# ---------------------------------------------------------------------------
def iter_grid(c_filter=None, limit=None):
    out = []
    i = 0
    for C in CS2:
        for N in NS:
            for spread in SPREADS:
                for outp in OUTLIERS:
                    name = f"c{C}_n{N}_s{spread}_o{outp}"
                    if c_filter is None or C in c_filter:
                        out.append((name, 4200 + 13 * i, C, N, spread, outp))
                    i += 1
    return out[:limit] if limit else out


def make_group(seed, C, N, spread, outlier_p, test_t=TEST_T):
    tokens = CALIB_T + tuple(test_t)
    g = synth.make_linear_group(seed, N, C, tokens=tokens,
                                spread=spread, outlier_p=outlier_p)
    nc = len(CALIB_T)
    return {
        "weight": g["weight"],
        "calib_activation_list": g["calib_activation_list"][:nc],
        "test_activation_list": g["test_activation_list"][nc:],
    }


def build_group(name, test_t=TEST_T):
    _, seed, C, N, spread, outp = next(g for g in iter_grid() if g[0] == name)
    return make_group(seed, C, N, spread, outp, test_t)


def jload(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def jsave(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def calibrate(name, group, tag):
    """Calibrate with the module for `tag`; cache cal state + wall time."""
    cpath = os.path.join(CACHE, f"{name}_{tag}.pt")
    if os.path.exists(cpath):
        return torch.load(cpath, weights_only=True)
    mod = mod_for(tag)
    torch.manual_seed(0)
    t0 = time.perf_counter()
    cal = mod.hif4_calibration_and_quantize_weight(
        group["weight"][0], group["weight"][1],
        group["calib_activation_list"])
    cal_s = time.perf_counter() - t0
    torch.save({"cal": cal, "cal_s": cal_s}, cpath)
    return {"cal": cal, "cal_s": cal_s}


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


def score_group(mod, group, w_ref, w_std, weight_params, st, keep=TEST_T):
    out = []
    for p in group["test_activation_list"]:
        if p[0].shape[0] not in keep:
            continue
        out.append(score_case(mod, p, w_ref, w_std, weight_params, st))
    return out


# ---------------------------------------------------------------------------
# pop (task 1 + 2a + 2c)
# ---------------------------------------------------------------------------
def run_pop(c_filter, limit):
    res = jload(RES)
    grid = iter_grid(c_filter, limit)
    print(f"[pop] {len(grid)} groups")
    for name, seed, C, N, spread, outp in grid:
        entry = res.get(name, {})
        tags = ["ship"] + (["nohash", "all4096"] if C == 4096 else [])
        todo = [t for t in tags if t not in entry]
        if not todo:
            print(f"[pop] {name}: cached, skip")
            continue
        t0 = time.perf_counter()
        group = make_group(seed, C, N, spread, outp)
        w_ref = H.dequantize_nvfp4(*group["weight"])
        parity = int(w_ref.float().double().abs().sum().item() * 1e3) % 2
        w_std = V.deq(V.quant_alg1(w_ref.float()))
        entry.update({"C": C, "N": N, "spread": spread, "outlier_p": outp,
                      "parity_even": parity == 0})
        for tag in todo:
            cc = calibrate(name, group, tag)
            st = cc["cal"]["activation_state"]
            wp = cc["cal"]["weight_params"]
            cases = score_group(mod_for(tag), group, w_ref, w_std, wp, st)
            entry[tag] = {"cal_s": cc["cal_s"], "cases": cases}
            sc = [c["score"] * 100 for c in cases]
            print(f"[pop] {name} {tag}: cal {cc['cal_s']:.1f}s pp "
                  f"{['%.1f' % s for s in sc]} ({time.perf_counter()-t0:.1f}s)")
            sys.stdout.flush()
        res[name] = entry
        jsave(RES, res)
    print("[pop] complete")


# ---------------------------------------------------------------------------
# t2048 (task 2b)
# ---------------------------------------------------------------------------
def run_t2048(c_filter):
    res = jload(RES)
    res2 = jload(RES_T2K)
    grid = [g for g in iter_grid(c_filter) if g[2] <= 4096]
    print(f"[t2048] {len(grid)} groups")
    checked = set()
    for name, seed, C, N, spread, outp in grid:
        cpath = os.path.join(CACHE, f"{name}_ship.pt")
        if not os.path.exists(cpath):
            print(f"[t2048] {name}: no ship cache, SKIP")
            continue
        entry = res2.get(name, {})
        if "ship" in entry and "tmax4096" in entry:
            print(f"[t2048] {name}: cached, skip")
            continue
        t0 = time.perf_counter()
        group = build_group(name, test_t=TEST_T2)
        # sanity: calib tensors bit-identical to the pop group (draw order)
        if name not in checked:
            g0 = build_group(name)
            for a, b in zip(group["calib_activation_list"],
                            g0["calib_activation_list"]):
                assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            for i in (0, 1, 2, 3, 4):
                a = group["test_activation_list"][i]
                b = g0["test_activation_list"][i]
                assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            checked.add(name)
            print(f"[t2048] {name}: calib/test prefix bit-identical to pop OK")
        w_ref = H.dequantize_nvfp4(*group["weight"])
        w_std = V.deq(V.quant_alg1(w_ref.float()))
        cc = torch.load(cpath, weights_only=True)
        st = cc["cal"]["activation_state"]
        wp = cc["cal"]["weight_params"]
        for tag in ("ship", "tmax4096"):
            mod = mod_for(tag)
            # only the last T=1024 case (pairing) and the T=2048 case
            cases = [score_case(mod, p, w_ref, w_std, wp, st)
                     for p in group["test_activation_list"][3:]]
            entry[tag] = cases
            sc = [c["score"] * 100 for c in cases]
            print(f"[t2048] {name} {tag}: pp {['%.1f' % s for s in sc]} "
                  f"dt {['%.2f' % c['dt'] for c in cases]} "
                  f"({time.perf_counter()-t0:.1f}s)")
            sys.stdout.flush()
        res2[name] = entry
        jsave(RES_T2K, res2)
    print("[t2048] complete")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _fmt(x):
    return f"{x:>8.2f}" if x == x else "     ---"


def rep():
    res = jload(RES)
    names = sorted(res.keys())
    print(f"groups: {len(names)}")
    refined = [n for n in names
               if res[n]["C"] <= 2048
               or (res[n]["C"] == 4096 and res[n]["parity_even"])]

    print("\n=== Table i: ship per-case score (pp) by test-T bucket ===")
    print("(all = every ship group; refined = grams carried (C<=2048 or even 4096))")
    for T in (10, 128, 512, 1024):
        row = {"all": [], "refined": []}
        actw = []
        for n in names:
            for c in res[n]["ship"]["cases"]:
                if c["T"] == T:
                    row["all"].append(c["score"] * 100)
                    if n in refined:
                        row["refined"].append(c["score"] * 100)
                        actw.append(c["mse_act"] / max(c["mse_w"], 1e-30))
        print(f"T={T:>5}: all {_mean(row['all']):>7.2f} "
              f"refined {_mean(row['refined']):>7.2f} (n={len(row['refined'])}) "
              f"act/w={_mean(actw):>6.2f}")

    print("\n=== Table ii: ship score (pp) by (C, T) + mean dynamic dt (s) ===")
    print(f"{'C':>6} {'grp':>4} " + " ".join(f"{('T'+str(T)):>12}" for T in (10, 128, 512, 1024))
          + f" {'all':>8} {'dt/bl':>6} {'act/w':>6}")
    for C in CS2:
        per_T = {T: [] for T in (10, 128, 512, 1024)}
        dts, actw = [], []
        for n in names:
            if res[n]["C"] != C:
                continue
            for c in res[n]["ship"]["cases"]:
                per_T[c["T"]].append(c["score"] * 100)
                dts.append(c["dt"])
                actw.append(c["mse_act"] / max(c["mse_w"], 1e-30))
        allv = [s for ts in per_T.values() for s in ts]
        if allv:
            print(f"{C:>6} {len(per_T[10]):>4} "
                  + " ".join(f"{_mean(per_T[T]):>12.2f}" for T in per_T)
                  + f" {_mean(allv):>8.2f} {_mean(dts):>6.2f} {_mean(actw):>6.2f}")

    print("\n=== 2a: C=4096 forced-_e4 isolation (score pp by T) ===")
    n4096 = [n for n in names if res[n]["C"] == 4096]
    print(f"{'variant':>10} " + " ".join(f"{('T'+str(T)):>8}" for T in (10, 128, 512, 1024)))
    for tag in ("all4096", "ship", "nohash"):
        per_T = {T: [] for T in (10, 128, 512, 1024)}
        for n in n4096:
            for c in res[n].get(tag, {}).get("cases", []):
                per_T[c["T"]].append(c["score"] * 100)
        if per_T[10]:
            print(f"{tag:>10} " + " ".join(f"{_mean(per_T[T]):>8.2f}" for T in per_T))
    per_T = {T: [] for T in (10, 128, 512, 1024)}
    for n in n4096:
        for c in res[n]["all4096"]["cases"]:
            per_T[c["T"]].append((c["score"]
                                  - next(x["score"] for x in res[n]["nohash"]["cases"]
                                         if x["T"] == c["T"])) * 100)
    print(f"{'gap a-n':>10} " + " ".join(f"{_mean(per_T[T]):>+8.2f}" for T in per_T))
    for n in sorted(n4096):
        print(f"  {n}: parity {'even' if res[n]['parity_even'] else 'odd'}")

    res2 = jload(RES_T2K)
    if res2:
        print("\n=== 2b: T=2048 vs paired T=1024 (score pp; ship gate vs refine) ===")
        for C in (512, 1024, 2048, 4096):
            rows = {"ship_1024": [], "ship_2048": [], "r_2048": [],
                    "d": []}
            dts = []
            for n, e in sorted(res2.items()):
                if next(g for g in iter_grid() if g[0] == n)[2] != C:
                    continue
                for c0, c1 in zip(e["ship"], e["tmax4096"]):
                    if c0["T"] == 1024:
                        rows["ship_1024"].append(c0["score"] * 100)
                    else:
                        rows["ship_2048"].append(c0["score"] * 100)
                        rows["r_2048"].append(c1["score"] * 100)
                        rows["d"].append((c1["score"] - c0["score"]) * 100)
                        dts.append((c1["dt"], c0["dt"]))
            if rows["ship_1024"]:
                print(f"C={C:>5}: T1024 {_mean(rows['ship_1024']):>7.2f} "
                      f"T2048ship {_mean(rows['ship_2048']):>7.2f} "
                      f"T2048ref {_mean(rows['r_2048']):>7.2f} "
                      f"d(ref-ship) {_mean(rows['d']):>+7.2f} "
                      f"dt ship/ref {_mean([a for _, a in dts]):.2f}/"
                      f"{_mean([b for b, _ in dts]):.2f}s")
        print("\nT=2048 vs T=1024 same-config mse ratio (ship), and T=1024 refined-vs-not:")
        for C in (512, 1024, 2048, 4096):
            r = []
            for n, e in sorted(res2.items()):
                if next(g for g in iter_grid() if g[0] == n)[2] != C:
                    continue
                c1024 = next(c for c in e["ship"] if c["T"] == 1024)
                c2048 = next(c for c in e["ship"] if c["T"] == 2048)
                r.append(c2048["mse_play"] / max(c1024["mse_play"], 1e-30))
            if r:
                print(f"  C={C}: mse(T2048)/mse(T1024) mean {_mean(r):.2f}")

    print("\n=== 2c: C=8192 (never refined) vs C=2048 (always refined) ===")
    for C in (2048, 8192):
        per_T = {T: [] for T in (10, 128, 512, 1024)}
        actw = []
        for n in names:
            if res[n]["C"] != C:
                continue
            for c in res[n]["ship"]["cases"]:
                per_T[c["T"]].append(c["score"] * 100)
                actw.append(c["mse_act"] / max(c["mse_w"], 1e-30))
        allv = [s for ts in per_T.values() for s in ts]
        print(f"C={C}: T10 {_mean(per_T[10]):.2f} T128 {_mean(per_T[128]):.2f} "
              f"T512 {_mean(per_T[512]):.2f} T1024 {_mean(per_T[1024]):.2f} "
              f"all {_mean(allv):.2f} act/w {_mean(actw):.2f} n={len(per_T[10])}")


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
    if mode == "pop":
        run_pop(c_filter, limit)
    elif mode == "t2048":
        run_t2048(c_filter)
    else:
        rep()


if __name__ == "__main__":
    main()
