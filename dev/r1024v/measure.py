"""r1024v measurement harness: mechanisms A (R>1024 prefix refinement) and
B (per-call exact variant selection) on the dev copy dev/r1024v/solution.py.

Scoring conventions copied from dev/decomp2/study2.py (exact paper Alg.1
baseline via variants.quant_alg1; same 40-group enumeration seeds so the
C2048/C4096 groups match decomp2's).

Subcommands:
  mini  - mechanism B on the official mini group (all 5 tests x B0/B2/B3)
  a     - mechanism A on synthetic groups C{2048,4096} N=8192 (4 seeds each),
          extra test draws T{2048,4096}, configs A0/A8/A20/Full
  b     - mechanism B on synthetic groups (5 shapes incl. one C4096) with
          standard tests + one T=2048 draw, configs B0/B2/B3
  rep   - tables from results/*.json

Usage: python dev/r1024v/measure.py {mini|a|b|rep}
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
import hif4 as H          # noqa: E402
import synth              # noqa: E402
import variants as V      # noqa: E402

SOL_PATH = os.path.join(HERE, "solution.py")
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

_spec = None


def sol():
    """Single shared module instance; switches are set per config at runtime."""
    global _spec
    if _spec is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_r1024v_sol", SOL_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _spec = mod
    return _spec


SWITCHES = ("PREFIX_REFINE", "PREFIX_SWEEPS", "DYN_TMAX_OVERRIDE", "DYN_SELECT")
DEFAULTS = dict(PREFIX_REFINE=False, PREFIX_SWEEPS=0,
                DYN_TMAX_OVERRIDE=None, DYN_SELECT=0)


def apply_cfg(cfg: dict):
    m = sol()
    for k in SWITCHES:
        setattr(m, k, DEFAULTS[k])
    for k, v in cfg.items():
        setattr(m, k, v)
    return m


# --- group construction (seeds identical to decomp2 iter_grid) --------------
CS2 = (512, 1024, 2048, 4096, 8192)
NS = (1024, 8192)
SPREADS = (0.5, 0.9)
OUTLIERS = (0.0, 0.002)
CALIB_T = (10, 128, 512, 1024)
TEST_T = (10, 128, 512, 1024, 1024)


def iter_grid():
    out = []
    i = 0
    for C in CS2:
        for N in NS:
            for spread in SPREADS:
                for outp in OUTLIERS:
                    out.append((f"c{C}_n{N}_s{spread}_o{outp}",
                                4200 + 13 * i, C, N, spread, outp))
                    i += 1
    return out


def grid_by_name(name):
    return next(g for g in iter_grid() if g[0] == name)


def make_group(name, test_t=TEST_T):
    _, seed, C, N, spread, outp = grid_by_name(name)
    tokens = CALIB_T + tuple(test_t)
    g = synth.make_linear_group(seed, N, C, tokens=tokens,
                                spread=spread, outlier_p=outp)
    nc = len(CALIB_T)
    return {
        "weight": g["weight"],
        "calib_activation_list": g["calib_activation_list"][:nc],
        "test_activation_list": g["test_activation_list"][nc:],
    }


def make_extra_act(seed, C, spread, outlier_p, T):
    """Independent seeded T-draw (statistically one more make_act_pair draw;
    keeps the cached cal state valid). Copied from decomp2.study2."""
    gen = torch.Generator().manual_seed(seed + 9779)
    x = (torch.randn(T, 1, generator=gen)
         * synth._chan_gains(C, spread, gen).unsqueeze(0)
         * torch.randn(T, C, generator=gen))
    if outlier_p > 0:
        mask = torch.rand(T, C, generator=gen) < outlier_p
        x = x + mask.float() * torch.randn(T, C, generator=gen) * x.abs().amax() * 3
    xb = x.reshape(-1, 16)
    amax = xb.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    scale = ((amax / 6.0).to(torch.bfloat16).float()).clamp_min(1e-30)
    q = xb / scale
    idx = torch.bucketize(q.abs(), (synth.E2M1_GRID[1:] + synth.E2M1_GRID[:-1]) / 2.0)
    carrier = (torch.sign(q) * synth.E2M1_GRID[idx]).reshape(T, -1).to(torch.bfloat16)
    return carrier, scale.reshape(T, -1).to(torch.bfloat16)


CALDIR = os.path.join(HERE, "calcache")
os.makedirs(CALDIR, exist_ok=True)


def calibrate(name, group):
    """Ship calibration (mechanisms A/B never touch it); cached on disk."""
    cpath = os.path.join(CALDIR, f"{name}.pt")
    if os.path.exists(cpath):
        return torch.load(cpath, weights_only=True)
    m = apply_cfg({})
    torch.manual_seed(0)
    t0 = time.perf_counter()
    cal = m.hif4_calibration_and_quantize_weight(
        group["weight"][0], group["weight"][1], group["calib_activation_list"])
    cal_s = time.perf_counter() - t0
    torch.save({"cal": cal, "cal_s": cal_s}, cpath)
    return {"cal": cal, "cal_s": cal_s}


def jload(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def jsave(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def score_case(mod, pair, w_ref, w_std, weight_params, st, reps=1, dbg_list=None):
    """Run one dynamic call, score vs exact Alg.1, time as min(reps)."""
    x_ref = H.dequantize_nvfp4(*pair)
    ref = H.linear_ref(x_ref, w_ref)
    x_std = V.deq(V.quant_alg1(x_ref.float()))
    mse_std = ((H.linear_ref(x_std, w_std) - ref) ** 2).mean().item()
    w_play = H.hif4_dequantize(weight_params)
    if dbg_list is not None:
        mod.DYN_DEBUG = dbg_list
    else:
        mod.DYN_DEBUG = None
    dt = None
    for _ in range(reps):
        t0 = time.perf_counter()
        p = mod.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
        el = time.perf_counter() - t0
        dt = el if dt is None else min(dt, el)
    mod.DYN_DEBUG = None
    x_play = H.hif4_dequantize(p)
    mse_play = ((H.linear_ref(x_play, w_play) - ref) ** 2).mean().item()
    return {"T": int(pair[0].shape[0]), "dt": dt, "mse_std": mse_std,
            "mse_play": mse_play, "score": (mse_std - mse_play) / mse_std}


def reps_for(pair):
    T = pair[0].shape[0]
    C = pair[0].shape[1]
    return 3 if T * C <= 4_000_000 else 2


# ---------------------------------------------------------------------------
# mini (mechanism B on official data)
# ---------------------------------------------------------------------------
def run_mini():
    out = jload(os.path.join(RES, "mini.json"))
    lin = torch.load(os.path.join(ROOT, "example", "mini_sample", "linear.pt"),
                     weights_only=True, map_location="cpu")[0]
    group = {"weight": lin["weight"],
             "calib_activation_list": lin["calib_activation_list"],
             "test_activation_list": lin["test_activation_list"]}
    cc = calibrate("mini", group)
    st = cc["cal"]["activation_state"]
    wp = cc["cal"]["weight_params"]
    w_ref = H.dequantize_nvfp4(*group["weight"])
    w_std = V.deq(V.quant_alg1(w_ref.float()))
    for tag, cfg in (("B0", {}), ("B2", {"DYN_SELECT": 2}), ("B3", {"DYN_SELECT": 3})):
        if tag in out and len(out[tag]["cases"]) == len(group["test_activation_list"]):
            continue
        mod = apply_cfg(cfg)
        dbg_all = []
        cases = []
        for pair in group["test_activation_list"]:
            # scoring+debug pass (dbg pollutes timing with fp64 scoring)
            score_case(mod, pair, w_ref, w_std, wp, st, reps=1, dbg_list=dbg_all)
            # clean timing pass
            c = score_case(mod, pair, w_ref, w_std, wp, st, reps=3)
            cases.append(c)
        out[tag] = {"cases": cases, "dbg": dbg_all}
        sc = [f"{c['score']*100:+.2f}" for c in cases]
        dt = [f"{c['dt']:.2f}" for c in cases]
        sels = [d.get("sel") for d in dbg_all]
        print(f"[mini] {tag}: pp {sc} dt {dt} sel {sels}")
        sys.stdout.flush()
        jsave(os.path.join(RES, "mini.json"), out)
    print("[mini] done")


# ---------------------------------------------------------------------------
# mechanism A: synthetic C{2048,4096} N8192, extra T{2048,4096}
# ---------------------------------------------------------------------------
A_GROUPS = [g for g in iter_grid()
            if g[2] in (2048, 4096) and g[3] == 8192]
A_CFGS = (
    ("A0", {}),
    ("A8", {"PREFIX_REFINE": True, "PREFIX_SWEEPS": 0}),
    ("A20", {"PREFIX_REFINE": True, "PREFIX_SWEEPS": 20}),
    ("Full", {"DYN_TMAX_OVERRIDE": 8192}),
)


def run_a():
    out = jload(os.path.join(RES, "a.json"))
    for name, seed, C, N, spread, outp in A_GROUPS:
        entry = out.get(name, {})
        if "cases" in entry:
            print(f"[a] {name}: cached, skip")
            continue
        group = make_group(name)
        cc = calibrate(name, group)
        st = cc["cal"]["activation_state"]
        wp = cc["cal"]["weight_params"]
        w_ref = H.dequantize_nvfp4(*group["weight"])
        w_std = V.deq(V.quant_alg1(w_ref.float()))
        grams = int(st.get("gw") is not None)
        tmax = int(st.get("tmax") or 1024)
        cases = {}
        for T in (2048, 4096):
            pair = make_extra_act(seed, C, spread, outp, T)
            for tag, cfg in A_CFGS:
                mod = apply_cfg(cfg)
                c = score_case(mod, pair, w_ref, w_std, wp, st, reps=reps_for(pair))
                cases[f"T{T}_{tag}"] = c
                print(f"[a] {name} T{T} {tag}: pp {c['score']*100:+.2f} "
                      f"dt {c['dt']:.2f}s")
                sys.stdout.flush()
        out[name] = {"C": C, "N": N, "spread": spread, "outlier_p": outp,
                     "grams": grams, "tmax": tmax, "cases": cases}
        jsave(os.path.join(RES, "a.json"), out)
    print("[a] done")


# ---------------------------------------------------------------------------
# mechanism B: synthetic shapes, standard tests + T=2048
# ---------------------------------------------------------------------------
B_GROUPS = [
    "c512_n8192_s0.5_o0.0",
    "c1024_n8192_s0.9_o0.002",
    "c2048_n8192_s0.5_o0.0",
    "c2048_n8192_s0.9_o0.002",
    "c4096_n8192_s0.5_o0.0",
]
B_CFGS = (("B0", {}), ("B2", {"DYN_SELECT": 2}), ("B3", {"DYN_SELECT": 3}))


def run_b():
    out = jload(os.path.join(RES, "b.json"))
    for name in B_GROUPS:
        entry = out.get(name, {})
        if "B0" in entry and "B2" in entry and "B3" in entry:
            print(f"[b] {name}: cached, skip")
            continue
        _, seed, C, N, spread, outp = grid_by_name(name)
        group = make_group(name)
        cc = calibrate(name, group)
        st = cc["cal"]["activation_state"]
        wp = cc["cal"]["weight_params"]
        w_ref = H.dequantize_nvfp4(*group["weight"])
        w_std = V.deq(V.quant_alg1(w_ref.float()))
        grams = int(st.get("gw") is not None)
        gflag = int(st.get("g") or 0)
        pairs = list(group["test_activation_list"]) + [
            make_extra_act(seed, C, spread, outp, 2048)]
        for tag, cfg in B_CFGS:
            mod = apply_cfg(cfg)
            dbg_all = []
            cases = []
            for pair in pairs:
                score_case(mod, pair, w_ref, w_std, wp, st, reps=1, dbg_list=dbg_all)
                c = score_case(mod, pair, w_ref, w_std, wp, st, reps=reps_for(pair))
                cases.append(c)
            entry[tag] = {"cases": cases, "dbg": dbg_all}
            sc = [f"{c['score']*100:+.2f}" for c in cases]
            sels = [d.get("sel") or d.get("default") for d in dbg_all]
            print(f"[b] {name} {tag}: pp {sc} sel {sels}")
            sys.stdout.flush()
            entry["_meta"] = {"C": C, "grams": grams, "g": gflag}
            out[name] = entry
            jsave(os.path.join(RES, "b.json"), out)
    print("[b] done")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def rep():
    a = jload(os.path.join(RES, "a.json"))
    if a:
        print("=== Mechanism A: pp/case by (C, T, config) + dt (s, local) ===")
        grams_str = {n: ("grams" if a[n]["grams"] else "NOGRAM") for n in a}
        for C in (2048, 4096):
            for grams in (1, 0):
                names = [n for n in sorted(a) if a[n]["C"] == C and a[n]["grams"] == grams]
                if not names:
                    continue
                for T in (2048, 4096):
                    row = {t: [] for t, _ in A_CFGS}
                    dts = {t: [] for t, _ in A_CFGS}
                    for n in names:
                        for t, _ in A_CFGS:
                            c = a[n]["cases"][f"T{T}_{t}"]
                            row[t].append(c["score"] * 100)
                            dts[t].append(c["dt"])
                    base = _mean(row["A0"])
                    line = " ".join(
                        f"{t}:{_mean(row[t]):>7.2f}({(_mean(row[t])-base):+6.2f},"
                        f"{_mean(dts[t]):.2f}s)" for t, _ in A_CFGS)
                    print(f"C={C} {grams_str[names[0]]} T={T} n={len(names)}: {line}")
    b = jload(os.path.join(RES, "b.json"))
    if b:
        print("\n=== Mechanism B: per-call pp by config ===")
        for n in sorted(b):
            e = b[n]
            meta = e.get("_meta", {})
            for T in (10, 128, 512, 1024, 2048):
                idx = [i for i, c in enumerate(e["B0"]["cases"]) if c["T"] == T]
                if not idx:
                    continue
                i = idx[0]
                p0 = e["B0"]["cases"][i]["score"] * 100
                p2 = e["B2"]["cases"][i]["score"] * 100
                p3 = e["B3"]["cases"][i]["score"] * 100
                d2 = e["B2"]["dbg"][i]
                d3 = e["B3"]["dbg"][i]
                print(f"{n} T={T}: B0 {p0:+.2f} B2 {p2:+.2f}({p2-p0:+.3f}) "
                      f"B3 {p3:+.2f}({p3-p0:+.3f}) sel2={d2.get('sel')} "
                      f"sel3={d3.get('sel')} dt0/2/3 "
                      f"{e['B0']['cases'][i]['dt']:.2f}/{e['B2']['cases'][i]['dt']:.2f}/"
                      f"{e['B3']['cases'][i]['dt']:.2f} grams={meta.get('grams')} "
                      f"g={meta.get('g')}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "rep"
    if mode == "mini":
        run_mini()
    elif mode == "a":
        run_a()
    elif mode == "b":
        run_b()
    else:
        rep()


if __name__ == "__main__":
    main()
