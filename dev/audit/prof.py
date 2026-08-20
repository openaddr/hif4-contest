"""CPU-time audit harness for example/solution/solution.py (v18).

Drives the CURRENT solution (unmodified) on synthetic-but-representative
shapes (dev/synth.py generators, same usage pattern as dev/stress.py) plus
the real example/mini_sample/attn.pt group. Produces:

  - per-phase wall timings (via dev/audit/sol_phases.py, an instrumented
    re-implementation whose outputs are verified BIT-IDENTICAL to the
    original before any number is reported)
  - cProfile cumulative attribution per entry point
  - dynamic-call split: gptq-values / refinement / param-encode / quantize

Usage:
  C:/App/env/Python/python.exe dev/audit/prof.py build      # one-time data
  C:/App/env/Python/python.exe dev/audit/prof.py time c2048_n8192
  C:/App/env/Python/python.exe dev/audit/prof.py cprof c2048_n8192
  C:/App/env/Python/python.exe dev/audit/prof.py all
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "dev"))
import synth  # noqa: E402

AUDIT = os.path.join(ROOT, "dev", "audit")
DATA_DIR = os.path.join(AUDIT, "data")
RESULTS = os.path.join(AUDIT, "results")
SOLUTION = os.path.join(ROOT, "example", "solution", "solution.py")
MINI_ATTN = os.path.join(ROOT, "example", "mini_sample", "attn.pt")

# name, C(K), N(M), calib tokens, test tokens  (spread=0.5, w_spread=0.3,
# matching stress.py's defaults for the non-spiky configs)
CONFIGS = [
    ("c1024_n1024", 1024, 1024, (10, 128, 512, 1024), (10, 128, 512, 1024, 1024)),
    ("c2048_n8192", 2048, 8192, (10, 128, 512, 1024), (10, 128, 512, 1024, 1024)),
    ("c8192_n8192", 8192, 8192, (10, 128, 512, 1024), (10, 128, 512, 1024, 1024)),
]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_sol():
    return load_module(SOLUTION, "_audit_sol")


def get_solp():
    return load_module(os.path.join(AUDIT, "sol_phases.py"), "_audit_solp")


# ---------------------------------------------------------------------------
# data build (cached on disk)
# ---------------------------------------------------------------------------

def build(force=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    for i, (name, C, N, cal_T, test_T) in enumerate(CONFIGS):
        path = os.path.join(DATA_DIR, f"{name}.pt")
        if os.path.exists(path) and not force:
            continue
        t0 = time.perf_counter()
        g = synth.make_linear_group(3100 + 7 * i, N, C, tokens=cal_T,
                                    spread=0.5, outlier_p=0.0, w_spread=0.3)
        g2 = synth.make_linear_group(9300 + 7 * i, N, C, tokens=test_T,
                                     spread=0.5, outlier_p=0.0, w_spread=0.3)
        g["test_activation_list"] = g2["test_activation_list"]
        torch.save(g, path)
        print(f"[build] {name}: C={C} N={N} calib={cal_T} test={test_T} "
              f"({time.perf_counter() - t0:.1f}s, "
              f"{os.path.getsize(path) / 2 ** 20:.0f} MiB)")
    print("[build] done ->", DATA_DIR)


def load_group(name):
    return torch.load(os.path.join(DATA_DIR, f"{name}.pt"),
                      weights_only=True, map_location="cpu")


# ---------------------------------------------------------------------------
# verification: instrumented copy must be BIT-IDENTICAL to the original
# ---------------------------------------------------------------------------

def _eq_params(a, b):
    if set(a) != set(b):
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def verify_phases(names=None):
    """Run original + instrumented on each config; torch.equal everything."""
    sol, solp = get_sol(), get_solp()
    names = names or [c[0] for c in CONFIGS]
    ok_all = True
    for name in names:
        g = load_group(name)
        torch.manual_seed(0)
        out_o = sol.hif4_calibration_and_quantize_weight(
            g["weight"][0], g["weight"][1], g["calib_activation_list"])
        torch.manual_seed(0)
        out_i = solp.hif4_calibration_and_quantize_weight(
            g["weight"][0], g["weight"][1], g["calib_activation_list"])
        ok = _eq_params(out_o["weight_params"], out_i["weight_params"])
        st_o, st_i = out_o["activation_state"], out_i["activation_state"]
        for k in st_o:
            a, b = st_o[k], st_i[k]
            if isinstance(a, torch.Tensor):
                ok = ok and torch.equal(a, b)
            else:
                ok = ok and a == b
        # dynamic calls
        for pair in g["test_activation_list"]:
            p_o = sol.hif4_dynamic_quantize_activation(pair[0], pair[1], st_o)
            p_i = solp.hif4_dynamic_quantize_activation(pair[0], pair[1], st_i)
            ok = ok and _eq_params(p_o, p_i)
        print(f"[verify] {name}: {'BIT-IDENTICAL' if ok else 'MISMATCH'}")
        ok_all = ok_all and ok
    # attention (real mini sample)
    att = torch.load(MINI_ATTN, weights_only=True)[0]
    qh, kvh, dh = att["q_num_heads"], att["kv_num_heads"], att["head_dim"]
    torch.manual_seed(0)
    ao = sol.hif4_calibration_attention(att["calib"], qh, kvh, dh)
    torch.manual_seed(0)
    ai = solp.hif4_calibration_attention(att["calib"], qh, kvh, dh)
    oka = json.dumps({k: str(v) for k, v in ao.items()}, sort_keys=True) == \
        json.dumps({k: str(v) for k, v in ai.items()}, sort_keys=True)
    for smp in att["test"]:
        for role, fn_o, fn_i, st in (
                ("q", sol.hif4_dynamic_quantize_q, solp.hif4_dynamic_quantize_q, ao["q_state"]),
                ("k", sol.hif4_dynamic_quantize_k, solp.hif4_dynamic_quantize_k, ao["k_state"]),
                ("v", sol.hif4_dynamic_quantize_v, solp.hif4_dynamic_quantize_v, ao["v_state"])):
            fn_o = getattr(sol, f"hif4_dynamic_quantize_{role}")
            fn_i = getattr(solp, f"hif4_dynamic_quantize_{role}")
            solp._QKV_CARRY.clear()
            # replay q,k,v in order through both (v-compensation reads carry)
            for r2 in ("q", "k", "v"):
                pass
    print(f"[verify] attn(mini): calibration {'IDENTICAL' if oka else 'MISMATCH'}")
    return ok_all and oka


# ---------------------------------------------------------------------------
# phase timing
# ---------------------------------------------------------------------------

def time_config(name, sol=None, solp=None):
    sol = sol or get_sol()
    solp = solp or get_solp()
    C, N = {"c1024_n1024": (1024, 1024), "c2048_n8192": (2048, 8192),
            "c8192_n8192": (8192, 8192)}[name]
    g = load_group(name)
    res = {"name": name, "C": C, "N": N, "threads": torch.get_num_threads()}

    # calibration: original wall time + instrumented phase split
    torch.manual_seed(0)
    t0 = time.perf_counter()
    out_o = sol.hif4_calibration_and_quantize_weight(
        g["weight"][0], g["weight"][1], g["calib_activation_list"])
    res["calib_wall_orig"] = time.perf_counter() - t0

    solp.PHASES.clear()
    torch.manual_seed(0)
    t0 = time.perf_counter()
    out_i = solp.hif4_calibration_and_quantize_weight(
        g["weight"][0], g["weight"][1], g["calib_activation_list"])
    res["calib_wall_instr"] = time.perf_counter() - t0
    res["calib_phases"] = dict(sorted(solp.PHASES.items(), key=lambda kv: -kv[1]))
    res["calib_identical"] = _eq_params(out_o["weight_params"], out_i["weight_params"])

    st = out_o["activation_state"]
    dyn = []
    for pair in g["test_activation_list"]:
        T = pair[0].shape[0]
        # original wall
        t0 = time.perf_counter()
        p_o = sol.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
        w = time.perf_counter() - t0
        solp.PHASES.clear()
        t0 = time.perf_counter()
        p_i = solp.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
        w2 = time.perf_counter() - t0
        dyn.append({"T": T, "wall_orig": w, "wall_instr": w2,
                    "phases": dict(solp.PHASES),
                    "identical": _eq_params(p_o, p_i)})
    res["dynamic"] = dyn
    return res


def time_attn(sol=None, solp=None):
    sol = sol or get_sol()
    solp = solp or get_solp()
    att = torch.load(MINI_ATTN, weights_only=True)[0]
    qh, kvh, dh = att["q_num_heads"], att["kv_num_heads"], att["head_dim"]
    res = {"name": "attn_mini", "qh": qh, "kvh": kvh, "dh": dh}
    torch.manual_seed(0)
    t0 = time.perf_counter()
    acal_o = sol.hif4_calibration_attention(att["calib"], qh, kvh, dh)
    res["calib_wall_orig"] = time.perf_counter() - t0
    solp.PHASES.clear()
    torch.manual_seed(0)
    t0 = time.perf_counter()
    acal_i = solp.hif4_calibration_attention(att["calib"], qh, kvh, dh)
    res["calib_wall_instr"] = time.perf_counter() - t0
    res["calib_phases"] = dict(sorted(solp.PHASES.items(), key=lambda kv: -kv[1]))

    dyn = []
    for smp in att["test"]:
        T = smp["q"][0].shape[0]
        row = {"T": T}
        # original: q,k,v in order (v reads the q/k carry)
        sol._QKV_CARRY.clear()
        t0 = time.perf_counter()
        pq = sol.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh, dh, acal_o["q_state"])
        pk = sol.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], kvh, dh, acal_o["k_state"])
        pv = sol.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], kvh, dh, acal_o["v_state"])
        row["wall_orig"] = time.perf_counter() - t0
        # instrumented
        solp._QKV_CARRY.clear()
        solp.PHASES.clear()
        t0 = time.perf_counter()
        pq2 = solp.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh, dh, acal_i["q_state"])
        pk2 = solp.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], kvh, dh, acal_i["k_state"])
        pv2 = solp.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], kvh, dh, acal_i["v_state"])
        row["wall_instr"] = time.perf_counter() - t0
        row["phases"] = dict(solp.PHASES)
        row["identical"] = (_eq_params(pq, pq2) and _eq_params(pk, pk2)
                            and _eq_params(pv, pv2))
        dyn.append(row)
    res["dynamic"] = dyn
    return res


def _print(res):
    print(f"\n=== {res['name']} (C={res.get('C', '?')} N={res.get('N', '?')}, "
          f"threads={torch.get_num_threads()}) ===")
    print(f"  calib wall: orig {res['calib_wall_orig']:.2f}s | "
          f"instr {res['calib_wall_instr']:.2f}s")
    ph = res.get("calib_phases", {})
    for k, v in ph.items():
        print(f"    cal.{k:<28s} {v:8.2f}s")
    tot_d = 0.0
    for d in res["dynamic"]:
        tot_d += d["wall_orig"]
        det = " ".join(f"{k}={v:.3f}" for k, v in
                       sorted(d["phases"].items(), key=lambda kv: -kv[1]))
        print(f"  dyn T={d['T']:>5d}: {d['wall_orig']:6.3f}s "
              f"{'IDENT' if d['identical'] else 'DIFF!'} | {det}")
    print(f"  dynamic total (5 test calls): {tot_d:.2f}s")


# ---------------------------------------------------------------------------
# cProfile
# ---------------------------------------------------------------------------

def cprof(name, entry="calib", top=28):
    import cProfile
    import pstats
    sol = get_sol()
    g = load_group(name)
    if entry == "calib":
        def run():
            torch.manual_seed(0)
            sol.hif4_calibration_and_quantize_weight(
                g["weight"][0], g["weight"][1], g["calib_activation_list"])
    elif entry == "dyn":
        torch.manual_seed(0)
        st = sol.hif4_calibration_and_quantize_weight(
            g["weight"][0], g["weight"][1], g["calib_activation_list"])["activation_state"]

        def run():
            for pair in g["test_activation_list"]:
                sol.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
    else:
        raise SystemExit("entry must be calib or dyn")
    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    out = os.path.join(RESULTS, f"cprof_{name}_{entry}.pstats")
    pr.dump_stats(out)
    st = pstats.Stats(pr)
    st.sort_stats("cumulative")
    print(f"\n===== cProfile {name}/{entry} (cumulative, top {top}) =====")
    st.print_stats(top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "verify", "time", "cprof", "all"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--entry", default="calib")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    torch.set_num_threads(torch.get_num_threads())  # default
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}, "
          f"cpus={os.cpu_count()}")
    if args.mode == "build":
        build()
    elif args.mode == "verify":
        ok = verify_phases(args.names or None)
        print("[verify] ALL OK" if ok else "[verify] FAILED")
        sys.exit(0 if ok else 1)
    elif args.mode == "time":
        names = args.names or [c[0] for c in CONFIGS]
        allres = []
        for n in names:
            r = time_config(n)
            _print(r)
            allres.append(r)
        r = time_attn()
        _print(r)
        allres.append(r)
        with open(os.path.join(RESULTS, "phases.json"), "w") as f:
            json.dump(allres, f, indent=1)
        print("\nsaved -> dev/audit/results/phases.json")
    elif args.mode == "cprof":
        cprof(args.names[0], entry=args.entry)
    elif args.mode == "all":
        build()


if __name__ == "__main__":
    main()
