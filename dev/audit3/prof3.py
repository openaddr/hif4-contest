"""Round-3 profile: wall phases + cProfile top consumers on CURRENT v31.

Usage:
  python dev/audit3/prof3.py build          # build synth data (once)
  python dev/audit3/prof3.py wall           # wall phases all configs + mini
  python dev/audit3/prof3.py cprof c2048_n8192 dyn   # cProfile one phase
  python dev/audit3/prof3.py cprof attn calib
  python dev/audit3/prof3.py top            # aggregated solution-func table
"""
from __future__ import annotations

import cProfile
import os
import pstats
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402


def wall_linear(sol, g, name, reps=3):
    cals, dyns, per_t = [], [], {}
    for _ in range(reps):
        out, ps, tc, td = harness.run_linear(sol, g)
        cals.append(tc)
        dyns.append(td)
    st = out["activation_state"]
    # per-T dyn timing (1 rep, informative only)
    for pair in g["test_activation_list"]:
        T = pair[0].shape[0]
        ts = []
        for _ in range(2):
            t0 = time.perf_counter()
            sol.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
            ts.append(time.perf_counter() - t0)
        per_t[T] = min(ts)
    mc, md = statistics.median(cals), statistics.median(dyns)
    print(f"{name}: calib {mc:6.2f}s | dyn(5 calls) {md:6.2f}s | "
          f"gw={'Y' if st.get('gw') is not None else 'N'} "
          f"tmax={st.get('tmax')} | per-T: "
          + " ".join(f"T{t}={per_t[t]:.3f}" for t in sorted(per_t)))
    sys.stdout.flush()
    return mc, md


def wall():
    sol = harness.load_variant()
    tot = 0.0
    for name, *_ in harness.CONFIGS:
        g = harness.load_group(name)
        mc, md = wall_linear(sol, g, name)
        tot += mc + md
    lin = torch.load(os.path.join(harness.MINI, "linear.pt"), weights_only=True)[0]
    mc, md = wall_linear(sol, lin, "mini_linear")
    tot += mc + md
    print(f"TOTAL (3 synth + mini linear): {tot:.2f}s local")
    # attention, judge-like
    att = torch.load(os.path.join(harness.MINI, "attn.pt"), weights_only=True)[0]
    qh, kvh, dh = att["q_num_heads"], att["kv_num_heads"], att["head_dim"]
    print(f"attn mini: qh={qh} kvh={kvh} dh={dh} "
          f"calib={[s['q'][0].shape[0] for s in att['calib']]} "
          f"test={[s['q'][0].shape[0] for s in att['test']]}")
    ts = []
    for _ in range(3):
        per = []
        t0 = time.perf_counter()
        acal = sol.hif4_calibration_attention(att["calib"], qh, kvh, dh)
        tc = time.perf_counter() - t0
        harness.run_attn_judge(sol, att, per)
        ts.append((tc, sum(t for _, _, t in per)))
    tc = statistics.median(t[0] for t in ts)
    td = statistics.median(t[1] for t in ts)
    print(f"attn mini (judge-like): calib {tc:.3f}s | dyn(5x q/k/v) {td:.3f}s")


def _prof(fn, tag, top=18):
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    st = pstats.Stats(pr)
    st.sort_stats("cumulative")
    print(f"\n===== cProfile {tag} (cumulative, top {top}) =====")
    st.print_stats(top)
    os.makedirs(harness.RESULTS, exist_ok=True)
    pr.dump_stats(os.path.join(harness.RESULTS, f"cprof_{tag.replace(' ', '_')}.pstats"))


def cprof(target, phase):
    sol = harness.load_variant()
    if target == "attn":
        att = torch.load(os.path.join(harness.MINI, "attn.pt"), weights_only=True)[0]
        if phase in ("calib", "both"):
            _prof(lambda: (torch.manual_seed(0),
                           sol.hif4_calibration_attention(
                               att["calib"], att["q_num_heads"],
                               att["kv_num_heads"], att["head_dim"])),
                  "attn_calib")
        if phase in ("dyn", "both"):
            _prof(lambda: harness.run_attn_judge(sol, att), "attn_dyn")
        return
    g = harness.load_group(target)
    if phase in ("calib", "both"):
        _prof(lambda: (torch.manual_seed(0),
                       sol.hif4_calibration_and_quantize_weight(
                           g["weight"][0], g["weight"][1],
                           g["calib_activation_list"])), f"{target}_calib")
    if phase in ("dyn", "both"):
        torch.manual_seed(0)
        st = sol.hif4_calibration_and_quantize_weight(
            g["weight"][0], g["weight"][1], g["calib_activation_list"]
        )["activation_state"]
        _prof(lambda: [sol.hif4_dynamic_quantize_activation(p[0], p[1], st)
                       for p in g["test_activation_list"]], f"{target}_dyn")


def top_table():
    """Aggregate solution-module functions per phase from saved pstats."""
    import glob
    rows = []
    for f in sorted(glob.glob(os.path.join(harness.RESULTS, "*.pstats"))):
        tag = os.path.basename(f)[6:-7]
        try:
            st = pstats.Stats(f)
        except Exception:
            continue
        for (fn, lineno, name), (cc, nc, tt, ct, callers) in st.stats.items():
            if "solution.py" in fn or "_variant_" in fn:
                rows.append((tag, name, nc, tt, ct))
    rows.sort(key=lambda r: -r[3])
    print(f"{'phase':<22}{'function':<32}{'ncalls':>9}{'self':>9}{'cum':>9}")
    for tag, name, nc, tt, ct in rows[:60]:
        print(f"{tag:<22}{name:<32}{nc:>9}{tt:>9.3f}{ct:>9.3f}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "build":
        harness.build()
    elif mode == "wall":
        wall()
    elif mode == "cprof":
        cprof(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "both")
    elif mode == "top":
        top_table()
    else:
        raise SystemExit(__doc__)
