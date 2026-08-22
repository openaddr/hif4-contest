"""Phase/cProfile attribution for v25 (audit round 2).

Usage:
  python dev/audit2/prof2.py attn              # attn mini calib+dyn timing + cProfile
  python dev/audit2/prof2.py linear c2048_n8192 [calib|dyn|both]
"""
from __future__ import annotations

import cProfile
import os
import pstats
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402


def _prof_run(fn, tag, top=22):
    pr = cProfile.Profile()
    pr.enable()
    fn()
    pr.disable()
    st = pstats.Stats(pr)
    st.sort_stats("cumulative")
    print(f"\n===== cProfile {tag} (cumulative, top {top}) =====")
    st.print_stats(top)
    out = os.path.join(harness.RESULTS, f"cprof_{tag.replace(' ', '_')}.pstats")
    pr.dump_stats(out)


def attn():
    sol = harness.load_variant()
    att = torch.load(os.path.join(harness.MINI, "attn.pt"), weights_only=True)[0]
    qh, kvh, dh = att["q_num_heads"], att["kv_num_heads"], att["head_dim"]
    print(f"attn mini: qh={qh} kvh={kvh} dh={dh} "
          f"calib={[s['q'][0].shape[0] for s in att['calib']]} "
          f"test={[s['q'][0].shape[0] for s in att['test']]}")

    torch.manual_seed(0)
    t0 = time.perf_counter()
    acal = sol.hif4_calibration_attention(att["calib"], qh, kvh, dh)
    print(f"attn calib wall: {time.perf_counter() - t0:.3f}s")

    for smp in att["test"]:
        T = smp["q"][0].shape[0]
        sol._QKV_CARRY.clear()
        sol._VCOMP.update({"n": 0, "el": 0.0})
        t0 = time.perf_counter()
        pq = sol.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh, dh, acal["q_state"])
        tq = time.perf_counter() - t0
        t0 = time.perf_counter()
        pk = sol.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], kvh, dh, acal["k_state"])
        tk = time.perf_counter() - t0
        t0 = time.perf_counter()
        pv = sol.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], kvh, dh, acal["v_state"])
        tv = time.perf_counter() - t0
        print(f"  dyn T={T:>5d}: q {tq:.3f}s k {tk:.3f}s v {tv:.3f}s "
              f"(vcomp n={sol._VCOMP['n']} el={sol._VCOMP['el']:.3f}s)")

    _prof_run(lambda: (torch.manual_seed(0),
                       sol.hif4_calibration_attention(att["calib"], qh, kvh, dh)),
              "attn_calib")


def linear(name, entry):
    sol = harness.load_variant()
    g = harness.load_group(name)
    if entry in ("calib", "both"):
        t0 = time.perf_counter()
        out = run_calib(sol, g)
        print(f"{name} calib wall: {time.perf_counter() - t0:.2f}s")
        st = out["activation_state"]
        if entry == "calib":
            _prof_run(lambda: run_calib(sol, g), f"{name}_calib")
    else:
        torch.manual_seed(0)
        st = sol.hif4_calibration_and_quantize_weight(
            g["weight"][0], g["weight"][1], g["calib_activation_list"])["activation_state"]
    if entry in ("dyn", "both"):
        for pair in g["test_activation_list"]:
            t0 = time.perf_counter()
            sol.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
            print(f"  dyn T={pair[0].shape[0]:>5d}: {time.perf_counter() - t0:.3f}s")
        _prof_run(lambda: [sol.hif4_dynamic_quantize_activation(p[0], p[1], st)
                           for p in g["test_activation_list"]], f"{name}_dyn")


def run_calib(sol, g):
    torch.manual_seed(0)
    return sol.hif4_calibration_and_quantize_weight(
        g["weight"][0], g["weight"][1], g["calib_activation_list"])


if __name__ == "__main__":
    if sys.argv[1] == "attn":
        attn()
    else:
        linear(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "both")
