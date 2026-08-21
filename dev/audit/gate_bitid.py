"""Gate 1: bit-identity of the patched live solution vs the v19 baseline.

Compares dev/audit/solution_v19_baseline.py (reference) against
example/solution/solution.py (patched) on:
  - synthetic linear groups C/N = 1024/1024, 2048/8192, 4096/4096, 8192/8192,
    calib (10,128,512,1024), test T=(10,128,512,1024,1024)
  - 3 additional seeds at C=2048 N=8192 (tie-storm coverage)
  - real mini linear + attention groups (bonus)
torch.equal on: every weight_params tensor, every activation_state tensor
(incl. bf16 grams gw/gwf, u_act, order, s), every dynamic-call param dict,
and attention q/k/v states + dynamic params.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "dev"))
import synth  # noqa: E402

AUDIT = os.path.join(ROOT, "dev", "audit")
DATA_DIR = os.path.join(AUDIT, "data")
BASE = os.path.join(AUDIT, "solution_v19_baseline.py")
LIVE = os.path.join(ROOT, "example", "solution", "solution.py")
MINI = os.path.join(ROOT, "example", "mini_sample")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def eq_params(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def eq_state(a, b):
    for k in set(a) | set(b):
        x, y = a.get(k), b.get(k)
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
                    and torch.equal(x, y)):
                return False
        elif x != y:
            return False
    return True


def make_group(seed, N, C, cal_T, test_T):
    g = synth.make_linear_group(seed, N, C, tokens=cal_T, spread=0.5,
                                outlier_p=0.0, w_spread=0.3)
    g2 = synth.make_linear_group(seed + 7777, N, C, tokens=test_T, spread=0.5,
                                 outlier_p=0.0, w_spread=0.3)
    g["test_activation_list"] = g2["test_activation_list"]
    return g


def check_linear(base, live, g, tag):
    torch.manual_seed(0)
    ob = base.hif4_calibration_and_quantize_weight(
        g["weight"][0], g["weight"][1], g["calib_activation_list"])
    torch.manual_seed(0)
    ov = live.hif4_calibration_and_quantize_weight(
        g["weight"][0], g["weight"][1], g["calib_activation_list"])
    ok = eq_params(ob["weight_params"], ov["weight_params"])
    ok = ok and eq_state(ob["activation_state"], ov["activation_state"])
    n_dyn = 0
    for pair in g["test_activation_list"]:
        pb = base.hif4_dynamic_quantize_activation(pair[0], pair[1], ob["activation_state"])
        pv = live.hif4_dynamic_quantize_activation(pair[0], pair[1], ov["activation_state"])
        ok = ok and eq_params(pb, pv)
        n_dyn += 1
    print(f"[gate1] {tag:<28s} weight/state/{n_dyn}dyn params "
          f"torch.equal: {'PASS' if ok else 'FAIL'}")
    return ok


def check_attn(base, live):
    att = torch.load(os.path.join(MINI, "attn.pt"), weights_only=True)[0]
    qh, kvh, dh = att["q_num_heads"], att["kv_num_heads"], att["head_dim"]
    torch.manual_seed(0)
    ab = base.hif4_calibration_attention(att["calib"], qh, kvh, dh)
    torch.manual_seed(0)
    av = live.hif4_calibration_attention(att["calib"], qh, kvh, dh)
    ok = all(eq_state(ab[k], av[k]) for k in ("q_state", "k_state")) and \
        ab["v_state"] == av["v_state"]
    for smp in att["test"]:
        for mod_b, mod_v, ac_b, ac_v in ((base, live, ab, av),):
            mod_b._QKV_CARRY.clear(); mod_v._QKV_CARRY.clear()
            pq = mod_b.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh, dh, ac_b["q_state"])
            pv = mod_v.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh, dh, ac_v["q_state"])
            pk = mod_b.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], kvh, dh, ac_b["k_state"])
            kv = mod_v.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], kvh, dh, ac_v["k_state"])
            pb = mod_b.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], kvh, dh, ac_b["v_state"])
            pvv = mod_v.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], kvh, dh, ac_v["v_state"])
            ok = ok and eq_params(pq, pv) and eq_params(pk, kv) and eq_params(pb, pvv)
    print(f"[gate1] attn mini (state + q/k/v dyn)     torch.equal: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    base = load(BASE, "_gate_base")
    live = load(LIVE, "_gate_live")
    ok = True
    configs = [
        ("c1024_n1024", 1024, 1024, (10, 128, 512, 1024)),
        ("c2048_n8192", 2048, 8192, (10, 128, 512, 1024)),
        ("c4096_n4096", 4096, 4096, (10, 128, 512, 1024)),
        ("c8192_n8192", 8192, 8192, (10, 128, 512, 1024)),
    ]
    for name, C, N, cal_T in configs:
        p = os.path.join(DATA_DIR, f"{name}.pt")
        if os.path.exists(p):
            g = torch.load(p, weights_only=True, map_location="cpu")
        else:
            os.makedirs(DATA_DIR, exist_ok=True)
            g = make_group(3100 + C, N, C, cal_T, (10, 128, 512, 1024, 1024))
        ok = ok and check_linear(base, live, g, name)
    # 3 extra seeds at C=2048 N=8192
    for seed in (4242, 5151, 6262):
        g = make_group(seed, 8192, 2048, (10, 128, 512, 1024), (10, 128, 512, 1024, 1024))
        ok = ok and check_linear(base, live, g, f"c2048_n8192 seed{seed}")
    # real mini linear
    lin = torch.load(os.path.join(MINI, "linear.pt"), weights_only=True)[0]
    ok = ok and check_linear(base, live, lin, "mini linear (real)")
    ok = ok and check_attn(base, live)
    print("[gate1] OVERALL:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
