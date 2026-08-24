"""Q/K joint channel balancing: double-holdout battery (synthetic).

Protocol per (config, seed):
  A = calib[:-1] fit + calib[-1] guard  (the pipeline's own convention)
  B = independent test samples (shared channel gains, fresh tokens)
  score/case = (mse_std - mse_play)/mse_std x 100 vs exact alg1 baseline

Arms: off (bit-identical baseline), pre (closed-form s ~ (B/A)^(1/4)),
pre_ng (guard OFF; only on iid/outlier configs -- guard-necessity control).

Usage: python dev/smattn/battery.py [--seeds 8] [--configs ...] [--out r.jsonl]
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
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "dev"))
import hif4  # noqa: E402
import variants as V  # noqa: E402
from measure_persist import make_shared_attn  # noqa: E402


def load_sol():
    spec = importlib.util.spec_from_file_location(
        "_qks_sol", os.path.join(HERE, "solution.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SOL = load_sol()


def clone_state(st):
    if isinstance(st, torch.Tensor):
        return st.clone()
    if isinstance(st, dict):
        return {k: clone_state(v) for k, v in st.items()}
    return st


def state_bytes(st) -> int:
    if isinstance(st, torch.Tensor):
        return st.numel() * st.element_size()
    if isinstance(st, dict):
        return sum(state_bytes(v) for v in st.values())
    return 0


def score_group(group, arm):
    qh, kvh, dh = group["q_num_heads"], group["kv_num_heads"], group["head_dim"]
    SOL.QKS_MODE = "off" if arm == "off" else "pre"
    SOL.QKS_GUARD = not (arm.endswith("_ng") or arm == "pre_noguard")
    SOL.QKS_STABILITY = arm != "pre_nostab"
    SOL.QKS_GAMMA = 0.5 if arm.endswith("_g05") else 1.0
    SOL.QKS_DEBUG.clear()
    torch.manual_seed(0)
    t0 = time.perf_counter()
    cal = SOL.hif4_calibration_attention(group["calib"], qh, kvh, dh)
    t_cal = time.perf_counter() - t0
    dbg = dict(SOL.QKS_DEBUG)
    q_state, k_state, v_state = cal["q_state"], cal["k_state"], cal["v_state"]
    rows = []
    tq = tk = 0.0
    for smp in group["test"]:
        q_ref = hif4.dequantize_nvfp4(*smp["q"])
        k_ref = hif4.dequantize_nvfp4(*smp["k"])
        v_ref = hif4.dequantize_nvfp4(*smp["v"])
        ref = hif4.attn_ref(q_ref, k_ref, v_ref, qh, kvh, dh)
        mse_std = ((hif4.attn_ref(V.deq(V.quant_alg1(q_ref.float())),
                                  V.deq(V.quant_alg1(k_ref.float())),
                                  V.deq(V.quant_alg1(v_ref.float())), qh, kvh, dh)
                    - ref) ** 2).mean().item()
        t0 = time.perf_counter()
        pq = SOL.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh, dh,
                                         clone_state(q_state))
        tq += time.perf_counter() - t0
        t0 = time.perf_counter()
        pk = SOL.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], kvh, dh,
                                         clone_state(k_state))
        tk += time.perf_counter() - t0
        pv = SOL.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], kvh, dh,
                                         clone_state(v_state))
        out = hif4.attn_ref(hif4.hif4_dequantize(pq), hif4.hif4_dequantize(pk),
                            hif4.hif4_dequantize(pv), qh, kvh, dh)
        mse_play = ((out - ref) ** 2).mean().item()
        rows.append((mse_std - mse_play) / mse_std * 100.0)
    s = q_state.get("qs")
    sstats = None
    if s is not None:
        sstats = [round(float(x), 3) for x in
                  (s.float().min(), s.float().max(), s.float().log().std())]
    return {"scores": rows, "t_cal": t_cal, "dbg": dbg, "s_stats": sstats,
            "t_q": tq, "t_k": tk,
            "state_mb": state_bytes(q_state) / 2 ** 20}


# name, qh, kvh, dh, q_spread, k_spread, share, outlier_p
CONFIGS = [
    ("gqa32x8x64",    32, 8, 64,  0.5, 0.4, 1.0, 0.0),
    ("gqa16x2x256",   16, 2, 256, 0.5, 0.4, 1.0, 0.0),   # mini shape
    ("mha8x8x128",     8, 8, 128, 0.5, 0.4, 1.0, 0.0),   # rep=1 MHA
    ("gqa28x4x128",   28, 4, 128, 0.5, 0.4, 1.0, 0.0),
    ("gqa16x2x256_sp08", 16, 2, 256, 0.8, 0.6, 1.0, 0.0),
    ("gqa16x2x256_partial", 16, 2, 256, 0.5, 0.4, 0.7, 0.0),
    ("gqa16x2x256_outl", 16, 2, 256, 0.5, 0.4, 1.0, 0.002),
    ("gqa16x2x256_flat", 16, 2, 256, 0.15, 0.12, 1.0, 0.0),
    ("gqa16x2x256_iid", 16, 2, 256, 0.5, 0.4, 0.0, 0.0),  # true-iid negative control
    ("gqa16x2x256_cs_tf", 16, 2, 256, 0.5, 0.4, 1.0, 0.0),  # adversarial (below)
]
# adversarial: calib fully structured, test gains FRESH (guard blind by
# construction -- fit+hold share calib structure, test does not).  Measures
# the worst-case damage of an accepted-but-wrong s.
NG_CONFIGS = {"gqa16x2x256_iid", "gqa16x2x256_outl", "gqa16x2x256_partial",
              "gqa16x2x256_cs_tf"}
# damped-arm configs (gamma 0.5): quantify the cs_tf damage / upside tradeoff
G05_CONFIGS = {"gqa16x2x256", "gqa16x2x256_cs_tf", "gqa16x2x256_partial"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "results.jsonl"))
    args = ap.parse_args()
    cfgs = CONFIGS if not args.configs else [c for c in CONFIGS
                                             if c[0] in args.configs]
    with open(args.out, "a", encoding="utf-8") as fh:
        for name, qh, kvh, dh, qsp, ksp, share, outp in cfgs:
            for k in range(args.seeds):
                seed = 8100 + 137 * k + (sum(map(ord, name)) % 911)
                grp = make_shared_attn(seed, qh, kvh, dh, q_spread=qsp,
                                       k_spread=ksp, share=share,
                                       outlier_p=outp)
                if name.endswith("_cs_tf"):
                    # keep the structured calib, swap in FRESH-gain tests
                    grp2 = make_shared_attn(seed + 55501, qh, kvh, dh,
                                            q_spread=qsp, k_spread=ksp,
                                            share=0.0, outlier_p=outp)
                    grp = {**grp, "test": grp2["test"]}
                base_mean = None
                arms = ["off", "pre"]
                if name in NG_CONFIGS:
                    arms.append("pre_ng")
                if name in G05_CONFIGS:
                    arms.append("pre_g05")
                for arm in arms:
                    try:
                        r = score_group(grp, arm)
                    except Exception as exc:
                        fh.write(json.dumps({"cfg": name, "seed": seed,
                                             "arm": arm, "error": repr(exc)}) + "\n")
                        fh.flush()
                        continue
                    mean = sum(r["scores"]) / len(r["scores"])
                    rec = {"cfg": name, "seed": seed, "arm": arm,
                           "mean_pp": round(mean, 3),
                           "scores": [round(x, 3) for x in r["scores"]],
                           "t_cal": round(r["t_cal"], 2), "dbg": r["dbg"],
                           "s_stats": r["s_stats"],
                           "t_q_ms": round(r["t_q"] * 1000 / len(grp["test"]), 1),
                           "t_k_ms": round(r["t_k"] * 1000 / len(grp["test"]), 1),
                           "state_mb": round(r["state_mb"], 2)}
                    if arm == "off":
                        base_mean = mean
                    rec["delta_pp"] = round(mean - (base_mean or mean), 3)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    print(f"[{name} s{k}] {arm:7s} mean={mean:+8.2f} "
                          f"d={rec['delta_pp']:+8.2f} cal={r['t_cal']:.1f}s "
                          f"acc={r['dbg'].get('accepted')}", flush=True)
    SOL.QKS_MODE = "pre"
    SOL.QKS_GUARD = True


if __name__ == "__main__":
    main()
