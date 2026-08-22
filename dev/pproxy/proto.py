"""Prototype: calibration-fitted attention-P proxy for V-side lattice refinement.

Judge attention: out = softmax(q k^T / sqrt(dh)) @ v, NO output projection
(Wv = I). The v dynamic call receives ONLY v (call isolation proven), so the
exact-P oracle (dev/attn_refine: +7.4..+7.8 pp/case) is unshippable. This
prototype tests the middle ground: a calibration-FITTED time-Gram proxy.

Design: at calibration time, per distinct token count R (calib and test use
the SAME R multiset -- verified on mini and on every synthetic group), build
  G_R[hv] = sum_{q heads h in group hv} P_h^T P_h,   P_h = softmax(q_cal k_cal^T / sqrt(dh))
from the calibration (q,k) pair(s) with R rows, and carry it in v_state
keyed by R (bf16 when shipped). At the v-dynamic call the incoming tensor's
row count selects G_R; v values are refined against the Gram image of the
residual with the solution's flip machinery (top-1 per COLUMN: under a
time-Gram the columns of D = v_hat - v are independent).

Proxy variants (fit on calib[:-1], mirroring the solution's hold-out
discipline): single = first calib sample per R; meanP = E[P]^T E[P];
EptP = E[P^T P] (expectations over calib samples sharing R).

Data regimes (synthetic; calib/test relationship controlled at the (q,k)
level before NVFP4 re-quantization):
  (a) same  : test samples == calib samples (upper bound; the only residual
              mismatch is the solution's own q/k quantization noise),
  (b) diff  : test q/k = 0.8*calib + sqrt(1-0.8^2)*fresh same-distribution
              draws (measured P-cos ~0.45-0.70, matching the mini judge
              profile 0.27..0.76; v is fresh),
  (c) shift : independent draws with channel spread*2.25 (stress).

Scored per test call as s = (mse_std - mse_play)/mse_std, mse_std from the
paper Alg-1 quantizer (variants.quant_alg1), plain path (carry cleared
before the v call = the online reality). Deltas are pp of case score vs the
current solution's baseline output. Run: python dev/pproxy/proto.py
(writes results.json next to this file).
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "dev"))
import hif4  # noqa: E402
import synth  # noqa: E402
import variants as V  # noqa: E402


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# reuse the prior agent's attention machinery verbatim (refine loop, oracle,
# exact-P objective) and its already-loaded copy of the solution module
AR = load_mod(os.path.join(ROOT, "dev", "attn_refine", "proto.py"), "ar_proto")
SOL = AR.SOL
SWEEPS = (2, 6)
MODES = ("single", "meanP", "EptP")
ALPHA = 0.8          # regime-(b) blend weight (P-cos ~0.45-0.70, judge-like)


# ---------------------------------------------------------------- proxies
def head_Ps(q, k, qh, kvh, dh):
    T = q.shape[0]
    qf = q.view(T, qh, dh).transpose(0, 1)
    kf = k.view(T, kvh, dh).transpose(0, 1)
    rep = qh // kvh
    return [torch.softmax(qf[h] @ kf[h // rep].T / math.sqrt(dh), dim=-1)
            for h in range(qh)]


def calib_grams(calib, qh, kvh, dh, mode):
    """{R: (kvh,R,R) fp32} fitted on calib[:-1] (last sample = hold-out)."""
    by_R: dict[int, list] = {}
    for smp in calib[:-1]:
        by_R.setdefault(int(smp["q"][0].shape[0]), []).append(smp)
    rep = qh // kvh
    out = {}
    for R, smps in sorted(by_R.items()):
        smps = smps[:1] if mode == "single" else smps
        G = torch.zeros(kvh, R, R, dtype=torch.float32)
        if mode == "EptP":
            for smp in smps:
                q = hif4.dequantize_nvfp4(*smp["q"]).float()
                k = hif4.dequantize_nvfp4(*smp["k"]).float()
                G += AR.attention_grams(q, k, qh, kvh, dh)
            G /= len(smps)
        else:  # single / meanP: square the (per-head) mean P
            acc = [torch.zeros(R, R) for _ in range(qh)]
            for smp in smps:
                q = hif4.dequantize_nvfp4(*smp["q"]).float()
                k = hif4.dequantize_nvfp4(*smp["k"]).float()
                for h, P in enumerate(head_Ps(q, k, qh, kvh, dh)):
                    acc[h] += P
            for h in range(qh):
                Pbar = acc[h] / len(smps)
                G[h // rep] += Pbar.T @ Pbar
        out[R] = G
    return out


# ---------------------------------------------------------------- synth data
def nvfp4_pair(x):
    """x fp32 -> (carrier, scale) NVFP4 pair, same recipe as synth."""
    T, C = x.shape
    xb = x.reshape(-1, 16)
    amax = xb.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    scale = ((amax / 6.0).to(torch.bfloat16).float()).clamp_min(1e-30)
    q = xb / scale
    idx = torch.bucketize(q.abs(), (synth.E2M1_GRID[1:] + synth.E2M1_GRID[:-1]) / 2.0)
    carrier = (torch.sign(q) * synth.E2M1_GRID[idx]).reshape(T, -1).to(torch.bfloat16)
    return carrier, scale.reshape(T, -1).to(torch.bfloat16)


def synth_regimes(seed, qh, kvh, dh, seqlens, spread):
    base = synth.make_attn_group(seed, qh, kvh, dh, seqlens=seqlens, spread=spread)
    fresh = synth.make_attn_group(seed + 1000, qh, kvh, dh, seqlens=seqlens, spread=spread)
    shift = synth.make_attn_group(seed + 2000, qh, kvh, dh, seqlens=seqlens,
                                  spread=spread * 2.25)
    b = math.sqrt(1.0 - ALPHA * ALPHA)
    diff = []
    for cs, fs in zip(base["calib"], fresh["test"]):
        q1 = hif4.dequantize_nvfp4(*cs["q"]).float()
        k1 = hif4.dequantize_nvfp4(*cs["k"]).float()
        q2 = hif4.dequantize_nvfp4(*fs["q"]).float()
        k2 = hif4.dequantize_nvfp4(*fs["k"]).float()
        diff.append({"q": nvfp4_pair(ALPHA * q1 + b * q2),
                     "k": nvfp4_pair(ALPHA * k1 + b * k2),
                     "v": fs["v"]})
    tests = {"same": base["calib"], "diff": diff, "shift": shift["test"]}
    return base, tests


# ---------------------------------------------------------------- evaluation
def g_cos(Ga, Gb):
    kvh = Ga.shape[0]
    v = [float((Ga[h] * Gb[h]).sum() / (Ga[h].norm() * Gb[h].norm() + 1e-30))
         for h in range(kvh)]
    return sum(v) / kvh


def eval_group(gname, group, tests, qh, kvh, dh, grams_mini_ref=None):
    torch.manual_seed(0)
    t0 = time.perf_counter()
    acal = SOL.hif4_calibration_attention(group["calib"], qh, kvh, dh)
    tcal = time.perf_counter() - t0
    proxies = {m: calib_grams(group["calib"], qh, kvh, dh, m) for m in MODES}
    test_R = {int(s["q"][0].shape[0]) for s in group["test"]}
    cal_R = set(proxies["single"].keys())
    rmatch = test_R <= cal_R | {int(s["q"][0].shape[0]) for s in group["calib"]}
    print(f"\n=== {gname}: qh={qh} kvh={kvh} dh={dh} C={kvh*dh} "
          f"calib {tcal:.1f}s  R-match(test in calib)={rmatch} "
          f"cal_R={sorted(cal_R)} test_R={sorted(test_R)}")
    recs = []
    for regime, smps in tests.items():
        for ti, smp in enumerate(smps):
            q_ref = hif4.dequantize_nvfp4(*smp["q"])
            k_ref = hif4.dequantize_nvfp4(*smp["k"])
            v_ref = hif4.dequantize_nvfp4(*smp["v"])
            ref = hif4.attn_ref(q_ref, k_ref, v_ref, qh, kvh, dh)
            qs = V.deq(V.quant_alg1(q_ref.float()))
            ks = V.deq(V.quant_alg1(k_ref.float()))
            vs = V.deq(V.quant_alg1(v_ref.float()))
            mse_std = ((hif4.attn_ref(qs, ks, vs, qh, kvh, dh) - ref) ** 2).mean().item()

            pq = SOL.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh, dh, acal["q_state"])
            pk = SOL.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], kvh, dh, acal["k_state"])
            SOL._QKV_CARRY.clear()          # judge per-call isolation
            pv = SOL.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], kvh, dh, acal["v_state"])
            q_hat = hif4.hif4_dequantize(pq)
            k_hat = hif4.hif4_dequantize(pk)
            mse_base = ((hif4.attn_ref(q_hat, k_hat, hif4.hif4_dequantize(pv),
                                       qh, kvh, dh) - ref) ** 2).mean().item()
            s_base = (mse_std - mse_base) / mse_std

            x = v_ref.float()
            values = SOL._deq_params(pv)
            unit = SOL._params_unit_flat(pv)
            T = x.shape[0]
            G_or = AR.attention_grams(q_hat.float(), k_hat.float(), qh, kvh, dh)

            rec = {"group": gname, "regime": regime, "ti": ti, "T": T,
                   "s_base": s_base, "modes": {}, "oracle": {}, "gcos": {}}
            print(f"[{gname} {regime} t{ti}] T={T} base={s_base:+.4f}", end="")
            for m in MODES:
                G = proxies[m].get(T)
                if G is None:
                    for ns in SWEEPS:
                        rec["modes"].setdefault(m, {})[ns] = 0.0
                    continue
                rec["gcos"][m] = g_cos(G, G_or)
                for ns in SWEEPS:
                    t1 = time.perf_counter()
                    vr = AR.refine_with_gram(x, values.clone(), unit.clone(), G, ns)
                    dt = time.perf_counter() - t1
                    p_r = SOL._values_to_params(vr.contiguous(), pv)
                    mse_r = ((hif4.attn_ref(q_hat, k_hat, hif4.hif4_dequantize(p_r),
                                            qh, kvh, dh) - ref) ** 2).mean().item()
                    ds = (mse_std - mse_r) / mse_std - s_base
                    rec["modes"].setdefault(m, {})[ns] = ds
                    rec["modes"][m][f"ms{ns}"] = dt * 1000.0
            # oracle ceiling (exact per-call P, unshippable)
            t1 = time.perf_counter()
            vr = AR.refine_exact_p(x, values.clone(), unit.clone(), q_hat.float(),
                                   k_hat.float(), qh, kvh, dh, SWEEPS[-1])
            dt = time.perf_counter() - t1
            p_r = SOL._values_to_params(vr.contiguous(), pv)
            mse_r = ((hif4.attn_ref(q_hat, k_hat, hif4.hif4_dequantize(p_r),
                                    qh, kvh, dh) - ref) ** 2).mean().item()
            rec["oracle"][SWEEPS[-1]] = (mse_std - mse_r) / mse_std - s_base
            rec["oracle"]["ms"] = dt * 1000.0
            rec["oracle"]["gcos"] = 1.0
            fmt = "  ".join(f"{m}:{rec['modes'][m][ns]*100:+.2f}" for m in MODES)
            gcs = " ".join(f"{m}={rec['gcos'][m]:.2f}" for m in MODES if m in rec["gcos"])
            print(f" | {fmt} pp | oracle6:{rec['oracle'][6]*100:+.2f}pp | gcos {gcs}")
            recs.append(rec)
    return recs, proxies


# ---------------------------------------------------------------- timing
def gram_image(G, dv, kvh, dh):
    M = torch.empty_like(dv)
    for hv in range(kvh):
        sl = slice(hv * dh, (hv + 1) * dh)
        M[:, sl] = G[hv] @ dv[:, sl]
    return M


def timing_block(timing_cases):
    """timing_cases: list of (label, G(R->(kvh,R,R)), kvh, dh, R_list)."""
    rows = []
    for label, Gd, kvh, dh, Rlist in timing_cases:
        C = kvh * dh
        for R in Rlist:
            if R not in Gd:
                continue
            torch.manual_seed(0)
            x = torch.randn(R, C) * 0.05
            p = SOL._dyn_table(x, None, has_scale=False)
            values = SOL._deq_params(p)
            unit = SOL._params_unit_flat(p)
            G = Gd[R]
            dv = (values - x)
            t0 = time.perf_counter()
            for _ in range(3):
                gram_image(G, dv, kvh, dh)
            t_minit = (time.perf_counter() - t0) / 3 * 1000
            t0 = time.perf_counter()
            AR.refine_with_gram(x, values.clone(), unit.clone(), G, 2)
            t2 = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            AR.refine_with_gram(x, values.clone(), unit.clone(), G, 6)
            t6 = (time.perf_counter() - t0) * 1000
            rows.append({"case": label, "C": C, "kvh": kvh, "R": R,
                         "minit_ms": t_minit, "sw2_ms": t2, "sw6_ms": t6})
            print(f"  [{label} C={C} kvh={kvh}] R={R:5d}  M-init {t_minit:7.1f} ms  "
                  f"refine sw2 {t2:7.1f} ms  sw6 {t6:7.1f} ms  "
                  f"state/R {kvh*R*R*2/1024/1024:.2f} MiB")
    return rows


def state_bytes(proxies_by_group):
    tot = {}
    for gname, Gd in proxies_by_group.items():
        b = sum(G.shape[0] * G.shape[1] * G.shape[2] * 2 for G in Gd.values())
        tot[gname] = b
        print(f"  state {gname}: {b/1024/1024:.2f} MiB bf16 "
              f"({sorted((R, G.shape[0]) for R, G in Gd.items())})")
    return tot


# ---------------------------------------------------------------- main
def main():
    torch.manual_seed(0)
    att = torch.load(os.path.join(ROOT, "example", "mini_sample", "attn.pt"),
                     weights_only=True, map_location="cpu")[0]
    all_recs = []
    grams_for_timing = {}

    recs, prox = eval_group("mini", att, {"judge": att["test"]},
                            att["q_num_heads"], att["kv_num_heads"], att["head_dim"])
    all_recs += recs
    grams_for_timing["mini"] = prox["single"]

    groups = [
        ("synA", 101, (16, 2, 256), (10, 128, 512, 1024, 1024), 0.4),
        ("synB", 102, (8, 8, 128), (128, 128, 512, 1024, 1024), 0.5),
        ("synC", 103, (32, 4, 128), (10, 512, 512, 1024, 1024), 0.3),
    ]
    for gname, seed, (qh, kvh, dh), seqlens, spread in groups:
        base, tests = synth_regimes(seed, qh, kvh, dh, seqlens, spread)
        recs, prox = eval_group(gname, base, tests, qh, kvh, dh)
        all_recs += recs
        grams_for_timing[gname] = prox["single"]

    print("\n=== 3x3 summary (dscore pp vs baseline, sw6 / sw2 in brackets; "
          "mean over test calls; oracle sw6 = ceiling)")
    table = {}
    for regime in ("same", "diff", "shift", "judge"):
        rs = [r for r in all_recs if r["regime"] == regime]
        if not rs:
            continue
        row = {}
        for m in MODES:
            row[m] = (sum(r["modes"][m][6] for r in rs) / len(rs) * 100,
                      sum(r["modes"][m][2] for r in rs) / len(rs) * 100)
        row["oracle6"] = sum(r["oracle"][6] for r in rs) / len(rs) * 100
        table[regime] = row
        cells = "  ".join(f"{m}:{row[m][0]:+6.2f}({row[m][1]:+6.2f})" for m in MODES)
        print(f"  {regime:6s} n={len(rs):2d}: {cells}  oracle:{row['oracle6']:+6.2f}")

    print("\n=== timing per v-call (local CPU)")
    timing_cases = [
        ("mini", grams_for_timing["mini"], 2, 256, (10, 128, 512, 1024)),
        ("synB", grams_for_timing["synB"], 8, 128, (128, 512, 1024)),
    ]
    trows = timing_block(timing_cases)

    print("\n=== carried state (bf16 Grams per distinct R)")
    sbytes = state_bytes(grams_for_timing)

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump({"cases": all_recs, "table": table, "timing": trows,
                   "state_bytes": sbytes}, f, indent=1)
    print("wrote results.json")


if __name__ == "__main__":
    main()
