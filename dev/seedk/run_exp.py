"""seedk stage A: rotation sign-seed variance through the FULL v33 stack.

For K=16 seeds per side (linear block-rotation base, attention Q/K rotation
base), run the real pipeline (calibration + dynamic quantization on test
samples) on the mini groups and >=4 synthetic shapes, and record:
  - per-case score s = (mse_std - mse_play)/mse_std  (diag3 convention)
  - cross-seed distribution per group (std / range / best-vs-median, pp/case)
  - bit-identity digests of every produced tensor (weight params, state,
    dynamic outputs) to test the equivariance (no-op) conjecture at the
    bit level
  - wall-clock calibration time per seed (cost section)

Incremental JSON dump after every (group, seed) so an interrupt never loses
more than one cell.
"""
from __future__ import annotations

import hashlib
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
import synth  # noqa: E402
import variants as V  # noqa: E402

ORIG_SOL = os.path.join(ROOT, "example", "solution", "solution.py")
COPY_SOL = os.path.join(HERE, "solution.py")
MINI = os.path.join(ROOT, "example", "mini_sample")
RESULTS = os.path.join(HERE, "results_stageA.json")

LIN_BASES = [777] + [1001 + i for i in range(15)]          # K=16, incl. shipped
ATTN_BASES = [0xA5A5] + [2001 + i for i in range(15)]      # K=16, incl. shipped

_mod_n = [0]


def load_mod(path, base_lin=None, base_attn=None):
    _mod_n[0] += 1
    spec = importlib.util.spec_from_file_location(f"_seedk_sol_{_mod_n[0]}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if base_lin is not None:
        mod._ROT_LIN_SEED_BASE = base_lin
    if base_attn is not None:
        mod._ROT_ATTN_SEED_BASE = base_attn
    return mod


def tdigest(t):
    if t is None:
        return "None"
    if not isinstance(t, torch.Tensor):
        return json.dumps(t, sort_keys=True)
    tc = t.detach().contiguous().cpu()
    if tc.dtype in (torch.bfloat16, torch.float16):
        tc = tc.view(torch.int16)
    return hashlib.sha256(tc.numpy().tobytes()).hexdigest()[:16]


def state_digest(st):
    if not isinstance(st, dict):
        return tdigest(st)
    return hashlib.sha256("|".join(
        f"{k}={state_digest(v)}" for k, v in sorted(st.items()))
        .encode()).hexdigest()[:16]


def params_digest(p):
    keys = ("scale_factor", "scale_lv2", "scale_lv3", "sign", "mant")
    return "|".join(tdigest(p[k]) for k in keys)


# ---- group builders -------------------------------------------------------

def mini_groups():
    lin = torch.load(os.path.join(MINI, "linear.pt"), weights_only=True,
                     map_location="cpu")
    att = torch.load(os.path.join(MINI, "attn.pt"), weights_only=True,
                     map_location="cpu")
    return [("mini_linear", "lin", lin[0]), ("mini_attn", "attn", att[0])]


def synth_groups():
    out = []

    def lin(name, seed, C, N, calib, test, spread, outp, wsp):
        g = synth.make_linear_group(seed, N, C, tokens=calib, spread=spread,
                                    outlier_p=outp, w_spread=wsp)
        g2 = synth.make_linear_group(seed + 50000, N, C, tokens=test,
                                     spread=spread, outlier_p=outp,
                                     w_spread=wsp)
        g["test_activation_list"] = g2["test_activation_list"]
        out.append((name, "lin", g))

    def attn(name, seed, qh, kvh, dh, seqlens, spread, outp):
        g = synth.make_attn_group(seed, qh, kvh, dh, seqlens=seqlens,
                                  spread=spread, outlier_p=outp)
        g2 = synth.make_attn_group(seed + 50000, qh, kvh, dh, seqlens=seqlens,
                                   spread=spread, outlier_p=outp)
        g["test"] = g2["test"]
        out.append((name, "attn", g))

    lin("syn_lin_a_2048x4096", 4101, 2048, 4096, (10, 128, 512), (128, 512), 0.5, 0.0, 0.3)
    lin("syn_lin_b_1024x8192_spiky", 4102, 1024, 8192, (10, 128, 512), (128, 512), 0.6, 0.002, 0.6)
    attn("syn_attn_a_8x2x128", 4201, 8, 2, 128, (128, 512), 0.4, 0.0)
    attn("syn_attn_b_16x4x64_spiky", 4202, 16, 4, 64, (128, 512), 0.6, 0.002)
    return out


# ---- scoring (diag3 convention) -------------------------------------------

def run_linear(mod, group):
    t0 = time.perf_counter()
    cal = mod.hif4_calibration_and_quantize_weight(
        group["weight"][0], group["weight"][1], group["calib_activation_list"])
    t_cal = time.perf_counter() - t0
    w_ref = hif4.dequantize_nvfp4(*group["weight"])
    w_play = hif4.hif4_dequantize(cal["weight_params"])
    w_std = V.deq(V.quant_alg1(w_ref.float()))
    cases = []
    digests = [params_digest(cal["weight_params"]),
               state_digest(cal["activation_state"])]
    st = cal["activation_state"]
    flags = {"mode": st.get("mode"), "g": st.get("g"),
             "has_gw": st.get("gw") is not None}
    for pair in group["test_activation_list"]:
        x_ref = hif4.dequantize_nvfp4(*pair)
        ref = hif4.linear_ref(x_ref, w_ref)
        x_std = V.deq(V.quant_alg1(x_ref.float()))
        mse_std = ((hif4.linear_ref(x_std, w_std) - ref) ** 2).mean().item()
        p = mod.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
        x_play = hif4.hif4_dequantize(p)
        mse_play = ((hif4.linear_ref(x_play, w_play) - ref) ** 2).mean().item()
        cases.append((mse_std - mse_play) / mse_std)
        digests.append(tdigest(x_play))
    return cases, t_cal, digests, flags


def run_attn(mod, group):
    qh, kvh, dh = group["q_num_heads"], group["kv_num_heads"], group["head_dim"]
    t0 = time.perf_counter()
    acal = mod.hif4_calibration_attention(
        group["calib"], qh, kvh, dh)
    t_cal = time.perf_counter() - t0
    cases = []
    digests = [state_digest(acal["q_state"]), state_digest(acal["k_state"]),
               state_digest(acal["v_state"])]
    flags = {"rot": acal["q_state"].get("rot"), "gq": acal["q_state"].get("gq")}
    for smp in group["test"]:
        q_ref = hif4.dequantize_nvfp4(*smp["q"])
        k_ref = hif4.dequantize_nvfp4(*smp["k"])
        v_ref = hif4.dequantize_nvfp4(*smp["v"])
        ref = hif4.attn_ref(q_ref, k_ref, v_ref, qh, kvh, dh)
        qs = V.deq(V.quant_alg1(q_ref.float()))
        ks = V.deq(V.quant_alg1(k_ref.float()))
        vs = V.deq(V.quant_alg1(v_ref.float()))
        mse_std = ((hif4.attn_ref(qs, ks, vs, qh, kvh, dh) - ref) ** 2).mean().item()
        pq = mod.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh, dh,
                                         acal["q_state"])
        pk = mod.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], kvh, dh,
                                         acal["k_state"])
        pv = mod.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], kvh, dh,
                                         acal["v_state"])
        out = hif4.attn_ref(hif4.hif4_dequantize(pq),
                            hif4.hif4_dequantize(pk),
                            hif4.hif4_dequantize(pv), qh, kvh, dh)
        mse_play = ((out - ref) ** 2).mean().item()
        cases.append((mse_std - mse_play) / mse_std)
        digests += [tdigest(hif4.hif4_dequantize(pq)),
                    tdigest(hif4.hif4_dequantize(pk)),
                    tdigest(hif4.hif4_dequantize(pv))]
    return cases, t_cal, digests, flags


# ---- sanity: copy == original at default seeds ----------------------------

def sanity_copy_is_original():
    print("[sanity] copy vs original at default seeds (bit-identity)")
    ok = True
    for name, kind, g in mini_groups():
        mo = load_mod(ORIG_SOL)
        mc = load_mod(COPY_SOL)
        if kind == "lin":
            _, _, d_o, f_o = run_linear(mo, g)
            _, _, d_c, f_c = run_linear(mc, g)
        else:
            _, _, d_o, f_o = run_attn(mo, g)
            _, _, d_c, f_c = run_attn(mc, g)
        same = (d_o == d_c) and (f_o == f_c)
        ok &= same
        print(f"  {name}: digests {'IDENTICAL' if d_o == d_c else 'DIFFER'}"
              f" flags {f_o} vs {f_c} -> {'PASS' if same else 'FAIL'}")
    return ok


# ---- main sweep ------------------------------------------------------------

def sweep():
    groups = mini_groups() + synth_groups()
    try:
        res = json.load(open(RESULTS))
    except Exception:
        res = {}
    for name, kind, g in groups:
        if name in res and len(res[name]["seeds"]) >= 16:
            print(f"[sweep] {name}: already complete, skip")
            continue
        res[name] = {"kind": kind, "seeds": {}}
        bases = LIN_BASES if kind == "lin" else ATTN_BASES
        for base in bases:
            t0 = time.perf_counter()
            mod = load_mod(COPY_SOL, base_lin=base, base_attn=base)
            if kind == "lin":
                cases, t_cal, digs, flags = run_linear(mod, g)
            else:
                cases, t_cal, digs, flags = run_attn(mod, g)
            res[name]["seeds"][str(base)] = {
                "cases": cases, "mean_pp": 100.0 * sum(cases) / len(cases),
                "t_cal": t_cal, "digest": hashlib.sha256(
                    "|".join(digs).encode()).hexdigest()[:16], "flags": flags,
            }
            json.dump(res, open(RESULTS, "w"), indent=1)
            print(f"[sweep] {name} seed={base}: mean={100*sum(cases)/len(cases):+.4f}pp "
                  f"t_cal={t_cal:.1f}s flags={flags} "
                  f"(cell {time.perf_counter()-t0:.1f}s)", flush=True)
    print("[sweep] all cells done")


def report():
    res = json.load(open(RESULTS))
    print("\n=== stage A report ===")
    for name, r in res.items():
        seeds = r["seeds"]
        means = sorted(v["mean_pp"] for v in seeds.values())
        n = len(means)
        med = means[n // 2] if n % 2 else 0.5 * (means[n // 2 - 1] + means[n // 2])
        import statistics as st
        sd = st.pstdev(means)
        dset = {v["digest"] for v in seeds.values()}
        fset = {json.dumps(v["flags"], sort_keys=True) for v in seeds.values()}
        print(f"{name} [{r['kind']}] K={n}: mean={st.fmean(means):+.4f}pp "
              f"std={sd:.6g}pp range={(max(means)-min(means)):.6g}pp "
              f"best-med={(max(means)-med):+.6g}pp "
              f"distinct_digests={len(dset)} distinct_flags={len(fset)} "
              f"t_cal_avg={st.fmean([v['t_cal'] for v in seeds.values()]):.1f}s")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("sanity", "all"):
        if not sanity_copy_is_original():
            print("[sanity] FAILED - abort")
            sys.exit(1)
    if mode in ("sweep", "all"):
        sweep()
    if mode in ("report", "all", "sweep"):
        report()
