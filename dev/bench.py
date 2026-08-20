"""Ablation benchmark on synthetic judge-like data.

Loads example/solution/solution.py, applies textual patches per variant,
and scores each variant (plus the reference solution and norm7 baseline)
on synthetic linear/attention groups. Reports mean and worst-group scores.
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hif4  # noqa: E402
import synth  # noqa: E402
import variants as V  # noqa: E402

SOL_PATH = os.path.join(os.path.dirname(__file__), "..", "example", "solution", "solution.py")
REF_PATH = r"C:\Users\ning\Downloads\solution\solution.py"
BASE_SRC = open(SOL_PATH, encoding="utf-8").read()

VARIANTS = {
    "v3_full": [],
    "v3_no_lvr": [("LV_REFINE = True", "LV_REFINE = False")],
    "v3_no_wgt": [("USE_WEIGHTS = True", "USE_WEIGHTS = False")],
    "v3_no_asm": [("ALPHA_GRID = (0.0, 0.15, 0.3, 0.5)", "ALPHA_GRID = (0.0,)")],
    "v3_no_bsm": [("BETA_GRID = (0.0, 0.25)", "BETA_GRID = (0.0,)")],
    "v3_no_all": [
        ("LV_REFINE = True", "LV_REFINE = False"),
        ("USE_WEIGHTS = True", "USE_WEIGHTS = False"),
        ("ALPHA_GRID = (0.0, 0.15, 0.3, 0.5)", "ALPHA_GRID = (0.0,)"),
        ("BETA_GRID = (0.0, 0.25)", "BETA_GRID = (0.0,)"),
        ("GAMMA_GRID = (0.0, 0.15, 0.3, 0.5)", "GAMMA_GRID = (0.0,)"),
    ],
}


def load_variant(name, patches):
    src = BASE_SRC
    for old, new in patches:
        assert old in src, f"{name}: patch anchor missing: {old}"
        src = src.replace(old, new)
    spec = importlib.util.spec_from_file_location(f"var_{name}", SOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__file__"] = SOL_PATH
    exec(compile(src, f"<{name}>", "exec"), mod.__dict__)
    return mod


def score_linear(sol, group) -> float:
    w_ref = hif4.dequantize_nvfp4(*group["weight"])
    w_std = V.deq(V.quant_alg1(w_ref.float()))
    cal = sol.hif4_calibration_and_quantize_weight(*group["weight"], group["calib_activation_list"])
    w_play = hif4.hif4_dequantize(cal["weight_params"])
    total = 0.0
    for pair in group["test_activation_list"]:
        x_ref = hif4.dequantize_nvfp4(*pair)
        ref = hif4.linear_ref(x_ref, w_ref)
        x_std = V.deq(V.quant_alg1(x_ref.float()))
        mse_std = ((hif4.linear_ref(x_std, w_std) - ref) ** 2).mean().item()
        p = sol.hif4_dynamic_quantize_activation(pair[0], pair[1], cal["activation_state"])
        mse_play = ((hif4.linear_ref(hif4.hif4_dequantize(p), w_play) - ref) ** 2).mean().item()
        total += (mse_std - mse_play) / mse_std
    return total / len(group["test_activation_list"])


def score_attn(sol, group) -> float:
    qh, kvh, dh = group["q_num_heads"], group["kv_num_heads"], group["head_dim"]
    cal = sol.hif4_calibration_attention(group["calib"], qh, kvh, dh)
    total = 0.0
    for smp in group["test"]:
        q_ref = hif4.dequantize_nvfp4(*smp["q"])
        k_ref = hif4.dequantize_nvfp4(*smp["k"])
        v_ref = hif4.dequantize_nvfp4(*smp["v"])
        ref = hif4.attn_ref(q_ref, k_ref, v_ref, qh, kvh, dh)
        qs = V.deq(V.quant_alg1(q_ref.float()))
        ks = V.deq(V.quant_alg1(k_ref.float()))
        vs = V.deq(V.quant_alg1(v_ref.float()))
        mse_std = ((hif4.attn_ref(qs, ks, vs, qh, kvh, dh) - ref) ** 2).mean().item()
        pq = sol.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh, dh, cal["q_state"])
        pk = sol.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], kvh, dh, cal["k_state"])
        pv = sol.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], kvh, dh, cal["v_state"])
        out = hif4.attn_ref(hif4.hif4_dequantize(pq), hif4.hif4_dequantize(pk), hif4.hif4_dequantize(pv), qh, kvh, dh)
        mse_play = ((out - ref) ** 2).mean().item()
        total += (mse_std - mse_play) / mse_std
    return total / len(group["test"])


def main():
    lin_groups = [
        ("lin_plain_2k", synth.make_linear_group(1, 2048, 1024, spread=0.3)),
        ("lin_med_4k", synth.make_linear_group(2, 4096, 2048, spread=0.5)),
        ("lin_big", synth.make_linear_group(3, 8192, 2048, spread=0.4)),
        ("lin_outlier", synth.make_linear_group(4, 4096, 2048, spread=0.5, outlier_p=0.003)),
        ("lin_flat", synth.make_linear_group(5, 2048, 2048, spread=0.05)),
        ("lin_spiky_w", synth.make_linear_group(6, 4096, 2048, spread=0.5, w_spread=0.8)),
    ]
    attn_groups = [
        ("attn_gqa_256", synth.make_attn_group(11, 16, 2, 256, spread=0.4)),
        ("attn_mha_128", synth.make_attn_group(12, 8, 8, 128, spread=0.3)),
        ("attn_gqa_128", synth.make_attn_group(13, 32, 4, 128, spread=0.5)),
        ("attn_outlier", synth.make_attn_group(14, 16, 2, 256, spread=0.4, outlier_p=0.003)),
        ("attn_flat", synth.make_attn_group(15, 16, 2, 256, spread=0.05)),
        ("attn_big_seq", synth.make_attn_group(16, 16, 2, 256, spread=0.4, seqlens=(512, 1024))),
    ]

    mods = {name: load_variant(name, patches) for name, patches in VARIANTS.items()}
    ref_spec = importlib.util.spec_from_file_location("refsol", REF_PATH)
    refsol = importlib.util.module_from_spec(ref_spec)
    ref_spec.loader.exec_module(refsol)
    mods["reference"] = refsol

    results = {name: {"lin": [], "attn": []} for name in mods}
    for gname, g in lin_groups:
        for name, mod in mods.items():
            results[name]["lin"].append((gname, score_linear(mod, g)))
        print(f"[done] {gname}: " + " ".join(
            f"{n}={results[n]['lin'][-1][1]:+.3f}" for n in mods))
    for gname, g in attn_groups:
        for name, mod in mods.items():
            results[name]["attn"].append((gname, score_attn(mod, g)))
        print(f"[done] {gname}: " + " ".join(
            f"{n}={results[n]['attn'][-1][1]:+.3f}" for n in mods))

    print("\n===== SUMMARY (mean / worst over synthetic groups) =====")
    print(f"{'variant':<14}{'linear mean':>12}{'lin worst':>10}{'attn mean':>11}{'attn worst':>11}{'overall':>9}")
    for name in mods:
        lm = sum(v for _, v in results[name]["lin"]) / len(results[name]["lin"])
        lw = min(v for _, v in results[name]["lin"])
        am = sum(v for _, v in results[name]["attn"]) / len(results[name]["attn"])
        aw = min(v for _, v in results[name]["attn"])
        print(f"{name:<14}{lm:>12.3f}{lw:>10.3f}{am:>11.3f}{aw:>11.3f}{(lm+am)/2:>9.3f}")


if __name__ == "__main__":
    main()
