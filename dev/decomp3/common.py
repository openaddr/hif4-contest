"""decomp3 shared harness: v40 solution loading, mini data, end-to-end scoring.

All "value" measurements go through the mini_holdout.py scoring convention:
    score(pp) = (mse_std - mse_play) / mse_std * 100
    mse_std  = exact Alg.1 baseline (both sides alg1-quantized) vs exact ref
    mse_play = player output vs exact ref
Baseline under study = example/solution/solution.py (v40), loaded read-only.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "dev"))
import hif4          # noqa: E402
import variants as V  # noqa: E402

SOL_PATH = os.path.join(ROOT, "example", "solution", "solution.py")


def load_sol(name="_decomp3_sol", path=SOL_PATH):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_mini():
    lin = torch.load(os.path.join(ROOT, "example", "mini_sample", "linear.pt"),
                     weights_only=True, map_location="cpu")[0]
    att = torch.load(os.path.join(ROOT, "example", "mini_sample", "attn.pt"),
                     weights_only=True, map_location="cpu")[0]
    return lin, att


# --------------------------------------------------------------------------
# linear scoring
# --------------------------------------------------------------------------

def linear_case_mses(x_play, w_play, x_ref, w_ref, x_std, w_std):
    """x/w tensors are 2D float (dequantized values). Returns dict of MSEs."""
    ref = hif4.linear_ref(x_ref, w_ref)
    mse_std = ((hif4.linear_ref(x_std, w_std) - ref) ** 2).mean().item()
    out = {"mse_std": mse_std}
    variants_ = {
        "play": (x_play, w_play),
        "w_exact": (x_play, w_ref),
        "x_exact": (x_ref, w_play),
        "both_exact": (x_ref, w_ref),
    }
    for k, (xv, wv) in variants_.items():
        out[f"mse_{k}"] = ((hif4.linear_ref(xv, wv) - ref) ** 2).mean().item()
    return out, ref


def run_linear_tests(SOL, cal, tests, weight_pair, extra_out_hook=None):
    """Run the ship dynamic path on each test; return per-case records with
    staged values (table / gptq / refined) and side-swap MSEs."""
    w_ref = hif4.dequantize_nvfp4(*weight_pair)
    w_std = V.deq(V.quant_alg1(w_ref.float()))
    w_play = hif4.hif4_dequantize(cal["weight_params"])
    st = cal["activation_state"]
    recs = []
    for pair in tests:
        x_ref = hif4.dequantize_nvfp4(*pair)
        # ship dynamic call (bit-exact player path)
        p = SOL.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
        x_play = hif4.hif4_dequantize(p)
        mses, ref = linear_case_mses(x_play, w_play, x_ref, w_ref,
                                     V.deq(V.quant_alg1(x_ref.float())), w_std)
        rec = {"T": pair[0].shape[0], **mses}
        if extra_out_hook is not None:
            rec.update(extra_out_hook(SOL, pair, st, ref, x_ref, w_ref, w_play))
        recs.append(rec)
    for r in recs:
        r["pp_play"] = (r["mse_std"] - r["mse_play"]) / r["mse_std"] * 100.0
        r["pp_w_exact"] = (r["mse_std"] - r["mse_w_exact"]) / r["mse_std"] * 100.0
        r["pp_x_exact"] = (r["mse_std"] - r["mse_x_exact"]) / r["mse_std"] * 100.0
    return recs


def summarize(recs, keys=("pp_play", "pp_w_exact", "pp_x_exact")):
    out = {}
    for k in keys:
        vals = [r[k] for r in recs if k in r]
        out[k] = round(sum(vals) / len(vals), 3) if vals else None
    return out


# --------------------------------------------------------------------------
# attention scoring
# --------------------------------------------------------------------------

def clone_state(st):
    if isinstance(st, torch.Tensor):
        return st.clone()
    if isinstance(st, dict):
        return {k: clone_state(v) for k, v in st.items()}
    return st


def run_attn_tests(SOL, cal, tests, QH, KVH, DH):
    """Ship dynamic path per test; side-swap outputs against exact ref."""
    recs = []
    for smp in tests:
        q_ref = hif4.dequantize_nvfp4(*smp["q"])
        k_ref = hif4.dequantize_nvfp4(*smp["k"])
        v_ref = hif4.dequantize_nvfp4(*smp["v"])
        ref = hif4.attn_ref(q_ref, k_ref, v_ref, QH, KVH, DH)
        mse_std = ((hif4.attn_ref(V.deq(V.quant_alg1(q_ref.float())),
                                  V.deq(V.quant_alg1(k_ref.float())),
                                  V.deq(V.quant_alg1(v_ref.float())), QH, KVH, DH)
                    - ref) ** 2).mean().item()
        pq = SOL.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], QH, DH,
                                         clone_state(cal["q_state"]))
        pk = SOL.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], KVH, DH,
                                         clone_state(cal["k_state"]))
        pv = SOL.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], KVH, DH,
                                         clone_state(cal["v_state"]))
        q_play = hif4.hif4_dequantize(pq)
        k_play = hif4.hif4_dequantize(pk)
        v_play = hif4.hif4_dequantize(pv)
        outs = {
            "mse_play": hif4.attn_ref(q_play, k_play, v_play, QH, KVH, DH),
            "mse_qe": hif4.attn_ref(q_ref, k_play, v_play, QH, KVH, DH),
            "mse_ke": hif4.attn_ref(q_play, k_ref, v_play, QH, KVH, DH),
            "mse_ve": hif4.attn_ref(q_play, k_play, v_ref, QH, KVH, DH),
            "mse_qke": hif4.attn_ref(q_ref, k_ref, v_play, QH, KVH, DH),
            "mse_qve": hif4.attn_ref(q_ref, k_play, v_ref, QH, KVH, DH),
            "mse_kve": hif4.attn_ref(q_play, k_ref, v_ref, QH, KVH, DH),
        }
        rec = {"T": smp["q"][0].shape[0], "mse_std": mse_std}
        for k, o in outs.items():
            rec[k] = ((o - ref) ** 2).mean().item()
            rec["pp_" + k[4:]] = (mse_std - rec[k]) / mse_std * 100.0
        recs.append(rec)
    return recs


def fmt_recs(recs, keys):
    hdr = ["T"] + list(keys)
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in recs:
        row = [str(r["T"])] + [f"{r[k]:+.2f}" if k in r else "-" for k in keys]
        lines.append("| " + " | ".join(row) + " |")
    mean = {k: round(sum(r[k] for r in recs) / len(recs), 3)
            for k in keys if all(k in r for r in recs)}
    lines.append("| **mean** | " + " | ".join(f"{mean[k]:+.3f}" for k in keys) + " |")
    return "\n".join(lines)
