"""T3c free-form smoothing: double-holdout experiment battery.

Synthetic regimes:
  share=1.0  channel gains drawn ONCE, shared by all calib+test samples
             (mini-like persistent structure; the judge shows test/calib
             channel-mean corr 0.64-0.94)
  share=0.7  partial sharing (test gains = rho*calib + sqrt(1-rho^2)*fresh)
  share=0.0  stock-synth iid regime (no structure; guard must reject)

Protocol per (config, seed):
  A = calib samples[:-1] (s fit; pipeline guards use calib[-1])
  B = independent test samples (shared structure, fresh tokens)
  score/case = (mse_std - mse_play)/mse_std vs exact alg1 baseline, x100

Usage: python dev/smooth/exp_smooth.py [--configs c2048_shared ...] [--seeds 8]
       [--arms base ff_icm ...] [--out results.jsonl]
"""
from __future__ import annotations

import argparse
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

E2M1_GRID = synth.E2M1_GRID


def load_sol():
    spec = importlib.util.spec_from_file_location(
        "_smooth_sol", os.path.join(HERE, "solution.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SOL = load_sol()


# ---------- shared-channel-structure synthetic generator --------------------

def _nvfp4_pair(x: torch.Tensor):
    T, C = x.shape
    xb = x.reshape(-1, 16)
    amax = xb.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    scale = ((amax / 6.0).to(torch.bfloat16).float()).clamp_min(1e-30)
    q = xb / scale
    idx = torch.bucketize(q.abs(), (E2M1_GRID[1:] + E2M1_GRID[:-1]) / 2.0)
    carrier = torch.sign(q) * E2M1_GRID[idx]
    return (carrier.reshape(T, -1).to(torch.bfloat16),
            scale.reshape(T, -1).to(torch.bfloat16))


def _gains(C, spread, gen):
    return torch.exp((torch.rand(C, generator=gen) - 0.5) * 2 * math.log(10.0) * spread)


def make_shared_group(seed, N, C, calib_T=(128, 512, 512), test_T=(128, 512),
                      spread=0.5, outlier_p=0.0, w_spread=0.3, share=1.0):
    """share=1.0: channel gains drawn once, ALL calib+test samples share them
    (mini-like persistent structure, judge test/calib corr 0.64-0.94).
    0<share<1: test gains = rho*calib + sqrt(1-rho^2)*fresh.
    share=0.0: EVERY sample (calib included) draws fresh gains -- the true
    iid regime (stock synth behaviour): nothing to fit, guard must reject."""
    gen = torch.Generator().manual_seed(seed)
    gx = _gains(C, spread, gen)
    gw = _gains(C, w_spread, gen)
    gx2 = _gains(C, spread, gen)              # fresh test-side gains
    if share > 0.0:
        if share < 1.0:
            lg = gx.log(); lg2 = gx2.log()
            lg_t = share * (lg - lg.mean()) + math.sqrt(1 - share ** 2) * (lg2 - lg2.mean())
            gx_test = (lg_t + lg.mean()).exp()
        else:
            gx_test = gx
    else:
        gx_test = gx2

    def make_act(T, gains):
        g = gains if share > 0.0 else _gains(C, spread, gen)
        x = (torch.randn(T, 1, generator=gen)
             * g.unsqueeze(0) * torch.randn(T, C, generator=gen))
        if outlier_p > 0:
            mask = torch.rand(T, C, generator=gen) < outlier_p
            x = x + mask.float() * torch.randn(T, C, generator=gen) * x.abs().amax() * 3
        return _nvfp4_pair(x)

    w = (torch.randn(N, 1, generator=gen) * gw.unsqueeze(0)
         * torch.randn(N, C, generator=gen)) * 0.05
    return {
        "weight": _nvfp4_pair(w),
        "calib_activation_list": [make_act(T, gx) for T in calib_T],
        "test_activation_list": [make_act(T, gx_test) for T in test_T],
    }


# ---------- scoring -----------------------------------------------------------

def score_group(group, extra_test=None):
    """Run one (arm-configured) calibration; return per-case pp scores."""
    w_ref = hif4.dequantize_nvfp4(*group["weight"])
    t0 = time.perf_counter()
    cal = SOL.hif4_calibration_and_quantize_weight(
        *group["weight"], group["calib_activation_list"])
    t_cal = time.perf_counter() - t0
    w_play = hif4.hif4_dequantize(cal["weight_params"])
    w_std = V.deq(V.quant_alg1(w_ref.float()))
    tests = list(group["test_activation_list"]) + list(extra_test or [])
    scores = []
    for pair in tests:
        x_ref = hif4.dequantize_nvfp4(*pair)
        ref = hif4.linear_ref(x_ref, w_ref)
        x_std = V.deq(V.quant_alg1(x_ref.float()))
        mse_std = ((hif4.linear_ref(x_std, w_std) - ref) ** 2).mean().item()
        p = SOL.hif4_dynamic_quantize_activation(pair[0], pair[1],
                                                 cal["activation_state"])
        mse_play = ((hif4.linear_ref(hif4.hif4_dequantize(p), w_play) - ref) ** 2).mean().item()
        scores.append((mse_std - mse_play) / mse_std * 100.0)
    dbg = dict(SOL.SMOOTH_DEBUG)
    st = cal["activation_state"]
    state_mb = sum(v.numel() * v.element_size() for v in st.values()
                   if isinstance(v, torch.Tensor)) / 2 ** 20
    s = st.get("s")
    sstats = (float(s.min()), float(s.max()), float(s.log().std())) if s is not None else None
    return {"scores": scores, "t_cal": t_cal, "dbg": dbg,
            "state_mb": state_mb, "s_stats": sstats}


# ---------- battery ------------------------------------------------------------

CONFIGS = [
    # name, C, N, spread, outlier_p, w_spread, share
    ("c1024_shared",    1024, 2048, 0.5, 0.0,   0.3, 1.0),
    ("c2048_shared",    2048, 2048, 0.5, 0.0,   0.3, 1.0),
    ("c4096_shared",    4096, 2048, 0.5, 0.0,   0.3, 1.0),
    ("c2048_shared_sp08", 2048, 2048, 0.8, 0.0, 0.3, 1.0),
    ("c2048_shared_outl", 2048, 2048, 0.5, 0.002, 0.3, 1.0),
    ("c2048_shared_wsp09", 2048, 2048, 0.5, 0.0, 0.9, 1.0),
    ("c4096_shared_flat", 4096, 2048, 0.15, 0.0, 0.1, 1.0),
    ("c2048_partial",   2048, 2048, 0.5, 0.0,   0.3, 0.7),
    ("c2048_iid",       2048, 2048, 0.5, 0.0,   0.3, 0.0),
]

ARMS = ["base", "ff_icm", "ff_bal", "mag_scan", "ff_icm_ng"]
# ff_icm_ng (guard off) only run on structure configs for diagnosis
NG_CONFIGS = {"c2048_shared", "c2048_iid", "c2048_partial", "c4096_shared"}


def run_battery(configs, seeds, arms, out_path):
    with open(out_path, "a", encoding="utf-8") as fh:
        for name, C, N, spread, outp, wspread, share in configs:
            for k in range(seeds):
                seed = 5100 + 131 * k + (sum(map(ord, name)) % 977)
                torch.manual_seed(0)
                group = make_shared_group(seed, N, C, spread=spread,
                                          outlier_p=outp, w_spread=wspread,
                                          share=share)
                row_base = None
                for arm in arms:
                    if arm == "ff_icm_ng" and name not in NG_CONFIGS:
                        continue
                    SOL.SMOOTH_MODE = "base" if arm == "base" else arm.replace("_ng", "")
                    SOL.SMOOTH_GUARD = arm != "ff_icm_ng"
                    SOL.SMOOTH_DEBUG.clear()
                    torch.manual_seed(0)
                    try:
                        r = score_group(group)
                    except Exception as exc:  # keep battery alive
                        fh.write(json.dumps({"cfg": name, "seed": seed, "arm": arm,
                                             "error": repr(exc)}) + "\n")
                        fh.flush()
                        continue
                    mean = sum(r["scores"]) / len(r["scores"])
                    rec = {"cfg": name, "seed": seed, "arm": arm, "C": C,
                           "mean_pp": mean, "scores": r["scores"],
                           "t_cal": round(r["t_cal"], 2), "dbg": r["dbg"],
                           "state_mb": round(r["state_mb"], 1),
                           "s_stats": [round(v, 3) for v in r["s_stats"]] if r["s_stats"] else None}
                    if arm == "base":
                        row_base = mean
                    rec["delta_pp"] = (mean - row_base) if row_base is not None else 0.0
                    rec["delta_pp"] = round(rec["delta_pp"], 3)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    print(f"[{name} seed{k}] {arm:10s} mean={mean:+.2f}pp "
                          f"delta={rec['delta_pp']:+.2f} cal={r['t_cal']:.1f}s "
                          f"acc={r['dbg'].get('accepted')}", flush=True)
    SOL.SMOOTH_MODE = "base"
    SOL.SMOOTH_GUARD = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--out", default=os.path.join(HERE, "results.jsonl"))
    args = ap.parse_args()
    cfgs = CONFIGS if not args.configs else [c for c in CONFIGS
                                             if c[0] in args.configs]
    run_battery(cfgs, args.seeds, args.arms, args.out)


if __name__ == "__main__":
    main()
