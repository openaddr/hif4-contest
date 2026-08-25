"""msel (multi-s per-call selection): STEP-1 worth-it measurement.

Question: with the ship's SINGLE weight_params fixed post-calibration, can any
calibration-fitted s variant (smallT ff_bal / mag_scan tau family) ever beat
the ship default s on a per-call basis, judged by the EXACT objective
    J(v) = <v, v@gw> - 2<v, x_ship@gwf>      (r1024v corrected coefficient)
which equals the judge MSE up to a candidate-independent constant?

Key structural fact (verified against hif4.py scoring): the judge computes
||v @ w_play^T - x_ref @ w_ref^T||^2 with w_play = dequant(weight_params)
FIXED at calibration.  The smoothing invariance (x*s)@(w/s)^T = x@w^T needs
the SAME s on both sides; the weight side is baked with s_ship, so a dynamic
call that walks variant s_v produces v ~= rot(x*s_v) which is scored against
rot(w/s_ship) -- carrying a mismatch penalty ||x diag(s_v/s_ship - 1) w^T||^2.

Measured per test call:
  J_def   : ship dynamic output (table+GPTQ+refine), exact J
  J_true  : variant light-path output (same machinery, s -> s_v), exact J
  J_lit   : the LITERAL proposal rule (cross term x_v@gwf -- wrong target)
  J_perf  : the variant's PERFECT output x_v itself (zero quant error) -- the
            best ANY variant-path quantizer could ever score on this axis
  mse_*   : direct judge MSE cross-check (J differences must track these)

Usage: python dev/msel/measure.py [smoke|run|rep]
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import importlib.util

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.dirname(HERE)
ROOT = os.path.dirname(DEV)
sys.path.insert(0, DEV)
import hif4 as H          # noqa: E402
import synth              # noqa: E402
import variants as V      # noqa: E402


def load_sol():
    spec = importlib.util.spec_from_file_location(
        "_msel_sol", os.path.join(HERE, "solution.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = load_sol()
E2M1_GRID = synth.E2M1_GRID

CALDIR = os.path.join(HERE, "calcache")
RES = os.path.join(HERE, "results")
os.makedirs(CALDIR, exist_ok=True)
os.makedirs(RES, exist_ok=True)


# ---------- shared-channel-structure synthetic generator (judge-like T mix) --
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
    return torch.exp((torch.rand(C, generator=gen) - 0.5)
                     * 2 * math.log(10.0) * spread)


def make_group(seed, N, C, calib_T=(10, 128, 512, 1024),
               test_T=(10, 128, 512, 1024, 2048), spread=0.5, outlier_p=0.0,
               w_spread=0.3, share=1.0):
    """share=1.0: persistent channel structure (mini-like, ff_bal regime).
    share=0.7: test gains partially fresh (the hypothesis's value source:
    test distribution differs from calib fit samples)."""
    gen = torch.Generator().manual_seed(seed)
    gx = _gains(C, spread, gen)
    gw = _gains(C, w_spread, gen)
    gx2 = _gains(C, spread, gen)
    if share > 0.0:
        if share < 1.0:
            lg = gx.log()
            lg2 = gx2.log()
            lg_t = share * (lg - lg.mean()) + math.sqrt(1 - share ** 2) * (lg2 - lg2.mean())
            gx_test = (lg_t + lg.mean()).exp()
        else:
            gx_test = gx
    else:
        gx_test = gx2

    def make_act(T, gains):
        g = gains
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


# (name, N, C, spread, outlier_p, w_spread, share)
CONFIGS = (
    ("c1024_shared",      2048, 1024, 0.5, 0.0,   0.3, 1.0),
    ("c2048_shared",      2048, 2048, 0.5, 0.0,   0.3, 1.0),
    ("c2048_sp09",        2048, 2048, 0.9, 0.0,   0.3, 1.0),
    ("c4096_shared",      2048, 4096, 0.5, 0.0,   0.3, 1.0),
    ("c2048_partial",     2048, 2048, 0.5, 0.0,   0.3, 0.7),
    ("c2048_wsp09",       2048, 2048, 0.5, 0.0,   0.9, 1.0),
)
SEEDS = (3101, 3102)


def jload(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def jsave(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def calibrate(name, group):
    cpath = os.path.join(CALDIR, f"{name}.pt")
    if os.path.exists(cpath):
        return torch.load(cpath, weights_only=True)
    torch.manual_seed(0)
    t0 = time.perf_counter()
    cal = M.hif4_calibration_and_quantize_weight(
        group["weight"][0], group["weight"][1], group["calib_activation_list"])
    cal_s = time.perf_counter() - t0
    dbg = dict(M.SMOOTH_DEBUG)
    out = {"cal": cal, "cal_s": cal_s, "sm dbg": dbg}
    torch.save(out, cpath)
    return out


# ---------- variant s fitting ------------------------------------------------

def build_fit_rows(acts):
    """Replicates _freeform_s round-robin row building for a given sample list."""
    rows = []
    budget = M.SMOOTH_FIT_ROWS
    per = max(1, budget // max(1, len(acts)))
    for a in acts:
        T = a.shape[0]
        if T > per:
            stride = T // per
            idx = torch.arange(0, T, stride)[:per]
        else:
            idx = torch.arange(T)
        rows.append(a[idx])
        budget -= idx.shape[0]
        if budget <= 0:
            break
    return torch.cat(rows, dim=0)[:M.SMOOTH_FIT_ROWS].contiguous()


def fit_variants(group, state):
    """Fit the non-default s variants on the SAME calib data the ship used.
    Returns {name: s} + diagnostics. Default arm = state['s'] (ship, exact)."""
    acts_raw = [H.dequantize_nvfp4(*p).float()
                for p in group["calib_activation_list"]]
    w = H.dequantize_nvfp4(*group["weight"]).float()
    R, C = w.shape
    gen = torch.Generator().manual_seed(7717 + R)
    wsub = w[torch.randperm(R, generator=gen)[: min(R, M.SMOOTH_W_ROWS)]].contiguous()
    gw_col = (w * w).sum(dim=0) + 1e-30

    idx_fit = list(range(len(acts_raw) - 1))          # calib[:-1] (guard excl.)
    idx_small = [i for i in idx_fit if acts_raw[i].shape[0] <= 128]
    xf_all = build_fit_rows([acts_raw[i] for i in idx_fit])
    xf_small = build_fit_rows([acts_raw[i] for i in idx_small])
    gx_all = (xf_all * xf_all).sum(dim=0) + 1e-30
    gx_small = (xf_small * xf_small).sum(dim=0) + 1e-30

    out = {}
    diag = {"n_small": len(idx_small), "rows_small": int(xf_small.shape[0]),
            "rows_all": int(xf_all.shape[0])}
    if xf_small.shape[0] >= 16:
        out["smallT"] = M._bal_search(xf_small, wsub, None, gw_col, gx_small)
    # mag_scan: parametric tau family, deploy-aware proxy pick (module branch)
    lm = 0.5 * gx_all.clamp_min(1e-30).log()
    lm = lm - lm.mean()
    best = None
    for tau in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0):
        s_try = (-tau * lm).clamp(-M.SMOOTH_LOGS_CLIP, M.SMOOTH_LOGS_CLIP).exp()
        s_try = s_try / torch.exp(s_try.log().mean())
        j = M._joint_proxy(s_try, xf_all, wsub)
        if best is None or j < best[0]:
            best = (j, s_try)
    out["mag"] = best[1]
    diag["mag_tau_j"] = best[0]
    return out, diag


# ---------- dynamic paths + exact J -------------------------------------------

def transformed(x_raw, s, mode):
    x = x_raw * s
    if mode == 1:
        x = M._rot_blocks(x)
    return x


def j_value(v, x_t, gw, gwf):
    """Exact output objective J(v) = <v, v@gw> - 2<v, x_t@gwf> (+const)."""
    xg = x_t @ gwf
    return (v * (v @ gw - 2.0 * xg)).sum().item()


def dyn_variant_values(x_raw, s_v, state):
    """The light variant path: EXACTLY the ship dynamic machinery with s->s_v
    (shared u_act/order/grams -- only the C-dim s vector differs, as the
    proposal specifies: 'all variants into state, one quantization walk each')."""
    mode = state.get("mode") or 0
    x = transformed(x_raw, s_v, mode)
    R, C = x.shape
    ones = torch.ones(1, C, dtype=torch.float32)
    p = M._quantize_weighted(x, ones)
    unit = M._params_unit_flat(p)
    values = None
    if state.get("g") == 1:
        u = state.get("u_act")
        order = state.get("order")
        if isinstance(u, torch.Tensor) and tuple(u.shape) == (C, C):
            if isinstance(order, torch.Tensor) and order.numel() == C:
                ol = order.long()
                q = M._gptq_quantize_values(x[:, ol], unit[:, ol], u.float())
                q0 = torch.empty_like(q)
                q0[:, ol] = q
                values = q0
            else:
                values = M._gptq_quantize_values(x, unit, u.float())
    tmax = state.get("tmax") or M.REFINE_T_MAX
    gw, gwf = state.get("gw"), state.get("gwf")
    if (R <= tmax and isinstance(gw, torch.Tensor)
            and isinstance(gwf, torch.Tensor)
            and tuple(gw.shape) == (C, C) and tuple(gwf.shape) == (C, C)):
        try:
            v0 = values if values is not None else M._deq_params(p)
            return M._refine_act_values(x, v0, unit, gw.float(), gwf.float()), x
        except Exception:
            pass
    return (values if values is not None else M._deq_params(p)), x


def run_group(name, group, out_path):
    out = jload(out_path)
    if name in out:
        print(f"[run] {name}: cached, skip")
        return out[name]
    cc = calibrate(name, group)
    cal = cc["cal"]
    st = cal["activation_state"]
    wp = cal["weight_params"]
    variants, vdiag = fit_variants(group, st)
    s_ship = st["s"].float()
    mode = st.get("mode") or 0
    grams = int(isinstance(st.get("gw"), torch.Tensor))
    # s-ratio diagnostics (mismatch penalty driver)
    srat = {}
    for vn, sv in variants.items():
        lr = (sv / s_ship).log()
        srat[vn] = {"rms_log_ratio": float(lr.pow(2).mean().sqrt()),
                    "max_abs_log_ratio": float(lr.abs().max())}
    w_ref = H.dequantize_nvfp4(*group["weight"])
    w_play = H.hif4_dequantize(wp)
    w_std = V.deq(V.quant_alg1(w_ref.float()))
    gw32 = st["gw"].float() if grams else None
    gwf32 = st["gwf"].float() if grams else None

    calls = []
    for pair in group["test_activation_list"]:
        x_raw = H.dequantize_nvfp4(*pair).float()
        R = x_raw.shape[0]
        x_ship = transformed(x_raw, s_ship, mode)
        ref = H.linear_ref(H.dequantize_nvfp4(*pair), w_ref)
        mse_std = ((H.linear_ref(V.deq(V.quant_alg1(x_raw)), w_std) - ref) ** 2).mean().item()
        # ship default dynamic (the real function, bit-exact deployment path)
        p_ship = M.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
        v_ship = M._deq_params(p_ship)
        mse_def = ((H.linear_ref(v_ship, w_play) - ref) ** 2).mean().item()
        j_def = j_value(v_ship, x_ship, gw32, gwf32) if grams else None
        row = {"T": int(R), "j_def": j_def, "mse_def": mse_def,
               "mse_std": mse_std, "pp": (mse_std - mse_def) / mse_std * 100,
               "variants": {}}
        for vn, sv in variants.items():
            v_v, x_v = dyn_variant_values(x_raw, sv, st)
            mse_v = ((H.linear_ref(v_v, w_play) - ref) ** 2).mean().item()
            e = {"j_true": j_value(v_v, x_ship, gw32, gwf32) if grams else None,
                 "j_lit": j_value(v_v, x_v, gw32, gwf32) if grams else None,
                 "j_perf": j_value(x_v, x_ship, gw32, gwf32) if grams else None,
                 "mse": mse_v}
            if j_def is not None and e["j_true"] is not None and j_def != 0.0:
                e["rel_j_true"] = (e["j_true"] - j_def) / abs(j_def)
                e["rel_j_lit"] = (e["j_lit"] - j_def) / abs(j_def)
                e["rel_j_perf"] = (e["j_perf"] - j_def) / abs(j_def)
            e["rel_mse"] = (mse_v - mse_def) / mse_def
            row["variants"][vn] = e
        # J<->MSE consistency check: dJ vs dMSE for the strongest variant
        if grams and variants:
            vn0 = sorted(variants)[0]
            e0 = row["variants"][vn0]
            if j_def is not None and e0["j_true"] is not None:
                dj = e0["j_true"] - j_def
                dm = e0["mse"] - mse_def
                scale = max(abs(mse_def), 1e-30)
                row["jm_consistency_relerr"] = abs(dj - dm) / scale if dm != 0 else 0.0
        calls.append(row)
        vr = {vn: f"{e['rel_j_true']:+.3e}" if e.get("rel_j_true") is not None else "n/a"
              for vn, e in row["variants"].items()}
        print(f"[{name}] T={R}: pp {row['pp']:+.2f} relJ {vr}")
        sys.stdout.flush()
    entry = {"cal_s": cc["cal_s"], "smooth_dbg": cc["sm dbg"], "mode": mode,
             "g": int(st.get("g") or 0), "grams": grams,
             "tmax": int(st.get("tmax") or M.REFINE_T_MAX),
             "s_ratio": srat, "fit_diag": vdiag, "calls": calls}
    out[name] = entry
    jsave(out_path, out)
    return entry


def mini_group():
    lin = torch.load(os.path.join(ROOT, "example", "mini_sample", "linear.pt"),
                     weights_only=True, map_location="cpu")[0]
    return {"weight": lin["weight"],
            "calib_activation_list": lin["calib_activation_list"],
            "test_activation_list": lin["test_activation_list"]}


def run(which="all"):
    out_path = os.path.join(RES, "step1.json")
    jobs = []
    if which in ("all", "mini"):
        jobs.append(("mini", mini_group()))
    if which in ("all", "synth"):
        for cn, N, C, sp, op, wsp, sh in CONFIGS:
            for seed in SEEDS:
                g = make_group(seed, N, C, spread=sp, outlier_p=op,
                               w_spread=wsp, share=sh)
                jobs.append((f"{cn}_s{seed}", g))
    if which == "smoke":
        g = make_group(3001, 1024, 1024)
        jobs.append(("smoke_c1024", g))
    for name, group in jobs:
        run_group(name, group, out_path)
    print("[run] done")


def rep():
    out = jload(os.path.join(RES, "step1.json"))
    if not out:
        print("no results")
        return
    print(f"{'group':<22} {'mode':>4} {'g':>2} {'tmx':>4} "
          f"{'smallT rms':>10} {'mag rms':>9} | per-call rel J_true (variant best vs default)")
    tot_calls = 0
    tot_wins = 0
    tot_lit_div = 0
    worst_rel = -float("inf")
    for name in sorted(out):
        e = out[name]
        if name == "smoke_c1024":
            continue
        sr = e["s_ratio"]
        s1 = sr.get("smallT", {}).get("rms_log_ratio", float("nan"))
        s2 = sr.get("mag", {}).get("rms_log_ratio", float("nan"))
        parts = []
        for row in e["calls"]:
            best = None
            for vn, ev in row["variants"].items():
                r = ev.get("rel_j_true")
                if r is not None and (best is None or r < best):
                    best = r
            lit_div = any(ev.get("rel_j_lit", 0) is not None and ev["rel_j_lit"] < -1e-4
                          for ev in row["variants"].values())
            tot_calls += 1
            if best is not None:
                if best < -1e-4:
                    tot_wins += 1
                worst_rel = max(worst_rel, best)
            if lit_div:
                tot_lit_div += 1
            b = best if best is not None else float("nan")
            mark = "W" if (b is not None and b < -1e-4) else "."
            parts.append(f"T{row['T']}:{b:+.1e}{mark}")
        print(f"{name:<22} {e['mode']:>4} {e['g']:>2} {e['tmax']:>4} "
              f"{s1:>10.3f} {s2:>9.3f} | {' '.join(parts)}")
    print(f"\npooled: calls={tot_calls} variant-wins (rel J_true < -1e-4) = "
          f"{tot_wins} ({100.0 * tot_wins / max(tot_calls, 1):.1f}%)  "
          f"worst(best rel J_true) = {worst_rel:+.3e}")
    print(f"literal-rule divergences (rel J_lit < -1e-4): {tot_lit_div}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "rep"
    if mode == "smoke":
        run("smoke")
    elif mode == "run":
        run("all")
    elif mode == "mini":
        run("mini")
    elif mode == "synth":
        run("synth")
    else:
        rep()


if __name__ == "__main__":
    main()
