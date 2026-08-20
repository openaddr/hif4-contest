"""Experiment C: two-round alternating calibration for the Linear weight path.

Round 1 is a verbatim copy of S.hif4_calibration_and_quantize_weight (same
RNG consumption, so r1 artifacts match the shipped pipeline bit-for-bit):
weight GPTQ uses Hs = sum of UNQUANTIZED calibration-activation Grams.

Round 2: push each calib sample (calib[:-1], holdout convention) through the
production dynamic activation quantizer (state from round 1) to get quantized
activations X-hat, rebuild H-hat = sum X-hat^T X-hat, recompute Uw, and re-run
_gptq_quantize_values on w_final with the round-1 weight unit. Guard on the
same holdout rows the pipeline uses (xh_pick = tf(calib[-1] subsample)):
accept the new weights only if holdout MSE improves. If accepted, re-score
the 5 mini tests with the new weight and the UNCHANGED activation_state.
"""
from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "dev")
import exp_common as E  # noqa: E402

S = E.S


def calib_two_round(weight_quant, weight_scale, calib_activation_list, verbose=True):
    """Verbatim round-1 pipeline + round-2 quantized-activation Hessian GPTQ."""
    torch.manual_seed(0)  # deterministic calibration subsampling
    w = S.dequantize_nvfp4(weight_quant, weight_scale).float()
    R, C = w.shape
    ones_w = torch.ones(1, C, dtype=torch.float32)
    acts_raw = [S.dequantize_nvfp4(aq, as_).float() for aq, as_ in calib_activation_list]

    # ---- alpha smoothing search (verbatim) ----
    abs_sum = torch.zeros(C, dtype=torch.float32)
    sq_sum = torch.zeros(C, dtype=torch.float32)
    n_tok = 0
    a_big = None
    for a in acts_raw:
        abs_sum += a.abs().sum(dim=0)
        sq_sum += (a * a).sum(dim=0)
        n_tok += a.shape[0]
        if a_big is None or a.shape[0] > a_big.shape[0]:
            a_big = a
    m = (abs_sum / max(n_tok, 1)).clamp_min(1e-12)
    logm = m.log()
    logm = logm - logm.mean()
    rows = torch.randperm(R)[: min(R, 256)]
    best_alpha = 0.0
    best_loss = None
    for alpha in S.ALPHA_GRID:
        s_c = torch.exp(logm * alpha)
        wp = S._quant_weight_fast(w[rows] / s_c, torch.ones(1, C))
        wq = (wp["sign"] * wp["mant"] * wp["scale_lv3"] * wp["scale_lv2"]
              * wp["scale_factor"]).flatten(-4, -1) * s_c
        loss = ((a_big @ wq.T - a_big @ w[rows].T) ** 2).mean().item()
        if best_loss is None or loss < best_loss:
            best_loss, best_alpha = loss, alpha
    s = torch.exp(logm * best_alpha)
    w_s = w / s
    acts_s = [a * s for a in acts_raw]

    # ---- transform choice: {0: none, 1: rotation} (verbatim) ----
    mode = 0
    Uw = None
    xh_pick = None
    Hs_kept = None
    if R > 64 and len(acts_s) >= 2 and acts_s[-1].shape[0] >= 8:
        rsub = torch.randperm(R)[: min(R, 256)]
        xh_last = acts_s[-1]
        sub = torch.randperm(xh_last.shape[0])[: min(xh_last.shape[0], 128)]

        def tf(t, md):
            if md == 1:
                return S._rot_blocks(t)
            return t

        spaces = []
        hss = []
        for md in (0, 1):
            Hs = torch.zeros(C, C, dtype=torch.float32)
            for a in acts_s[:-1]:
                at = tf(a, md)
                Hs += at.T @ at
            spaces.append(S._upper_cholesky_inv(Hs))
            hss.append(Hs)
        xh_sub = xh_last[sub]
        cand = []
        for md, U in enumerate(spaces):
            if U is None:
                cand.append(float("inf"))
                continue
            w_rsub = tf(w_s[rsub], md)
            pp = S._quant_weight_fast(w_rsub, torch.ones(1, C))
            qq = S._gptq_quantize_values(w_rsub, S._params_unit_flat(pp), U)
            xt = tf(xh_sub, md)
            cand.append(((xt @ qq.T - xt @ w_rsub.T) ** 2).mean().item())
        mode = int(torch.tensor(cand).argmin().item())
        if mode == 1 and spaces[1] is None:
            mode = 0
        Uw = spaces[mode]
        Hs_kept = hss[mode]
        xh_pick = tf(xh_sub, mode).contiguous()

    def tf_final(t):
        if mode == 1:
            return S._rot_blocks(t)
        return t

    w_final = tf_final(w_s)

    # ---- anchored search + hold-out-guarded GPTQ (verbatim round 1) ----
    weight_params = S._quantize_weighted(w_final, ones_w)
    q_used = (weight_params["sign"] * weight_params["mant"] * weight_params["scale_lv3"]
              * weight_params["scale_lv2"] * weight_params["scale_factor"]).flatten(-4, -1)
    if xh_pick is not None and Uw is not None:
        unit = S._params_unit_flat(weight_params)
        q_g = S._gptq_quantize_values(w_final, unit, Uw)
        ref = xh_pick @ w_final.T
        mse_r = ((xh_pick @ q_used.T - ref) ** 2).mean().item()
        mse_g = ((xh_pick @ q_g.T - ref) ** 2).mean().item()
        if mse_g < mse_r:
            weight_params = S._values_to_params(q_g, weight_params)
            q_used = q_g.contiguous()

    # ---- activation-side GPTQ with act-order (verbatim) ----
    u_act = None
    gptq_act = 0
    order = None
    if xh_pick is not None:
        Ha = q_used.T @ q_used
        Ua = S._upper_cholesky_inv(Ha)
        if Ua is not None:
            order = torch.argsort(Ha.diagonal(), descending=True)
            Ua_o = S._upper_cholesky_inv(Ha[order][:, order])
            if Ua_o is not None:
                Ua = Ua_o
            else:
                order = None
            p_r = S._quantize_weighted(xh_pick, ones_w)
            xr = (p_r["sign"] * p_r["mant"] * p_r["scale_lv3"] * p_r["scale_lv2"]
                  * p_r["scale_factor"]).flatten(-4, -1)
            unit_x = S._params_unit_flat(p_r)
            if order is not None:
                xo_src = xh_pick[:, order]
                unit_src = unit_x[:, order]
            else:
                xo_src = xh_pick
                unit_src = unit_x
            xg = S._gptq_quantize_values(xo_src, unit_src, Ua)
            if order is not None:
                xg0 = torch.empty_like(xg)
                xg0[:, order] = xg
                xg = xg0
            ref2 = xh_pick @ w_final.T
            mse_ar = ((xr @ q_used.T - ref2) ** 2).mean().item()
            mse_ag = ((xg @ q_used.T - ref2) ** 2).mean().item()
            if mse_ag < mse_ar:
                u_act = Ua.contiguous()
                gptq_act = 1
            else:
                order = None

    activation_state = {
        "s": s.contiguous(),
        "mode": mode,
        "u_act": u_act,
        "g": gptq_act,
        "order": (order.contiguous() if (gptq_act == 1 and order is not None) else None),
    }
    # =================== end round 1 (verbatim) ===================

    # =================== round 2: quantized-act Hessian ===================
    diag = {"accepted": False, "mse_old": None, "mse_new": None,
            "H_ratio": None, "t_quant_acts": None, "t_gptq2": None}
    weight_params2 = None
    if xh_pick is not None:
        t2 = time.perf_counter()
        H_hat = torch.zeros(C, C, dtype=torch.float32)
        for i in range(len(calib_activation_list) - 1):
            aq, as_ = calib_activation_list[i]
            p = S.hif4_dynamic_quantize_activation(aq, as_, activation_state)
            xh = S._deq_params(p)
            H_hat += xh.T @ xh
        diag["t_quant_acts"] = time.perf_counter() - t2
        Uw2 = S._upper_cholesky_inv(H_hat)
        if Uw2 is not None:
            t3 = time.perf_counter()
            unit1 = S._params_unit_flat(weight_params)   # round-1 unit
            q_g2 = S._gptq_quantize_values(w_final, unit1, Uw2)
            diag["t_gptq2"] = time.perf_counter() - t3
            ref = xh_pick @ w_final.T
            diag["mse_old"] = ((xh_pick @ q_used.T - ref) ** 2).mean().item()
            diag["mse_new"] = ((xh_pick @ q_g2.T - ref) ** 2).mean().item()
            diag["accepted"] = diag["mse_new"] < diag["mse_old"]
            weight_params2 = S._values_to_params(q_g2, weight_params)
        if Hs_kept is not None:
            diag["H_ratio"] = (H_hat.diagonal().mean() / Hs_kept.diagonal().mean()).item()
        if verbose:
            print(f"  mode={mode} gptq_act={gptq_act} best_alpha={best_alpha}")
            print(f"  holdout MSE: old(r1)={diag['mse_old']:.4e}  new(r2)={diag['mse_new']:.4e}"
                  f"  ({100 * (1 - diag['mse_new'] / max(diag['mse_old'], 1e-30)):+.1f}%)"
                  f"  accepted={diag['accepted']}")
            print(f"  H-hat/H diag-mean ratio: {diag['H_ratio']:.4f}")
            print(f"  round2 timing: quant_acts={diag['t_quant_acts']:.2f}s  gptq2={diag['t_gptq2']:.2f}s")
    return {
        "weight_params": weight_params,
        "weight_params2": weight_params2,
        "activation_state": activation_state,
        "diag": diag,
    }


def main():
    std = E.std_baseline()

    torch.manual_seed(0)
    t0 = time.perf_counter()
    res = calib_two_round(*E.LIN["weight"], E.LIN["calib_activation_list"])
    dt_total = time.perf_counter() - t0
    print(f"two-round calibration total: {dt_total:.2f}s")

    sc_r1 = E.score(res["weight_params"], res["activation_state"], std)
    E.report("C: round1 (= baseline)", sc_r1, dt_total - (res["diag"]["t_quant_acts"] or 0)
             - (res["diag"]["t_gptq2"] or 0), base=list(E.BASELINE_SCORES))

    if res["weight_params2"] is not None:
        sc_r2 = E.score(res["weight_params2"], res["activation_state"], std)
        tag = "C: round2 weights" + ("" if res["diag"]["accepted"] else " FORCED (guard rejected)")
        E.report(tag, sc_r2, dt_total, base=list(sc_r1))
        E.report(tag + " vs SHIPPED baseline", sc_r2, dt_total, base=list(E.BASELINE_SCORES))
        print(f"guard accepted round2: {res['diag']['accepted']} "
              f"(holdout {res['diag']['mse_old']:.4e} -> {res['diag']['mse_new']:.4e})")
    else:
        print("round-2 could not run (no holdout / cholesky failed)")


if __name__ == "__main__":
    main()
