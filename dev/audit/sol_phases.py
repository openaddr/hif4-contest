"""Instrumented re-implementation of solution.py's three entry points for
per-phase CPU attribution. All worker functions (_quant_chunk, _gptq_*,
_refine_*, dequantize_nvfp4, ...) are imported UNMODIFIED from the real
solution module; only the drivers are re-written with phase timers.

prof.py verify -- asserts every output tensor is BIT-IDENTICAL (torch.equal)
to the original before any timing number is trusted.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import time
from typing import Any

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SOL_PATH = os.path.join(_ROOT, "example", "solution", "solution.py")
_spec = importlib.util.spec_from_file_location("_audit_sol_base", _SOL_PATH)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

PHASES: dict[str, float] = {}


@contextlib.contextmanager
def _t(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        PHASES[name] = PHASES.get(name, 0.0) + (time.perf_counter() - t0)


# re-exported state used by dynamic v compensation
_QKV_CARRY = S._QKV_CARRY

dequantize_nvfp4 = S.dequantize_nvfp4
_quant_weight_fast = S._quant_weight_fast
_quantize_weighted = S._quantize_weighted
_gptq_quantize_values = S._gptq_quantize_values
_params_unit_flat = S._params_unit_flat
_values_to_params = S._values_to_params
_deq_params = S._deq_params
_upper_cholesky_inv = S._upper_cholesky_inv
_rot_blocks = S._rot_blocks
_make_R = S._make_R
_attention_out = S._attention_out
_quant_chunk = S._quant_chunk
_refine_act_values = S._refine_act_values
_dyn_table = S._dyn_table
_v_compensate = S._v_compensate
_gptq_quantize_batched = S._gptq_quantize_batched

ALPHA_GRID = S.ALPHA_GRID
REFINE_MAX_C = S.REFINE_MAX_C
REFINE_T_MAX = S.REFINE_T_MAX


# =============================================================================
# 1. Linear calibration + weight quantization  (mirrors solution.py v18)
# =============================================================================

def hif4_calibration_and_quantize_weight(weight_quant, weight_scale,
                                         calib_activation_list) -> dict[str, Any]:
    torch.manual_seed(0)
    with _t("cal.dequant"):
        w = dequantize_nvfp4(weight_quant, weight_scale).float()
        R, C = w.shape
        ones_w = torch.ones(1, C, dtype=torch.float32)
        acts_raw = [dequantize_nvfp4(aq, as_).float()
                    for aq, as_ in calib_activation_list]

    # ---- alpha smoothing search ----
    with _t("cal.alpha_search"):
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
        for alpha in ALPHA_GRID:
            s = torch.exp(logm * alpha)
            with _t("cal.alpha_search.quant_fast"):
                wp = _quant_weight_fast(w[rows] / s, torch.ones(1, C))
                wq = (wp["sign"] * wp["mant"] * wp["scale_lv3"] * wp["scale_lv2"]
                      * wp["scale_factor"]).flatten(-4, -1) * s
            with _t("cal.alpha_search.eval"):
                loss = ((a_big @ wq.T - a_big @ w[rows].T) ** 2).mean().item()
            if best_loss is None or loss < best_loss:
                best_loss, best_alpha = loss, alpha
        s = torch.exp(logm * best_alpha)
        w_s = w / s
        acts_s = [a * s for a in acts_raw]

    # ---- transform choice ----
    mode = 0
    Uw = None
    xh_pick = None
    with _t("cal.xform_choice"):
        if R > 64 and len(acts_s) >= 2 and acts_s[-1].shape[0] >= 8:
            rsub = torch.randperm(R)[: min(R, 256)]
            xh_last = acts_s[-1]
            sub = torch.randperm(xh_last.shape[0])[: min(xh_last.shape[0], 128)]

            def tf(t, md):
                if md == 1:
                    return _rot_blocks(t)
                return t

            spaces = []
            for md in (0, 1):
                with _t("cal.xform_choice.gram" + ("_rot" if md else "")):
                    Hs = torch.zeros(C, C, dtype=torch.float32)
                    for a in acts_s[:-1]:
                        at = tf(a, md)
                        Hs += at.T @ at
                with _t("cal.xform_choice.chol"):
                    spaces.append(_upper_cholesky_inv(Hs))
            xh_sub = xh_last[sub]
            cand = []
            for md, U in enumerate(spaces):
                if U is None:
                    cand.append(float("inf"))
                    continue
                with _t("cal.xform_choice.proxy"):
                    w_rsub = tf(w_s[rsub], md)
                    pp = _quant_weight_fast(w_rsub, torch.ones(1, C))
                    qq = _gptq_quantize_values(w_rsub, _params_unit_flat(pp), U)
                    xt = tf(xh_sub, md)
                    cand.append(((xt @ qq.T - xt @ w_rsub.T) ** 2).mean().item())
            mode = int(torch.tensor(cand).argmin().item())
            if mode == 1 and spaces[1] is None:
                mode = 0
            Uw = spaces[mode]
            xh_pick = tf(xh_sub, mode).contiguous()

    def tf_final(t):
        if mode == 1:
            return _rot_blocks(t)
        return t

    with _t("cal.weight_rot"):
        w_final = tf_final(w_s)

    # ---- anchored search ----
    with _t("cal.weight_quant"):
        weight_params = _quantize_weighted(w_final, ones_w)
    with _t("cal.q_used_flatten"):
        q_used = (weight_params["sign"] * weight_params["mant"]
                  * weight_params["scale_lv3"] * weight_params["scale_lv2"]
                  * weight_params["scale_factor"]).flatten(-4, -1)
    if xh_pick is not None and Uw is not None:
        with _t("cal.weight_gptq"):
            unit = _params_unit_flat(weight_params)
            q_g = _gptq_quantize_values(w_final, unit, Uw)
            ref = xh_pick @ w_final.T
            mse_r = ((xh_pick @ q_used.T - ref) ** 2).mean().item()
            mse_g = ((xh_pick @ q_g.T - ref) ** 2).mean().item()
            if mse_g < mse_r:
                weight_params = _values_to_params(q_g, weight_params)
                q_used = q_g.contiguous()

    # ---- activation-side GPTQ search ----
    u_act = None
    gptq_act = 0
    order = None
    if xh_pick is not None:
        with _t("cal.actgptq.gram"):
            Ha = q_used.T @ q_used
        with _t("cal.actgptq.chol"):
            Ua = _upper_cholesky_inv(Ha)
        if Ua is not None:
            with _t("cal.actgptq.order"):
                order = torch.argsort(Ha.diagonal(), descending=True)
                Ua_o = _upper_cholesky_inv(Ha[order][:, order])
            if Ua_o is not None:
                Ua = Ua_o
            else:
                order = None
            with _t("cal.actgptq.anchor_quant"):
                p_r = _quantize_weighted(xh_pick, ones_w)
                xr = (p_r["sign"] * p_r["mant"] * p_r["scale_lv3"] * p_r["scale_lv2"]
                      * p_r["scale_factor"]).flatten(-4, -1)
                unit_x = _params_unit_flat(p_r)
            if order is not None:
                xo_src = xh_pick[:, order]
                unit_src = unit_x[:, order]
            else:
                xo_src = xh_pick
                unit_src = unit_x
            with _t("cal.actgptq.gptq"):
                xg = _gptq_quantize_values(xo_src, unit_src, Ua)
            if order is not None:
                xg0 = torch.empty_like(xg)
                xg0[:, order] = xg
                xg = xg0
            with _t("cal.actgptq.eval"):
                ref2 = xh_pick @ w_final.T
                mse_ar = ((xr @ q_used.T - ref2) ** 2).mean().item()
                mse_ag = ((xg @ q_used.T - ref2) ** 2).mean().item()
            if mse_ag < mse_ar:
                u_act = Ua.contiguous()
                gptq_act = 1
            else:
                order = None

    # ---- Gram carries ----
    gw = gwf = None
    if C <= REFINE_MAX_C:
        try:
            with _t("cal.gram_carry"):
                gw = (q_used.T @ q_used).to(torch.bfloat16)
                gwf = (w_final.T @ q_used).to(torch.bfloat16)
        except Exception:
            gw = gwf = None

    activation_state = {
        "s": s.contiguous(),
        "mode": mode,
        "u_act": u_act,
        "g": gptq_act,
        "order": (order.contiguous() if (gptq_act == 1 and order is not None) else None),
        "gw": gw,
        "gwf": gwf,
    }
    return {"weight_params": weight_params, "activation_state": activation_state}


# =============================================================================
# 2. Dynamic activation
# =============================================================================

def hif4_dynamic_quantize_activation(activation_quant, activation_scale,
                                     activation_state):
    with _t("dyn.dequant"):
        x = dequantize_nvfp4(activation_quant, activation_scale).float()
        R, C = x.shape
        s = None
        mode = 0
        if isinstance(activation_state, dict):
            t = activation_state.get("s")
            if isinstance(t, torch.Tensor) and t.numel() == C:
                s = t.float()
            mode = activation_state.get("mode") or 0
        if s is None:
            s = torch.ones(C, dtype=torch.float32)
        x = x * s
        if mode == 1:
            x = _rot_blocks(x)
    with _t("dyn.quant"):
        p = _quantize_weighted(x, torch.ones(1, C, dtype=torch.float32))
        unit = _params_unit_flat(p)
    values = None
    if isinstance(activation_state, dict) and activation_state.get("g") == 1:
        u = activation_state.get("u_act")
        order = activation_state.get("order")
        if isinstance(u, torch.Tensor) and tuple(u.shape) == (C, C):
            with _t("dyn.gptq"):
                if isinstance(order, torch.Tensor) and order.numel() == C:
                    ol = order.long()
                    xs = x[:, ol]
                    q = _gptq_quantize_values(xs, unit[:, ol], u.float())
                    q0 = torch.empty_like(q)
                    q0[:, ol] = q
                    values = q0
                else:
                    values = _gptq_quantize_values(x, unit, u.float())
    if isinstance(activation_state, dict) and R <= REFINE_T_MAX:
        gw = activation_state.get("gw")
        gwf = activation_state.get("gwf")
        if (isinstance(gw, torch.Tensor) and isinstance(gwf, torch.Tensor)
                and tuple(gw.shape) == (C, C) and tuple(gwf.shape) == (C, C)):
            try:
                with _t("dyn.refine"):
                    v0 = values if values is not None else _deq_params(p)
                    v1 = _refine_act_values(x, v0, unit, gw.float(), gwf.float())
                with _t("dyn.encode"):
                    return _values_to_params(v1.contiguous(), p)
            except Exception:
                pass
    if values is not None:
        with _t("dyn.encode"):
            return _values_to_params(values.contiguous(), p)
    return p


# =============================================================================
# 3. Attention calibration (mirrors solution.py v18)
# =============================================================================

def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads,
                               head_dim):
    torch.manual_seed(0)
    qh, kvh, dh = q_num_heads, kv_num_heads, head_dim
    rep = qh // kvh
    R = _make_R(dh)

    with _t("acal.dequant_hold"):
        hold = calib_qkv_list[-1]
        q = dequantize_nvfp4(*hold["q"]).float()
        k = dequantize_nvfp4(*hold["k"]).float()
        v = dequantize_nvfp4(*hold["v"]).float()
        stride = max(1, (q.shape[0] + 511) // 512)
        q = q[::stride].contiguous()
        k = k[::stride].contiguous()
        v = v[::stride].contiguous()
        T = q.shape[0]

    with _t("acal.hold_ref"):
        ref = _attention_out(q, k, v, qh, kvh, dh)
        ones_q = torch.ones(1, qh * dh, dtype=torch.float32)
        ones_k = torch.ones(1, kvh * dh, dtype=torch.float32)
        pv_hold = _quantize_weighted(v, ones_k)

    def run(qt, kt):
        pq = _quantize_weighted(qt, ones_q)
        pk = _quantize_weighted(kt, ones_k)
        out = _attention_out(_deq_params(pq), _deq_params(pk),
                             _deq_params(pv_hold), qh, kvh, dh)
        return ((out - ref) ** 2).mean().item()

    with _t("acal.rot_choice"):
        loss_off = run(q, k)
        rot = 0
        if R is not None:
            qr = (q.view(T, qh, dh) @ R).reshape(T, qh * dh)
            kr = (k.view(T, kvh, dh) @ R).reshape(T, kvh * dh)
            loss_on = run(qr, kr)
            rot = 1 if loss_on < loss_off else 0

    u_q = None
    u_k = None
    gq = 0
    Uq = Uk = None
    if len(calib_qkv_list) >= 2:
        with _t("acal.hess"):
            Hq = torch.zeros(kvh, dh, dh)
            Hk = torch.zeros(kvh, dh, dh)
            for smp in calib_qkv_list[:-1]:
                Tt = int(smp["q"][0].shape[0])
                if Tt > 1024:
                    continue
                qd = dequantize_nvfp4(*smp["q"]).float()
                kd = dequantize_nvfp4(*smp["k"]).float()
                if rot and R is not None:
                    qd = (qd.view(Tt, qh, dh) @ R).reshape(Tt, -1)
                    kd = (kd.view(Tt, kvh, dh) @ R).reshape(Tt, -1)
                qv = qd.view(Tt, qh, dh)
                kv_ = kd.view(Tt, kvh, dh)
                for hv in range(kvh):
                    Hk[hv] += kv_[:, hv].T @ kv_[:, hv]
                    for h in range(hv * rep, (hv + 1) * rep):
                        Hq[hv] += qv[:, h].T @ qv[:, h]
        with _t("acal.chol"):
            Uq = _upper_cholesky_inv(Hk)
            Uk = _upper_cholesky_inv(Hq)

    T0 = int(hold["q"][0].shape[0])
    qk_ready = Uq is not None and Uk is not None

    if qk_ready:
        with _t("acal.gq_guard"):
            qf_ = dequantize_nvfp4(*hold["q"]).float()
            kf_ = dequantize_nvfp4(*hold["k"]).float()
            vb0 = dequantize_nvfp4(*hold["v"]).float()
            if rot and R is not None:
                qf_rot = (qf_.view(T0, qh, dh) @ R).reshape(T0, -1)
                kf_rot = (kf_.view(T0, kvh, dh) @ R).reshape(T0, -1)
            else:
                qf_rot, kf_rot = qf_, kf_
            pq0 = _quantize_weighted(qf_rot, ones_q)
            pk0 = _quantize_weighted(kf_rot, ones_k)
            qh_d = _deq_params(pq0)
            kh_d = _deq_params(pk0)
            ref_o = _attention_out(qf_, kf_, vb0, qh, kvh, dh)

            out_b = _attention_out(qh_d, kh_d, vb0, qh, kvh, dh)
            mse_b = ((out_b - ref_o) ** 2).mean().item()

            def qk_gptq_apply():
                u_q_flat = _params_unit_flat(pq0).view(T0, qh, dh).permute(1, 0, 2).contiguous()
                u_k_flat = _params_unit_flat(pk0).view(T0, kvh, dh).permute(1, 0, 2).contiguous()
                qs = qf_rot.view(T0, qh, dh).permute(1, 0, 2).contiguous()
                ks = kf_rot.view(T0, kvh, dh).permute(1, 0, 2).contiguous()
                uq_full = Uq[(torch.arange(qh) // rep).clamp_max(Uq.shape[0] - 1)].float()
                qv_b = _gptq_quantize_batched(qs, u_q_flat, uq_full)
                kv_b = _gptq_quantize_batched(ks, u_k_flat, Uk.float())
                return (qv_b.permute(1, 0, 2).reshape(T0, -1),
                        kv_b.permute(1, 0, 2).reshape(T0, -1))

            qv_flat, kv_flat = qk_gptq_apply()
            out_gq = _attention_out(qv_flat, kv_flat, vb0, qh, kvh, dh)
            mse_gq = ((out_gq - ref_o) ** 2).mean().item()
            if mse_gq < mse_b:
                gq = 1
                u_q = Uq.contiguous()
                u_k = Uk.contiguous()

    q_state = {"rot": rot, "kvh": kvh}
    k_state = {"rot": rot, "kvh": kvh}
    if gq == 1:
        q_state.update({"gq": 1, "u": u_q})
        k_state.update({"gq": 1, "u": u_k})
    return {"q_state": q_state, "k_state": k_state, "v_state": None}


# =============================================================================
# 4/5/6. Dynamic Q/K/V
# =============================================================================

def _dyn_qk(quant, scale, state, num_heads, head_dim, role=None):
    with _t(f"d{role}.dequant"):
        x = dequantize_nvfp4(quant, scale).float()
        rot = isinstance(state, dict) and state.get("rot") == 1
        if rot:
            R = _make_R(head_dim)
            if R is not None:
                T = x.shape[0]
                x = (x.view(T, num_heads, head_dim) @ R).reshape(T, -1).contiguous()
    with _t(f"d{role}.quant"):
        p = _dyn_table(x, None, has_scale=False)
    values = None
    if isinstance(state, dict) and state.get("gq") == 1:
        u = state.get("u")
        kvh_n = state.get("kvh")
        if isinstance(u, torch.Tensor) and isinstance(kvh_n, int) and num_heads % kvh_n == 0:
            rep_n = num_heads // kvh_n
            T = x.shape[0]
            with _t(f"d{role}.gptq"):
                unit = _params_unit_flat(p)
                xs = x.view(T, num_heads, head_dim).permute(1, 0, 2).contiguous()
                us = unit.view(T, num_heads, head_dim).permute(1, 0, 2).contiguous()
                if num_heads <= u.shape[0]:
                    u_full = u[:num_heads].float()
                else:
                    hv_of = torch.arange(num_heads) // rep_n
                    u_full = u[hv_of.clamp_max(u.shape[0] - 1)].float()
                qs = _gptq_quantize_batched(xs, us, u_full)
                values = qs.permute(1, 0, 2).reshape(T, -1).contiguous()
                p = _values_to_params(values, p)
    if role is not None:
        if role == "q":
            _QKV_CARRY.clear()
        if values is None:
            values = _deq_params(p)
        _QKV_CARRY[role] = (x.contiguous(), values.contiguous())
    return p


def _dyn_v(quant, scale, state, kvh, dh):
    with _t("dv.dequant"):
        x = dequantize_nvfp4(quant, scale).float()
        T, C = x.shape
        qc = _QKV_CARRY.get("q")
        kc = _QKV_CARRY.get("k")
    budget_left = (S._VCOMP["el"] / max(S._VCOMP["n"], 1)) * (250 - S._VCOMP["n"]) < S._VCOMP_BUDGET
    if (isinstance(qc, tuple) and isinstance(kc, tuple)
            and qc[0].shape[0] == T and kc[0].shape[0] == T
            and qc[0].shape[1] % dh == 0 and kc[1].shape[1] == C
            and qc[0].shape[1] // dh % kvh == 0
            and T <= S._VCOMP_T_CAP and S._VCOMP["n"] < 250 and budget_left):
        t0 = time.perf_counter()
        try:
            with _t("dv.compensate"):
                out = _v_compensate(x, qc[0], qc[1], kc[0], kc[1], kvh, dh)
            S._VCOMP["n"] += 1
            S._VCOMP["el"] += time.perf_counter() - t0
            _QKV_CARRY.clear()
            return out
        except Exception:
            pass
    _QKV_CARRY.clear()
    with _t("dv.table"):
        return _dyn_table(x, state, has_scale=False)


def hif4_dynamic_quantize_q(q_quant, q_scale, q_num_heads, head_dim, q_state):
    return _dyn_qk(q_quant, q_scale, q_state, q_num_heads, head_dim, role="q")


def hif4_dynamic_quantize_k(k_quant, k_scale, kv_num_heads, head_dim, k_state):
    return _dyn_qk(k_quant, k_scale, k_state, kv_num_heads, head_dim, role="k")


def hif4_dynamic_quantize_v(v_quant, v_scale, kv_num_heads, head_dim, v_state):
    return _dyn_v(v_quant, v_scale, v_state, kv_num_heads, head_dim)
