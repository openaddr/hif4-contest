"""v8 patch: three-way transform choice + act-ordered activation GPTQ."""
p = "example/solution/solution.py"
src = open(p, encoding="utf-8").read()

start = src.index('def hif4_calibration_and_quantize_weight(')
end = src.index('# =============================================================================\n# 2. Dynamic Activation quantization')
new_fn = '''def hif4_calibration_and_quantize_weight(
    weight_quant: torch.Tensor,
    weight_scale: torch.Tensor,
    calib_activation_list: list,
) -> dict[str, Any]:
    """Weight path: alpha smoothing -> transform choice {rotation | channel
    permutation | none} via GPTQ-level subsample proxy -> anchored search ->
    hold-out-guarded GPTQ -> activation-side GPTQ (act-ordered).

    Both transforms are exact matmul invariants: rotation Gaussianizes
    intra-block outliers; permutation clusters same-magnitude channels into
    blocks while preserving the cross-channel correlations GPTQ exploits.
    """
    w = dequantize_nvfp4(weight_quant, weight_scale).float()
    R, C = w.shape
    ones_w = torch.ones(1, C, dtype=torch.float32)
    acts_raw = [dequantize_nvfp4(aq, as_).float() for aq, as_ in calib_activation_list]

    # ---- alpha smoothing search ----
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
        wp = _quant_weight_fast(w[rows] / s, torch.ones(1, C))
        wq = (wp["sign"] * wp["mant"] * wp["scale_lv3"] * wp["scale_lv2"]
              * wp["scale_factor"]).flatten(-4, -1) * s
        loss = ((a_big @ wq.T - a_big @ w[rows].T) ** 2).mean().item()
        if best_loss is None or loss < best_loss:
            best_loss, best_alpha = loss, alpha
    s = torch.exp(logm * best_alpha)
    w_s = w / s
    acts_s = [a * s for a in acts_raw]

    # ---- transform choice: {0: none, 1: rotation, 2: permutation} ----
    mode = 0
    perm = None
    Uw = None
    xh_pick = None
    if R > 64 and len(acts_s) >= 2 and acts_s[-1].shape[0] >= 8:
        rsub = torch.randperm(R)[: min(R, 256)]
        xh_last = acts_s[-1]
        sub = torch.randperm(xh_last.shape[0])[: min(xh_last.shape[0], 128)]
        energy = sq_sum / max(n_tok, 1)
        difficulty = (energy * (w * w).sum(dim=0)).sqrt()
        perm = torch.argsort(difficulty, descending=True)

        def tf(t, md):
            if md == 1:
                return _rot_blocks(t)
            if md == 2:
                return t[:, perm]
            return t

        spaces = []
        for md in (0, 1, 2):
            Hs = torch.zeros(C, C, dtype=torch.float32)
            for a in acts_s[:-1]:
                at = tf(a, md)
                Hs += at.T @ at
            spaces.append(_upper_cholesky_inv(Hs))
        xh_sub = xh_last[sub]
        cand = []
        for md, U in enumerate(spaces):
            if U is None:
                cand.append(float("inf"))
                continue
            w_rsub = tf(w_s[rsub], md)
            pp = _quant_weight_fast(w_rsub, torch.ones(1, C))
            qq = _gptq_quantize_values(w_rsub, _params_unit_flat(pp), U)
            xt = tf(xh_sub, md)
            cand.append(((xt @ qq.T - xt @ w_rsub.T) ** 2).mean().item())
        mode = int(torch.tensor(cand).argmin().item())
        if mode == 1 and spaces[1] is None:
            mode = 0
        if mode == 2 and spaces[2] is None:
            mode = 0
        Uw = spaces[mode]
        xh_pick = tf(xh_sub, mode).contiguous()

    def tf_final(t):
        if mode == 1:
            return _rot_blocks(t)
        if mode == 2:
            return t[:, perm]
        return t

    w_final = tf_final(w_s)

    # ---- anchored search + hold-out-guarded GPTQ ----
    weight_params = _quantize_weighted(w_final, ones_w)
    q_used = (weight_params["sign"] * weight_params["mant"] * weight_params["scale_lv3"]
              * weight_params["scale_lv2"] * weight_params["scale_factor"]).flatten(-4, -1)
    if xh_pick is not None and Uw is not None:
        unit = _params_unit_flat(weight_params)
        q_g = _gptq_quantize_values(w_final, unit, Uw)
        ref = xh_pick @ w_final.T
        mse_r = ((xh_pick @ q_used.T - ref) ** 2).mean().item()
        mse_g = ((xh_pick @ q_g.T - ref) ** 2).mean().item()
        if mse_g < mse_r:
            weight_params = _values_to_params(q_g, weight_params)
            q_used = q_g.contiguous()

    # ---- activation-side GPTQ with act-order ----
    u_act = None
    gptq_act = 0
    order = None
    if xh_pick is not None:
        Ha = q_used.T @ q_used
        Ua = _upper_cholesky_inv(Ha)
        if Ua is not None:
            order = torch.argsort(Ha.diagonal(), descending=True)
            Ua_o = _upper_cholesky_inv(Ha[order][:, order])
            if Ua_o is not None:
                Ua = Ua_o
            else:
                order = None
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
            xg = _gptq_quantize_values(xo_src, unit_src, Ua)
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
        "perm": (perm.contiguous() if mode == 2 else None),
        "u_act": u_act,
        "g": gptq_act,
        "order": (order.contiguous() if (gptq_act == 1 and order is not None) else None),
    }
    return {"weight_params": weight_params, "activation_state": activation_state}


'''
src = src[:start] + new_fn + src[end:]

old_dyn = '''def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    x = dequantize_nvfp4(activation_quant, activation_scale).float()
    R, C = x.shape
    s = None
    if isinstance(activation_state, dict):
        t = activation_state.get("s")
        if isinstance(t, torch.Tensor) and t.numel() == C:
            s = t.float()
    if s is None:
        s = torch.ones(C, dtype=torch.float32)
    x = x * s
    if isinstance(activation_state, dict) and activation_state.get("rot") == 1:
        x = _rot_blocks(x)
    p = _quantize_weighted(x, torch.ones(1, C, dtype=torch.float32))
    if isinstance(activation_state, dict) and activation_state.get("g") == 1:
        u = activation_state.get("u_act")
        if isinstance(u, torch.Tensor) and tuple(u.shape) == (C, C):
            unit = _params_unit_flat(p)
            q = _gptq_quantize_values(x, unit, u.float())
            return _values_to_params(q, p)
    return p'''
new_dyn = '''def hif4_dynamic_quantize_activation(
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    activation_state: Any,
) -> dict[str, torch.Tensor]:
    x = dequantize_nvfp4(activation_quant, activation_scale).float()
    R, C = x.shape
    s = None
    mode = 0
    perm = None
    if isinstance(activation_state, dict):
        t = activation_state.get("s")
        if isinstance(t, torch.Tensor) and t.numel() == C:
            s = t.float()
        mode = activation_state.get("mode") or 0
        t = activation_state.get("perm")
        if isinstance(t, torch.Tensor) and t.numel() == C:
            perm = t.long()
    if s is None:
        s = torch.ones(C, dtype=torch.float32)
    x = x * s
    if mode == 1:
        x = _rot_blocks(x)
    elif mode == 2 and perm is not None:
        x = x[:, perm].contiguous()
    p = _quantize_weighted(x, torch.ones(1, C, dtype=torch.float32))
    if isinstance(activation_state, dict) and activation_state.get("g") == 1:
        u = activation_state.get("u_act")
        order = activation_state.get("order")
        if isinstance(u, torch.Tensor) and tuple(u.shape) == (C, C):
            unit = _params_unit_flat(p)
            if isinstance(order, torch.Tensor) and order.numel() == C:
                ol = order.long()
                xs = x[:, ol]
                q = _gptq_quantize_values(xs, unit[:, ol], u.float())
                q0 = torch.empty_like(q)
                q0[:, ol] = q
                return _values_to_params(q0.contiguous(), p)
            q = _gptq_quantize_values(x, unit, u.float())
            return _values_to_params(q, p)
    return p'''
assert old_dyn in src, "dyn anchor"
src = src.replace(old_dyn, new_dyn)
open(p, "w", encoding="utf-8").write(src)
print("v8 written")
