"""K-compensates-Q prototype: quantize K toward K* = K_r @ A^T where
A = (sum_h qhat^T qhat + lam I)^-1 (sum_h qhat^T q) per kv head, so that
Q dK^T cancels the KNOWN dQ K^T logit error. Cross-call carry of (q, qhat).

Measured: attention output MSE vs current pipeline, on mini + synthetic.
"""
import sys, os, importlib.util
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "example", "solution"))
sys.path.insert(0, ROOT)


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load_mod(os.path.join(ROOT, "..", "example", "solution", "solution.py"), "sol")
import hif4  # noqa: E402
import synth  # noqa: E402

torch.manual_seed(0)


def run(calib, tests, qh, kvh, dh, lam_exps=(0,)):
    rep = qh // kvh
    st = S.hif4_calibration_attention(calib, qh, kvh, dh)
    rot = st["q_state"].get("rot") == 1
    R = S._make_R(dh) if rot else None
    ones_q = torch.ones(1, qh * dh)
    ones_k = torch.ones(1, kvh * dh)

    def quant_k_comp(q_ref, qhat, k_ref, lam):
        """Compensated K quantization in rotated space."""
        T = k_ref.shape[0]
        if R is not None:
            qr = (q_ref.view(T, qh, dh) @ R).reshape(T, -1)
            kr = (k_ref.view(T, kvh, dh) @ R).reshape(T, -1)
        else:
            qr, kr = q_ref, k_ref
        qhr = qhat  # pipeline qhat is already in rotated space
        # per kv head: A = (Hhat)^-1 H, Hhat = sum_{h in group} qhat_h^T qhat_h
        qv = qr.view(T, qh, dh)
        qhv = qhr.view(T, qh, dh)
        k_target = kr.clone()
        Hs = []
        for hv in range(kvh):
            Hhat = torch.zeros(dh, dh)
            H = torch.zeros(dh, dh)
            for h in range(hv * rep, (hv + 1) * rep):
                Hhat += qhv[:, h].T @ qhv[:, h]
                H += qv[:, h].T @ qv[:, h]
            eye = torch.eye(dh)
            lam_d = lam * Hhat.diagonal().mean()
            A = torch.linalg.solve(Hhat + eye * lam_d, H)
            k_target.view(T, kvh, dh)[:, hv] = kr.view(T, kvh, dh)[:, hv] @ A.T
            Hs.append(Hhat)
        # quantize k_target: search units on it, then per-head GPTQ toward it
        pk = S._quantize_weighted(k_target.contiguous(), ones_k)
        unit = S._params_unit_flat(pk)
        us = unit.view(T, kvh, dh).permute(1, 0, 2).contiguous()
        xs = k_target.view(T, kvh, dh).permute(1, 0, 2).contiguous()
        U = S._upper_cholesky_inv(torch.stack(Hs))
        if U is None:
            return hif4.hif4_dequantize(pk), pk
        qs = S._gptq_quantize_batched(xs, us, U)
        k_flat = qs.permute(1, 0, 2).reshape(T, -1)
        return hif4.hif4_dequantize(S._values_to_params(k_flat.contiguous(), pk)), pk

    # baseline: current pipeline
    tot_base = 0.0
    tot_comp = {le: 0.0 for le in lam_exps}
    for smp in tests:
        q_ref = S.dequantize_nvfp4(*smp["q"]).float()
        k_ref = S.dequantize_nvfp4(*smp["k"]).float()
        v_ref = S.dequantize_nvfp4(*smp["v"]).float()
        ref = hif4.attn_ref(q_ref, k_ref, v_ref, qh, kvh, dh)
        pq = S.hif4_dynamic_quantize_q(*smp["q"], qh, dh, st["q_state"])
        pk = S.hif4_dynamic_quantize_k(*smp["k"], kvh, dh, st["k_state"])
        pv = S.hif4_dynamic_quantize_v(*smp["v"], kvh, dh, st["v_state"])
        qh_d = hif4.hif4_dequantize(pq)
        kh_d = hif4.hif4_dequantize(pk)
        vh_d = hif4.hif4_dequantize(pv)
        tot_base += ((hif4.attn_ref(qh_d, kh_d, vh_d, qh, kvh, dh) - ref) ** 2).mean().item()

        for le in lam_exps:
            lam = 10.0 ** le
            kc, _ = quant_k_comp(q_ref, qh_d, k_ref, lam)
            tot_comp[le] += ((hif4.attn_ref(qh_d, kc, vh_d, qh, kvh, dh) - ref) ** 2).mean().item()
    n = len(tests)
    out = f"base {tot_base/n:.3e}"
    for le in lam_exps:
        out += f"  lam1e{le:+d}: {tot_comp[le]/n:.3e} ({100*(1-tot_comp[le]/tot_base):+.0f}%)"
    return out


at = torch.load(os.path.join(ROOT, "..", "example", "mini_sample", "attn.pt"),
                weights_only=True, map_location="cpu")[0]
print("MINI    ", run(at["calib"], at["test"], at["q_num_heads"], at["kv_num_heads"], at["head_dim"],
                     lam_exps=(-4, -3, -2, 0)))
for name, g in [
    ("gqa_256", synth.make_attn_group(21, 16, 2, 256, spread=0.4)),
    ("mha_128", synth.make_attn_group(22, 8, 8, 128, spread=0.3)),
    ("gqa_128", synth.make_attn_group(24, 32, 4, 128, spread=0.5)),
    ("flat_256", synth.make_attn_group(23, 16, 2, 256, spread=0.1)),
]:
    print(f"{name:8s}", run(g["calib"], g["test"], g["q_num_heads"], g["kv_num_heads"],
                            g["head_dim"], lam_exps=(-4, -3, -2, 0)))
