"""V-compensates-P prototype: choose V to minimize ||Phat Vh - P V||^2 where
P = softmax of ORIGINAL q,k and Phat = softmax of our quantized qhat,khat
(both stashed from the q/k calls). Cancels the Q/K-induced output error
(the 71-85% pool) directly in V's quantization target.

V* = (sum_h Phat^T Phat + lam I)^-1 (sum_h Phat^T P) V  per kv head,
then GPTQ toward V* with Hessian sum Phat^T Phat.
"""
import sys, os, importlib.util, time
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


def probs_heads(q, k, qh, kvh, dh):
    """q (T, qh*dh) any space; returns (qh, T, T) softmax probs."""
    T = q.shape[0]
    qf = q.view(T, qh, dh).transpose(0, 1)
    kf = k.view(T, kvh, dh).transpose(0, 1)
    rep = qh // kvh
    sc = torch.bmm(qf, kf.repeat_interleave(rep, 0).transpose(1, 2)) / (dh ** 0.5)
    return torch.softmax(sc, dim=-1)


def run(calib, tests, qh, kvh, dh, lam_exps=(-4, -3, -2), t_cap=768):
    rep = qh // kvh
    st = S.hif4_calibration_attention(calib, qh, kvh, dh)
    rot = st["q_state"].get("rot") == 1
    R = S._make_R(dh) if rot else None
    ones_k = torch.ones(1, kvh * dh)

    tot_base = 0.0
    tot_comp = {le: 0.0 for le in lam_exps}
    tt = 0.0
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

        T = q_ref.shape[0]
        if T > t_cap:
            for le in lam_exps:
                tot_comp[le] += ((hif4.attn_ref(qh_d, kh_d, vh_d, qh, kvh, dh) - ref) ** 2).mean().item()
            continue

        t0 = time.time()
        # both in the pipeline's (rotated) space
        P = probs_heads(q_ref if not rot else (q_ref.view(T,qh,dh)@R).reshape(T,-1),
                        k_ref if not rot else (k_ref.view(T,kvh,dh)@R).reshape(T,-1), qh, kvh, dh)
        Ph = probs_heads(qh_d, kh_d, qh, kvh, dh)
        # per kv head Grams and cross terms
        G = torch.zeros(kvh, T, T)
        C = torch.zeros(kvh, T, T)
        for h in range(qh):
            hv = h // rep
            G[hv] += Ph[h].T @ Ph[h]
            C[hv] += Ph[h].T @ P[h]
        eye = torch.eye(T)
        for le in lam_exps:
            lam = (10.0 ** le) * G.diagonal(dim1=-2, dim2=-1).mean(-1).view(kvh, 1, 1)
            B = torch.linalg.solve(G + lam, C)          # (kvh, T, T)
            v_star = torch.bmm(B, v_ref.view(T, kvh, dh).permute(1, 0, 2)).permute(1, 0, 2).reshape(T, -1)
            v_star = v_star.contiguous()
            # quantize v_star: search + batched GPTQ toward it with Hessian G
            pv2 = S._quantize_weighted(v_star, ones_k)
            unit = S._params_unit_flat(pv2)
            xs = v_star.view(T, kvh, dh).permute(1, 2, 0).contiguous()
            us = unit.view(T, kvh, dh).permute(1, 2, 0).contiguous()
            U = S._upper_cholesky_inv(G)
            if U is None:
                vc = hif4.hif4_dequantize(pv2)
            else:
                qs = S._gptq_quantize_batched(xs, us, U)
                v_flat = qs.permute(2, 0, 1).reshape(T, -1)
                vc = hif4.hif4_dequantize(S._values_to_params(v_flat.contiguous(), pv2))
            tot_comp[le] += ((hif4.attn_ref(qh_d, kh_d, vc, qh, kvh, dh) - ref) ** 2).mean().item()
        tt += time.time() - t0
    n = len(tests)
    out = f"base {tot_base/n:.3e}"
    for le in lam_exps:
        out += f"  lam1e{le:+d}: {tot_comp[le]/n:.3e} ({100*(1-tot_comp[le]/tot_base):+.0f}%)"
    out += f"  [comp time {tt/len(lam_exps):.2f}s total]"
    return out


at = torch.load(os.path.join(ROOT, "..", "example", "mini_sample", "attn.pt"),
                weights_only=True, map_location="cpu")[0]
print("MINI    ", run(at["calib"], at["test"], at["q_num_heads"], at["kv_num_heads"], at["head_dim"]))
for name, g in [
    ("gqa_256", synth.make_attn_group(21, 16, 2, 256, spread=0.4)),
    ("mha_128", synth.make_attn_group(22, 8, 8, 128, spread=0.3)),
    ("gqa_128", synth.make_attn_group(24, 32, 4, 128, spread=0.5)),
    ("flat_256", synth.make_attn_group(23, 16, 2, 256, spread=0.1)),
]:
    print(f"{name:8s}", run(g["calib"], g["test"], g["q_num_heads"], g["kv_num_heads"], g["head_dim"]))
