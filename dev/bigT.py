"""Large-T V-compensation study.

E1: gain without the GPTQ step (plain quantize V*) at T<=512 — isolates the
    value of the target shift alone.
E2: mini T=1024 (t3/t4): strided-Gram compensation (query-row subsampling is
    an unbiased estimator of G/Cm) + plain quantize.
E3: stride sweep {1,2,4,8} at T=1024.
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

torch.manual_seed(0)
at = torch.load(os.path.join(ROOT, "..", "example", "mini_sample", "attn.pt"),
                weights_only=True, map_location="cpu")[0]
qh, kvh, dh = at["q_num_heads"], at["kv_num_heads"], at["head_dim"]
rep = qh // kvh
st = S.hif4_calibration_attention(at["calib"], qh, kvh, dh)
rot = st["q_state"].get("rot") == 1
R = S._make_R(dh) if rot else None
ones_k = torch.ones(1, kvh * dh)


def probs(q, k):
    T = q.shape[0]
    qf = q.view(T, qh, dh).transpose(0, 1)
    kf = k.view(T, kvh, dh).transpose(0, 1)
    sc = torch.bmm(qf, kf.repeat_interleave(rep, 0).transpose(1, 2)) / (dh ** 0.5)
    return torch.softmax(sc, dim=-1)


def comp_params(v, q_in, q_hat, k_in, k_hat, stride, use_gptq, lam=1e-4, clamp=0.5):
    T, C = v.shape
    qh_ = q_in.shape[1] // dh
    P = probs(q_in, k_in).double()
    Ph = probs(q_hat, k_hat).double()
    idx = torch.arange(0, T, stride)
    G = torch.zeros(kvh, T, T, dtype=torch.float64)
    Cm = torch.zeros(kvh, T, T, dtype=torch.float64)
    for h in range(qh_):
        hv = h // rep
        Ph_s = Ph[h, idx]
        P_s = P[h, idx]
        G[hv] += Ph_s.T @ Ph_s
        Cm[hv] += Ph_s.T @ P_s
    lam_d = lam * G.diagonal(dim1=-2, dim2=-1).mean(-1).view(kvh, 1, 1)
    B = torch.linalg.solve(G + lam_d, Cm)
    vs = torch.bmm(B, v.view(T, kvh, dh).permute(1, 0, 2).double()) \
        .permute(1, 0, 2).reshape(T, C).float().contiguous()
    dv = vs - v
    dn = (dv.norm() / v.norm().clamp_min(1e-12)).item()
    if dn > clamp:
        vs = v + dv * (clamp / dn)
    p = S._quantize_weighted(vs, torch.ones(1, C))
    if use_gptq:
        unit = S._params_unit_flat(p)
        xs = vs.view(T, kvh, dh).permute(1, 2, 0).contiguous()
        us = unit.view(T, kvh, dh).permute(1, 2, 0).contiguous()
        U = S._upper_cholesky_inv(G.float())
        if U is not None:
            qs = S._gptq_quantize_batched(xs, us, U)
            vf = qs.permute(2, 0, 1).reshape(T, C)
            return S._values_to_params(vf.contiguous(), p)
    return p


for ti, smp in enumerate(at["test"]):
    q_ref = S.dequantize_nvfp4(*smp["q"]).float()
    k_ref = S.dequantize_nvfp4(*smp["k"]).float()
    v_ref = S.dequantize_nvfp4(*smp["v"]).float()
    ref = hif4.attn_ref(q_ref, k_ref, v_ref, qh, kvh, dh)
    T = q_ref.shape[0]
    pq = S.hif4_dynamic_quantize_q(*smp["q"], qh, dh, st["q_state"])
    pk = S.hif4_dynamic_quantize_k(*smp["k"], kvh, dh, st["k_state"])
    pv = S.hif4_dynamic_quantize_v(*smp["v"], kvh, dh, st["v_state"])
    qh_d = hif4.hif4_dequantize(pq)
    kh_d = hif4.hif4_dequantize(pk)
    vh_d = hif4.hif4_dequantize(pv)
    base = ((hif4.attn_ref(qh_d, kh_d, vh_d, qh, kvh, dh) - ref) ** 2).mean().item()
    qr = (q_ref.view(T, qh, dh) @ R).reshape(T, -1) if rot else q_ref
    kr = (k_ref.view(T, kvh, dh) @ R).reshape(T, -1) if rot else k_ref
    line = f"t{ti} T={T:5d} base {base:.3e}"
    for tag, stride, gptq in [("nogptq", 1, False), ("gptq", 1, True)] if T <= 512 else \
            [(f"s{st_}", st_, False) for st_ in (1, 2, 4, 8)]:
        t0 = time.time()
        p2 = comp_params(v_ref, qr, qh_d, kr, kh_d, stride, gptq)
        el = time.time() - t0
        vc = hif4.hif4_dequantize(p2)
        m = ((hif4.attn_ref(qh_d, kh_d, vc, qh, kvh, dh) - ref) ** 2).mean().item()
        line += f"  {tag}:{100*(1-m/base):+5.1f}%({el:.1f}s)"
    print(line)
