"""Q/K channel smoothing (exact logit invariance, A = R diag(s)):
after rotation, scale q-rotated by s and k-rotated by 1/s where
s = (mean|k|/mean|q|)^beta per (kv-head, dim). Guarded on hold-out.
Measured on real mini attention + synthetic groups.
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


def eval_group(calib, tests, qh, kvh, dh, betas):
    rep = qh // kvh
    st = S.hif4_calibration_attention(calib, qh, kvh, dh)
    rot = st["q_state"].get("rot") == 1
    R = S._make_R(dh) if rot else None
    hold = calib[-1]
    qs_ = S.dequantize_nvfp4(*hold["q"]).float()
    ks_ = S.dequantize_nvfp4(*hold["k"]).float()
    vs_ = S.dequantize_nvfp4(*hold["v"]).float()
    T = qs_.shape[0]
    ref = hif4.attn_ref(qs_, ks_, vs_, qh, kvh, dh)

    def rotf(x, nh):
        if R is not None:
            t = x.shape[0]
            return (x.view(t, nh, dh) @ R).reshape(t, -1).contiguous()
        return x

    qr = rotf(qs_, qh)
    kr = rotf(ks_, kvh)

    # stats from calib[:-1] (fit), guard on hold
    mq = torch.zeros(kvh, dh)
    mk = torch.zeros(kvh, dh)
    for smp in calib[:-1]:
        qd = S.dequantize_nvfp4(*smp["q"]).float()
        kd = S.dequantize_nvfp4(*smp["k"]).float()
        Tt = qd.shape[0]
        if R is not None:
            qd = (qd.view(Tt, qh, dh) @ R).reshape(Tt, -1)
            kd = (kd.view(Tt, kvh, dh) @ R).reshape(Tt, -1)
        qh_avg = qd.view(Tt, qh, dh).view(Tt, kvh, rep, dh).mean(dim=2)
        mq += qh_avg.abs().mean(dim=0)
        mk += kd.view(Tt, kvh, dh).abs().mean(dim=0)
    n = max(len(calib) - 1, 1)
    mq /= n
    mk /= n

    res = {}
    for beta in betas:
        s = ((mk / mq.clamp_min(1e-12)) ** beta).clamp(0.5, 2.0)  # (kvh, dh)
        s_q = s.repeat_interleave(rep, 0)   # (qh, dh)
        pq = S._quantize_weighted((qr.view(T, qh, dh) * s_q).reshape(T, -1).contiguous(),
                                  torch.ones(1, qh * dh))
        pk = S._quantize_weighted((kr.view(T, kvh, dh) / s).reshape(T, -1).contiguous(),
                                  torch.ones(1, kvh * dh))
        pv = S._quantize_weighted(vs_, torch.ones(1, kvh * dh))
        out = hif4.attn_ref(hif4.hif4_dequantize(pq), hif4.hif4_dequantize(pk),
                            hif4.hif4_dequantize(pv), qh, kvh, dh)
        res[beta] = ((out - ref) ** 2).mean().item()

    # test-side: apply best beta (fit on calib[:-1] stats, chosen on hold) and
    # report TEST MSE vs beta=0 using the same chosen s
    b0 = res[0.0]
    best_beta = min(res, key=res.get)
    s = ((mk / mq.clamp_min(1e-12)) ** best_beta).clamp(0.5, 2.0)
    s_q = s.repeat_interleave(rep, 0)
    tot0 = totb = 0.0
    for smp in tests:
        qd = S.dequantize_nvfp4(*smp["q"]).float()
        kd = S.dequantize_nvfp4(*smp["k"]).float()
        vd = S.dequantize_nvfp4(*smp["v"]).float()
        Tt = qd.shape[0]
        ref_t = hif4.attn_ref(qd, kd, vd, qh, kvh, dh)
        qr_t = rotf(qd, qh)
        kr_t = rotf(kd, kvh)
        pq = S._quantize_weighted(qr_t, torch.ones(1, qh * dh))
        pk = S._quantize_weighted(kr_t, torch.ones(1, kvh * dh))
        pv = S._quantize_weighted(vd, torch.ones(1, kvh * dh))
        out0 = hif4.attn_ref(hif4.hif4_dequantize(pq), hif4.hif4_dequantize(pk),
                             hif4.hif4_dequantize(pv), qh, kvh, dh)
        pq2 = S._quantize_weighted((qr_t.view(Tt, qh, dh) * s_q).reshape(Tt, -1).contiguous(),
                                   torch.ones(1, qh * dh))
        pk2 = S._quantize_weighted((kr_t.view(Tt, kvh, dh) / s).reshape(Tt, -1).contiguous(),
                                   torch.ones(1, kvh * dh))
        outb = hif4.attn_ref(hif4.hif4_dequantize(pq2), hif4.hif4_dequantize(pk2),
                             hif4.hif4_dequantize(pv), qh, kvh, dh)
        tot0 += ((out0 - ref_t) ** 2).mean().item()
        totb += ((outb - ref_t) ** 2).mean().item()
    n_t = len(tests)
    return res[0.0], res[best_beta], best_beta, tot0 / n_t, totb / n_t


at = torch.load(os.path.join(ROOT, "..", "example", "mini_sample", "attn.pt"),
                weights_only=True, map_location="cpu")[0]
qh, kvh, dh = at["q_num_heads"], at["kv_num_heads"], at["head_dim"]
b0, bb, sel, t0, tb = eval_group(at["calib"], at["test"], qh, kvh, dh,
                                 (0.0, 0.15, 0.3, 0.45))
print(f"MINI  hold MSE {b0:.3e} -> {bb:.3e} (beta={sel})   TEST {t0:.3e} -> {tb:.3e}  "
      f"({100*(1-tb/t0):+.1f}%)")

for name, g in [
    ("gqa_256", synth.make_attn_group(21, 16, 2, 256, spread=0.4)),
    ("mha_128", synth.make_attn_group(22, 8, 8, 128, spread=0.3)),
    ("gqa_128", synth.make_attn_group(24, 32, 4, 128, spread=0.5)),
]:
    b0, bb, sel, t0, tb = eval_group(g["calib"], g["test"], g["q_num_heads"],
                                     g["kv_num_heads"], g["head_dim"],
                                     (0.0, 0.15, 0.3, 0.45))
    print(f"{name:8s} hold MSE {b0:.3e} -> {bb:.3e} (beta={sel})   TEST {t0:.3e} -> {tb:.3e}  "
          f"({100*(1-tb/t0):+.1f}%)")
