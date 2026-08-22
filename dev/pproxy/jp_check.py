"""Focused diagnostic (mini, test[3], R=1024): does the fitted-P proxy reduce
its own proxy objective while RAISING the true exact-P objective J_P?"""
import importlib.util
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import hif4  # noqa: E402


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


AR = load_mod(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "attn_refine", "proto.py"), "ar3")
SOL = AR.SOL
PP = load_mod(os.path.join(os.path.dirname(os.path.abspath(__file__)), "proto.py"), "pp")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
att = torch.load(os.path.join(ROOT, "example", "mini_sample", "attn.pt"),
                 weights_only=True, map_location="cpu")[0]
qh, kvh, dh = att["q_num_heads"], att["kv_num_heads"], att["head_dim"]
torch.manual_seed(0)
acal = SOL.hif4_calibration_attention(att["calib"], qh, kvh, dh)
G = PP.calib_grams(att["calib"], qh, kvh, dh, "single")


def proxy_J(vr, x, Gh):
    dv = vr - x
    j = 0.0
    for hv in range(kvh):
        sl = slice(hv * dh, (hv + 1) * dh)
        m = Gh[hv] @ dv[:, sl]
        j += (m * dv[:, sl]).sum().item()
    return j


for ti in (2, 3):
    smp = att["test"][ti]
    q_ref = hif4.dequantize_nvfp4(*smp["q"])
    k_ref = hif4.dequantize_nvfp4(*smp["k"])
    v_ref = hif4.dequantize_nvfp4(*smp["v"])
    pq = SOL.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh, dh, acal["q_state"])
    pk = SOL.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], kvh, dh, acal["k_state"])
    SOL._QKV_CARRY.clear()
    pv = SOL.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], kvh, dh, acal["v_state"])
    q_hat = hif4.hif4_dequantize(pq).float()
    k_hat = hif4.hif4_dequantize(pk).float()
    x = v_ref.float()
    values = SOL._deq_params(pv)
    unit = SOL._params_unit_flat(pv)
    T = x.shape[0]
    Gh = G[T]
    JP0 = AR.exact_p(q_hat, k_hat, values, x, qh, kvh, dh)
    PJ0 = proxy_J(values, x, Gh)
    vr = AR.refine_with_gram(x, values.clone(), unit.clone(), Gh, 6)
    JP1 = AR.exact_p(q_hat, k_hat, vr, x, qh, kvh, dh)
    PJ1 = proxy_J(vr, x, Gh)
    vo = AR.refine_exact_p(x, values.clone(), unit.clone(), q_hat, k_hat, qh, kvh, dh, 6)
    JPo = AR.exact_p(q_hat, k_hat, vo, x, qh, kvh, dh)
    print(f"t{ti} T={T}:  true J_P {JP0:.4e} -> proxy {JP1:.4e} "
          f"({(JP1/JP0-1)*100:+.1f}%)  oracle {JPo:.4e} ({(JPo/JP0-1)*100:+.1f}%)")
    print(f"        proxy-obj (G_cal) {PJ0:.4e} -> {PJ1:.4e} ({(1-PJ1/PJ0)*100:+.1f}% removed)")
