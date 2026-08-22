"""Profile _v_compensate internals at T=1024 (attn mini shape)."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

sol = harness.load_variant()
att = torch.load(os.path.join(harness.MINI, "attn.pt"), weights_only=True)[0]
qh, kvh, dh = att["q_num_heads"], att["kv_num_heads"], att["head_dim"]
torch.manual_seed(0)
acal = sol.hif4_calibration_attention(att["calib"], qh, kvh, dh)

for smp in att["test"]:
    T = smp["q"][0].shape[0]
    if T < 512:
        continue
    sol._QKV_CARRY.clear()
    xq = sol.dequantize_nvfp4(smp["q"][0], smp["q"][1]).float()
    xk = sol.dequantize_nvfp4(smp["k"][0], smp["k"][1]).float()
    xv = sol.dequantize_nvfp4(smp["v"][0], smp["v"][1]).float()
    pq = sol.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh, dh, acal["q_state"])
    pk = sol.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], kvh, dh, acal["k_state"])
    qc = sol._QKV_CARRY["q"]
    kc = sol._QKV_CARRY["k"]
    v = xv

    # instrumented re-run of _v_compensate phases
    import cProfile
    import pstats
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    out = sol._v_compensate(v, qc[0], qc[1], kc[0], kc[1], kvh, dh)
    t1 = time.perf_counter() - t0
    pr.disable()
    st = pstats.Stats(pr)
    st.sort_stats("tottime")
    print(f"=== T={T} v_compensate {t1:.3f}s ===")
    st.print_stats(14)

    # manual phase split
    qf = v  # placeholder
    def phases():
        t = {}
        t0 = time.perf_counter()
        T2, C = v.shape
        qf_ = qc[0].view(T2, qh, dh).transpose(0, 1)
        qhf = qc[1].view(T2, qh, dh).transpose(0, 1)
        kf_ = kc[0].view(T2, kvh, dh).transpose(0, 1)
        khf = kc[1].view(T2, kvh, dh).transpose(0, 1)
        G = torch.zeros(kvh, T2, T2, dtype=torch.float64)
        Cm = torch.zeros(kvh, T2, T2, dtype=torch.float64)
        for h in range(qh):
            hv = h // (qh // kvh)
            sc = (qf_[h] @ kf_[hv].T) / (dh ** 0.5)
            sch = (qhf[h] @ khf[hv].T) / (dh ** 0.5)
            P = torch.softmax(sc, dim=-1).double()
            Ph = torch.softmax(sch, dim=-1).double()
            G[hv] += Ph.T @ Ph
            Cm[hv] += Ph.T @ P
        t["gram_loop"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        lam = 1e-4 * G.diagonal(dim1=-2, dim2=-1).mean(-1).view(kvh, 1, 1)
        B = torch.linalg.solve(G + lam, Cm)
        t["solve"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        vs = torch.bmm(B, v.view(T2, kvh, dh).permute(1, 0, 2).double()) \
            .permute(1, 0, 2).reshape(T2, C).float().contiguous()
        t["bmm_v"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        p = sol._quantize_weighted(vs, torch.ones(1, C))
        t["quant"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        unit = sol._params_unit_flat(p)
        xs = vs.view(T2, kvh, dh).permute(1, 2, 0).contiguous()
        us = unit.view(T2, kvh, dh).permute(1, 2, 0).contiguous()
        U = sol._upper_cholesky_inv(G.float())
        qs = sol._gptq_quantize_batched(xs, us, U)
        t["gptq"] = time.perf_counter() - t0
        return t
    ph = phases()
    print("  phases:", {k: f"{v2:.3f}s" for k, v2 in ph.items()})
