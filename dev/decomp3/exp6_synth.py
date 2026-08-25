"""Exp6: replicate key pools on SHARED-STRUCTURE synthetic groups (per discipline:
never iid synth for anatomy conclusions).

Linear (make_shared_group, share=1.0): c1024/c2048 x 2 seeds
  - side split (w-exact / x-exact pools) with ff_bal ship calibration
  - fp32-gram pool at ship sweeps (dynamic-only change)
Attention (make_shared_attn, share=1.0): gqa16x2x256 (mini shape) x 2 seeds
  - q/k/v side pools (transform-preserving exact swaps)
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import hif4  # noqa: E402
sys.path.insert(0, os.path.join(C.ROOT, "dev", "smooth"))
sys.path.insert(0, os.path.join(C.ROOT, "dev", "smattn"))
from exp_smooth import make_shared_group  # noqa: E402
from measure_persist import make_shared_attn  # noqa: E402

SOL = C.load_sol()
out = {"linear": [], "attn": []}

# ---------------- linear ----------------
for name, Cdim, N in (("c1024_shared", 1024, 2048), ("c2048_shared", 2048, 2048)):
    for k in range(2):
        seed = 5100 + 131 * k + (sum(map(ord, name)) % 977)
        torch.manual_seed(0)
        g = make_shared_group(seed, N, Cdim)
        W, CAL, TST = g["weight"], g["calib_activation_list"], g["test_activation_list"]
        torch.manual_seed(0)
        SOL.SMOOTH_MODE = "ff_bal"
        cal = SOL.hif4_calibration_and_quantize_weight(*W, CAL)
        st = cal["activation_state"]
        s = st["s"].float()
        mode = st["mode"]
        tf = (lambda t: SOL._rot_blocks(t)) if mode == 1 else (lambda t: t)
        w_ref = hif4.dequantize_nvfp4(*W).float()
        wt = tf(w_ref / s)
        wq = hif4.hif4_dequantize(cal["weight_params"]).float()
        gw32, gwf32 = wq.T @ wq, wt.T @ wq
        gw16, gwf16 = (st["gw"].float(), st["gwf"].float()
                       if st["gw"] is not None else (None, None))
        u_act, order = st["u_act"], st["order"]
        w_std_t = tf(C.V.deq(C.V.quant_alg1(w_ref)).float() / s)
        rows = {"cfg": name, "seed": seed, "mode": mode, "g": st["g"],
                "grams": st["gw"] is not None, "acc": SOL.SMOOTH_DEBUG.get("accepted")}
        keys = []
        for i, pair in enumerate(TST):
            T_, Cn = pair[0].shape
            x_ref = hif4.dequantize_nvfp4(*pair).float()
            xs = x_ref * s
            if mode == 1:
                xs = SOL._rot_blocks(xs)
            ref_out = xs @ wt.T
            mse_std = ((tf(C.V.deq(C.V.quant_alg1(x_ref)).float() * s) @ w_std_t.T
                        - ref_out) ** 2).mean().item()
            p = SOL.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
            xq = hif4.hif4_dequantize(p).float()
            mse_play = ((xq @ wq.T - ref_out) ** 2).mean().item()
            mse_x_only = (((xq - xs) @ wt.T) ** 2).mean().item()
            mse_w_only = ((xs @ (wq - wt).T) ** 2).mean().item()
            row = {"T": T_,
                   "pp_play": (mse_std - mse_play) / mse_std * 100,
                   "pp_xpool": (mse_std - mse_w_only) / mse_std * 100
                   - (mse_std - mse_play) / mse_std * 100,
                   "pp_wpool": (mse_std - mse_x_only) / mse_std * 100
                   - (mse_std - mse_play) / mse_std * 100}
            # fp32-gram pool at ship sweeps (if grams carried)
            if st["gw"] is not None:
                p2 = SOL._quantize_weighted(xs, torch.ones(1, Cn))
                unit = SOL._params_unit_flat(p2)
                ol = order.long() if order is not None else None
                if st["g"] == 1 and u_act is not None:
                    if ol is not None:
                        qq = SOL._gptq_quantize_values(xs[:, ol], unit[:, ol],
                                                       u_act.float())
                        q0 = torch.empty_like(qq)
                        q0[:, ol] = qq
                        v0 = q0
                    else:
                        v0 = SOL._gptq_quantize_values(xs, unit, u_act.float())
                else:
                    v0 = SOL._deq_params(p2)
                v16 = SOL._refine_act_values(xs, v0, unit, gw16, gwf16)
                v32 = SOL._refine_act_values(xs, v0, unit, gw32, gwf32)
                mse16 = ((v16 @ wq.T - ref_out) ** 2).mean().item()
                mse32 = ((v32 @ wq.T - ref_out) ** 2).mean().item()
                row["pp_gram32"] = (mse16 - mse32) / mse_std * 100
                row["pp_play_recheck"] = (mse_std - mse16) / mse_std * 100
            rows.setdefault("cases", []).append(
                {k2: round(v2, 3) for k2, v2 in row.items()})
        mkeys = [k for k in rows["cases"][0] if k != "T"]
        for mk in mkeys:
            rows["mean_" + mk] = round(sum(c[mk] for c in rows["cases"])
                                       / len(rows["cases"]), 3)
        out["linear"].append(rows)
        print(json.dumps({k: v for k, v in rows.items() if k != "cases"}), flush=True)

# ---------------- attention ----------------
for name, qh, kvh, dh in (("gqa16x2x256", 16, 2, 256), ("gqa32x8x64", 32, 8, 64)):
    for k in range(2):
        seed = 7300 + 131 * k + (sum(map(ord, name)) % 977)
        torch.manual_seed(0)
        g = make_shared_attn(seed, qh, kvh, dh)
        ACAL, ATST = g["calib"], g["test"]
        SOL.QKS_MODE = "pre"
        torch.manual_seed(0)
        cal = SOL.hif4_calibration_attention(ACAL, qh, kvh, dh)
        qs, ks = cal["q_state"], cal["k_state"]
        s = qs.get("qs")
        s = s.float() if s is not None else None
        rot = qs.get("rot")
        R = SOL._make_R(dh) if rot == 1 else None
        rep = qh // kvh
        rows = {"cfg": name, "seed": seed, "rot": rot, "gq": qs.get("gq"),
                "rf": qs.get("rf"), "acc": SOL.QKS_DEBUG.get("accepted")}
        for smp in ATST:
            q_ref = hif4.dequantize_nvfp4(*smp["q"])
            k_ref = hif4.dequantize_nvfp4(*smp["k"])
            v_ref = hif4.dequantize_nvfp4(*smp["v"])
            ref = hif4.attn_ref(q_ref, k_ref, v_ref, qh, kvh, dh)
            mse_std = ((hif4.attn_ref(C.V.deq(C.V.quant_alg1(q_ref.float())),
                                      C.V.deq(C.V.quant_alg1(k_ref.float())),
                                      C.V.deq(C.V.quant_alg1(v_ref.float())),
                                      qh, kvh, dh) - ref) ** 2).mean().item()
            pq = SOL.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh, dh,
                                             C.clone_state(qs))
            pk = SOL.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], kvh, dh,
                                             C.clone_state(ks))
            pv = SOL.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], kvh, dh,
                                             C.clone_state(cal["v_state"]))
            q_play, k_play, v_play = (hif4.hif4_dequantize(pq),
                                      hif4.hif4_dequantize(pk),
                                      hif4.hif4_dequantize(pv))
            T = q_ref.shape[0]
            qe, ke = q_ref.float(), k_ref.float()
            if s is not None:
                qe = SOL._qks_apply_q(qe, s, qh, kvh, dh, inv=False)
                ke = SOL._qks_apply_q(ke, s, qh, kvh, dh, inv=True)
            if R is not None:
                qe = (qe.view(T, qh, dh) @ R).reshape(T, -1)
                ke = (ke.view(T, kvh, dh) @ R).reshape(T, -1)
            def pp(mse):
                return (mse_std - mse) / mse_std * 100
            mse_play = ((hif4.attn_ref(q_play, k_play, v_play, qh, kvh, dh)
                         - ref) ** 2).mean().item()
            rows.setdefault("cases", []).append({
                "T": T, "pp_play": round(pp(mse_play), 3),
                "pp_qpool": round(pp(((hif4.attn_ref(qe, k_play, v_play, qh, kvh, dh)
                                       - ref) ** 2).mean().item()) - pp(mse_play), 3),
                "pp_kpool": round(pp(((hif4.attn_ref(q_play, ke, v_play, qh, kvh, dh)
                                       - ref) ** 2).mean().item()) - pp(mse_play), 3),
                "pp_vpool": round(pp(((hif4.attn_ref(q_play, k_play, v_ref.float(),
                                                      qh, kvh, dh) - ref) ** 2)
                                     .mean().item()) - pp(mse_play), 3),
                "pp_qkpool": round(pp(((hif4.attn_ref(qe, ke, v_play, qh, kvh, dh)
                                        - ref) ** 2).mean().item()) - pp(mse_play), 3)})
        for mk in ("pp_play", "pp_qpool", "pp_kpool", "pp_vpool", "pp_qkpool"):
            rows["mean_" + mk] = round(sum(c[mk] for c in rows["cases"])
                                       / len(rows["cases"]), 3)
        out["attn"].append(rows)
        print(json.dumps({k: v for k, v in rows.items() if k != "cases"}), flush=True)

with open(os.path.join(C.HERE, "results_exp6.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print("DONE")
