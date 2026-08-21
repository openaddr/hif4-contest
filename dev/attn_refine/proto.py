"""Prototype: output-projected (uniform-row-weight) lattice refinement for the
V projection of the attention path.

Transformed-space analysis (see REPORT.md):
  * The v-quantizer operates on the RAW dequantized-NVFP4 fp32 tensor
    x = dequantize_nvfp4(v).float(), shape (T, C) with C = kv_num_heads*head_dim.
    v_state is always None (no smoothing s) and V is never rotated (only Q/K).
  * The judge's attention output is out_h = softmax(q_h k_hv^T / sqrt(dh)) @ v_hv
    -- there is NO output projection Wv in this task (task book + _attention_out
    + hif4.attn_ref).  Hence "Wv" = I_C (per kv-head column blocks are disjoint)
    and the uniform-row-weight proxy objective is  ||v_hat - v||_F^2, i.e.
    G = Wv^T Wv = I_C and _refine_act_values(x, values, unit, G, G).

Scored two ways per test call:
  * diag3-style full attention-output MSE score s = (mse_std - mse_play)/mse_std
    (baseline = exact paper Alg-1 quantizer V.quant_alg1).
  * exact-P V-side objective J_P = sum_{q heads h} ||P_h (v_hat - v)_hv||_F^2
    with P_h built from the SOLUTION's own quantized q/k (what the judge
    actually multiplies dv by) -- computable offline only.
  * proxy objective J_I = ||v_hat - v||_F^2 (what the refinement optimizes).

Both dynamic paths are measured:
  * "carry"  : the solution's own local path (q->k->v in one process fires
               _v_compensate via the module-global carry) -- the diag3 baseline.
  * "plain"  : carry cleared before the v call -- the ONLINE reality (the judge
               isolates calls; probe_carry proved the carry never assembles).
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "dev"))
import hif4  # noqa: E402
import variants as V  # noqa: E402


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SOL = load_mod(os.path.join(ROOT, "example", "solution", "solution.py"), "sol_attn_refine")
SWEEPS = (2, 4, 6)
ROUNDS = SOL.REFINE_ROUNDS  # 20, same as the linear side


def refine_uniform(x, values, unit, n_sweeps, rounds=ROUNDS):
    """Greedy top-1 lattice refinement against the uniform-row-weight objective
    ||(v_hat - v) @ Wv^T||^2 with Wv = I  (G = Wv^T Wv = I_C).

    Same machinery as the linear side: M = v_hat @ G - v @ G maintained via
    rank-1 Gram updates, flips stay on the legal v4 grid via SOL._flip_sel.
    """
    C = x.shape[1]
    G = torch.eye(C, dtype=torch.float32)
    v4 = torch.round(values / unit * 4.0)
    d = 0.25 * unit
    col2 = G.diagonal()
    M = (v4 * d) @ G - x @ G
    for _ in range(n_sweeps):
        for _ in range(rounds):
            g, dirn = SOL._flip_sel(d, M, col2, v4)
            idx = g.argmin(dim=1, keepdim=True)
            fin = torch.isfinite(g.gather(1, idx))
            dr = dirn.gather(1, idx) * fin.float()
            v4.scatter_add_(1, idx, dr)
            M += (dr * d.gather(1, idx)) * G[idx[:, 0]]
    return v4 * d


def attention_grams(q_hat, k_hat, qh, kvh, dh):
    """Per-kv-head time Grams Ghv = sum_{q heads h in group} P_h^T P_h, with
    P_h = softmax(q_h k_hv^T / sqrt(dh)) built from the played (quantized) q/k."""
    T = q_hat.shape[0]
    qf = q_hat.view(T, qh, dh).transpose(0, 1)
    kf = k_hat.view(T, kvh, dh).transpose(0, 1)
    rep = qh // kvh
    G = torch.zeros(kvh, T, T, dtype=torch.float32)
    for h in range(qh):
        hv = h // rep
        P = torch.softmax(qf[h] @ kf[hv].T / math.sqrt(dh), dim=-1)
        G[hv] += P.T @ P
    return G


def refine_with_gram(x, values, unit, G, n_sweeps, rounds=ROUNDS):
    """Greedy top-1 lattice refinement against J = sum_hv sum_{c in hv}
    d_c^T Ghv d_c, d_c = (v_hat - v)[:, c].  Columns are independent within
    the quadratic form, so top-1 PER COLUMN batched over all columns is exact
    coordinate descent (the row-transpose of the linear-side loop).
    Flip delta at (r, c): 2 s d M[r,c] + d^2 Ghv[r,r] with M = Ghv @ d_c."""
    T, C = x.shape
    dh = C // G.shape[0]
    kvh = G.shape[0]
    v4 = torch.round(values / unit * 4.0)
    d = 0.25 * unit
    dv = v4 * d - x
    M = torch.empty_like(dv)
    col2 = torch.empty_like(dv)
    for hv in range(kvh):
        sl = slice(hv * dh, (hv + 1) * dh)
        M[:, sl] = G[hv] @ dv[:, sl]
        col2[:, sl] = G[hv].diagonal().unsqueeze(1)
    for _ in range(n_sweeps):
        for _ in range(rounds):
            g = -2.0 * d * M.abs() + (d * d) * col2
            up = M < 0.0
            legal = torch.where(up, v4 < 7.0, v4 > -7.0)
            g = torch.where(legal & (g < 0.0), g, torch.full_like(g, float("inf")))
            idx = g.argmin(dim=0, keepdim=True)          # (1, C) top-1 row per column
            fin = torch.isfinite(g.gather(0, idx))
            dirn = torch.where(up, 1.0, -1.0).gather(0, idx) * fin.float()
            v4.scatter_add_(0, idx, dirn)
            dr = dirn * d.gather(0, idx)                  # (1, C) flip sizes
            for hv in range(kvh):
                cols = torch.arange(hv * dh, (hv + 1) * dh)
                ok = fin[0, cols]
                if not bool(ok.any()):
                    continue
                cc = cols[ok]
                M[:, cc] += G[hv][:, idx[0, cc]] * dr[0, cc]
    return v4 * d


def refine_exact_p(x, values, unit, q_hat, k_hat, qh, kvh, dh, n_sweeps, rounds=ROUNDS):
    """ORACLE (unshippable: needs this call's q/k): exact per-call P Grams."""
    G = attention_grams(q_hat, k_hat, qh, kvh, dh)
    return refine_with_gram(x, values, unit, G, n_sweeps, rounds)


def refine_flat_p(x, values, unit, kvh, n_sweeps, rounds=ROUNDS):
    """Shippable-P? structureless limit P = (1/T) 1 1^T  =>  P^T P = (1/T) 11^T.
    0 state bytes, T-independent Gram shape (all-ones/T per kv head)."""
    T = x.shape[0]
    G = torch.ones(kvh, T, T, dtype=torch.float32) / T
    return refine_with_gram(x, values, unit, G, n_sweeps, rounds)


def exact_p(q_hat, k_hat, v_vals, v_true, qh, kvh, dh):
    """J_P = sum over q heads of ||P_h (v_hat - v)_hv||_F^2, P_h from the
    played (quantized) q/k -- the exact linear map the judge applies to dv."""
    T = v_true.shape[0]
    dv = (v_vals - v_true).view(T, kvh, dh).transpose(0, 1).contiguous()
    qf = q_hat.view(T, qh, dh).transpose(0, 1)
    kf = k_hat.view(T, kvh, dh).transpose(0, 1)
    rep = qh // kvh
    J = 0.0
    for h in range(qh):
        hv = h // rep
        P = torch.softmax(qf[h] @ kf[hv].T / math.sqrt(dh), dim=-1)
        e = P @ dv[hv]
        J += (e * e).sum().item()
    return J


def main():
    torch.manual_seed(0)
    att = torch.load(os.path.join(ROOT, "example", "mini_sample", "attn.pt"),
                     weights_only=True, map_location="cpu")[0]
    qh, kvh, dh = att["q_num_heads"], att["kv_num_heads"], att["head_dim"]
    C = kvh * dh
    print(f"qh={qh} kvh={kvh} dh={dh}  C(v)={C}  "
          f"G = I_{C}: literal bf16 Gram would be {C*C*2} B = {C*C*2/1024:.0f} KiB "
          f"(identity is analytic -> 0 B actual state)")

    torch.manual_seed(0)
    t0 = time.perf_counter()
    acal = SOL.hif4_calibration_attention(att["calib"], qh, kvh, dh)
    print(f"calibration: {time.perf_counter()-t0:.2f}s  "
          f"v_state={acal['v_state']} (raw space: no s, no rotation)")

    out = {"C": C, "cases": []}
    for path in ("carry", "plain"):
        for ti, smp in enumerate(att["test"]):
            q_ref = hif4.dequantize_nvfp4(*smp["q"])
            k_ref = hif4.dequantize_nvfp4(*smp["k"])
            v_ref = hif4.dequantize_nvfp4(*smp["v"])
            ref = hif4.attn_ref(q_ref, k_ref, v_ref, qh, kvh, dh)
            qs = V.deq(V.quant_alg1(q_ref.float()))
            ks = V.deq(V.quant_alg1(k_ref.float()))
            vs = V.deq(V.quant_alg1(v_ref.float()))
            mse_std = ((hif4.attn_ref(qs, ks, vs, qh, kvh, dh) - ref) ** 2).mean().item()

            pq = SOL.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh, dh, acal["q_state"])
            pk = SOL.hif4_dynamic_quantize_k(smp["k"][0], smp["k"][1], kvh, dh, acal["k_state"])
            if path == "plain":
                SOL._QKV_CARRY.clear()  # simulate judge per-call isolation
            pv = SOL.hif4_dynamic_quantize_v(smp["v"][0], smp["v"][1], kvh, dh, acal["v_state"])
            q_hat = hif4.hif4_dequantize(pq)
            k_hat = hif4.hif4_dequantize(pk)
            v_hat = hif4.hif4_dequantize(pv)
            mse_base = ((hif4.attn_ref(q_hat, k_hat, v_hat, qh, kvh, dh) - ref) ** 2).mean().item()

            x = v_ref.float()                      # raw target the quantizer saw
            values = SOL._deq_params(pv)           # fp32 grid values (T, C)
            unit = SOL._params_unit_flat(pv)
            T = x.shape[0]
            JI_base = ((values - x) ** 2).sum().item()
            JP_base = exact_p(q_hat.float(), k_hat.float(), values, x, qh, kvh, dh)

            rec = {"path": path, "ti": ti, "T": T, "mse_std": mse_std,
                   "mse_base": mse_base, "JI_base": JI_base, "JP_base": JP_base,
                   "sweeps": {}}
            print(f"\n[{path} t{ti}] T={T}  std={mse_std:.4e} base={mse_base:.4e} "
                  f"score={(mse_std-mse_base)/mse_std:+.4f}  "
                  f"JI={JI_base:.4e} JP={JP_base:.4e}")
            for ns in SWEEPS:
                vv = values.clone()
                uu = unit.clone()
                t1 = time.perf_counter()
                vr = refine_uniform(x, vv, uu, ns)
                dt = time.perf_counter() - t1
                p_r = SOL._values_to_params(vr.contiguous(), pv)
                mse_r = ((hif4.attn_ref(q_hat, k_hat, hif4.hif4_dequantize(p_r),
                                        qh, kvh, dh) - ref) ** 2).mean().item()
                JI_r = ((vr - x) ** 2).sum().item()
                JP_r = exact_p(q_hat.float(), k_hat.float(), vr, x, qh, kvh, dh)
                changed = int((vr != values).sum().item())
                s_b = (mse_std - mse_base) / mse_std
                s_r = (mse_std - mse_r) / mse_std
                rec["sweeps"][ns] = {
                    "mse": mse_r, "JI": JI_r, "JP": JP_r, "changed": changed,
                    "ms": dt * 1000.0, "dscore": s_r - s_b,
                    "rel_JP": 1.0 - JP_r / JP_base if JP_base > 0 else 0.0,
                    "rel_JI": 1.0 - JI_r / JI_base if JI_base > 0 else 0.0,
                }
                print(f"  sweep {ns}: changed={changed:6d}/{values.numel()}  "
                      f"mse={mse_r:.4e} dscore={(s_r-s_b)*100:+.3f}pp  "
                      f"JI {JI_base:.4e}->{JI_r:.4e} ({(1-JI_r/JI_base)*100:+.1f}%)  "
                      f"JP {JP_base:.4e}->{JP_r:.4e} ({(1-JP_r/JP_base)*100:+.1f}%)  "
                      f"refine {dt*1000:.0f} ms")
            # ORACLE: exact-P-weighted refinement (unshippable upper bound)
            for ns in (SWEEPS[0], SWEEPS[-1]):
                t1 = time.perf_counter()
                vr = refine_exact_p(x, values.clone(), unit.clone(), q_hat.float(),
                                    k_hat.float(), qh, kvh, dh, ns)
                dt = time.perf_counter() - t1
                p_r = SOL._values_to_params(vr.contiguous(), pv)
                mse_r = ((hif4.attn_ref(q_hat, k_hat, hif4.hif4_dequantize(p_r),
                                        qh, kvh, dh) - ref) ** 2).mean().item()
                JP_r = exact_p(q_hat.float(), k_hat.float(), vr, x, qh, kvh, dh)
                changed = int((vr != values).sum().item())
                s_b = (mse_std - mse_base) / mse_std
                s_r = (mse_std - mse_r) / mse_std
                rec.setdefault("oracle", {})[ns] = {
                    "mse": mse_r, "JP": JP_r, "changed": changed, "ms": dt * 1000.0,
                    "dscore": s_r - s_b,
                    "rel_JP": 1.0 - JP_r / JP_base if JP_base > 0 else 0.0,
                }
                print(f"  ORACLE-P {ns}: changed={changed:6d}  "
                      f"mse={mse_r:.4e} dscore={(s_r-s_b)*100:+.3f}pp  "
                      f"JP {JP_base:.4e}->{JP_r:.4e} ({(1-JP_r/JP_base)*100:+.1f}%)  "
                      f"refine {dt*1000:.0f} ms")
            # flat-P proxy (structureless limit, shippable, 0 state)
            for ns in (SWEEPS[0], SWEEPS[-1]):
                t1 = time.perf_counter()
                vr = refine_flat_p(x, values.clone(), unit.clone(), kvh, ns)
                dt = time.perf_counter() - t1
                p_r = SOL._values_to_params(vr.contiguous(), pv)
                mse_r = ((hif4.attn_ref(q_hat, k_hat, hif4.hif4_dequantize(p_r),
                                        qh, kvh, dh) - ref) ** 2).mean().item()
                JP_r = exact_p(q_hat.float(), k_hat.float(), vr, x, qh, kvh, dh)
                changed = int((vr != values).sum().item())
                s_b = (mse_std - mse_base) / mse_std
                s_r = (mse_std - mse_r) / mse_std
                rec.setdefault("flatp", {})[ns] = {
                    "mse": mse_r, "JP": JP_r, "changed": changed, "ms": dt * 1000.0,
                    "dscore": s_r - s_b,
                    "rel_JP": 1.0 - JP_r / JP_base if JP_base > 0 else 0.0,
                }
                print(f"  FLAT-P   {ns}: changed={changed:6d}  "
                      f"mse={mse_r:.4e} dscore={(s_r-s_b)*100:+.3f}pp  "
                      f"JP {JP_base:.4e}->{JP_r:.4e} ({(1-JP_r/JP_base)*100:+.1f}%)  "
                      f"refine {dt*1000:.0f} ms")
            out["cases"].append(rec)

    # sanity: my loop must reproduce SOL._refine_act_values exactly (T<=512 -> 5 sweeps)
    smp = att["test"][0]
    v_ref = hif4.dequantize_nvfp4(*smp["v"])
    x = v_ref.float()
    p0 = SOL._dyn_table(x, None, has_scale=False)
    values0 = SOL._deq_params(p0)
    unit0 = SOL._params_unit_flat(p0)
    eye = torch.eye(C)
    a = SOL._refine_act_values(x, values0.clone(), unit0.clone(), eye, eye)
    b = refine_uniform(x, values0.clone(), unit0.clone(), 5)
    print(f"\nsanity vs SOL._refine_act_values @5 sweeps (T=10): "
          f"torch.equal={torch.equal(a, b)} maxdiff={float((a-b).abs().max())}")
    out["sanity_equal"] = bool(torch.equal(a, b))

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("wrote results.json")


if __name__ == "__main__":
    main()
