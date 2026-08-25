"""Exp2b: refinement-objective forensics on mini (clean rewrite).

Dissect exp2's finding: deepening lattice refinement past the ship tier makes
TRUE output MSE worse while the internal (bf16-Gram) objective keeps
improving.  Track J_true(v_rounds) for bf16-gram and exact fp32-gram runs.
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import hif4  # noqa: E402

SOL = C.load_sol()
grp, _ = C.load_mini()
W, CAL, TST = grp["weight"], grp["calib_activation_list"], grp["test_activation_list"]
cal = torch.load(os.path.join(C.HERE, "cache", "mini_linear_cal_v40.pt"),
                 weights_only=True)
st = cal["activation_state"]
s = st["s"].float()
mode = st["mode"]
tf = (lambda t: SOL._rot_blocks(t)) if mode == 1 else (lambda t: t)

w_ref = hif4.dequantize_nvfp4(*W).float()
wt = tf(w_ref / s)
wq = hif4.hif4_dequantize(cal["weight_params"]).float()
gw32 = wq.T @ wq
gwf32 = wt.T @ wq
gw16, gwf16 = st["gw"].float(), st["gwf"].float()
print(f"gram bf16 rel err: gw {((gw16-gw32).norm()/gw32.norm()).item():.2e} "
      f"gwf {((gwf16-gwf32).norm()/gwf32.norm()).item():.2e}", flush=True)

u_act, order = st["u_act"], st["order"]


def staged(pair):
    T_, C_ = pair[0].shape
    x = SOL.dequantize_nvfp4(pair[0], pair[1]).float()
    xs = x * s
    if mode == 1:
        xs = SOL._rot_blocks(xs)
    p = SOL._quantize_weighted(xs, torch.ones(1, C_, dtype=torch.float32))
    unit = SOL._params_unit_flat(p)
    ol = order.long() if order is not None else None
    if ol is not None:
        q = SOL._gptq_quantize_values(xs[:, ol], unit[:, ol], u_act.float())
        q0 = torch.empty_like(q)
        q0[:, ol] = q
        v_gptq = q0
    else:
        v_gptq = SOL._gptq_quantize_values(xs, unit, u_act.float())
    return xs, v_gptq, unit


def run_traj(xs, v0, unit, gw, gwf, ckpts):
    """Greedy top-1 rounds; return {rounds: J_true} at checkpoints."""
    T, Cn = xs.shape
    v4 = torch.round(v0 / unit * 4.0)
    d = 0.25 * unit
    col2 = gw.diagonal()
    M = (v4 * d) @ gw - xs @ gwf
    neg2d, d2col = -2.0 * d, (d * d) * col2
    ref_out = xs @ wt.T

    def J(v4x):
        return ((v4x * d @ wq.T - ref_out) ** 2).mean().item()

    out = {0: J(v4)}
    Ma, v4a = M.clone(), v4
    da, neg2da, d2ca = d, neg2d, d2col
    ipos, ineg = v4a >= 7.0, v4a <= -7.0
    rows = torch.arange(T)
    local = False
    A = T
    g = torch.empty(T, Cn)
    gb = torch.empty(T, Cn)
    up = torch.empty(T, Cn, dtype=torch.bool)
    ill = torch.empty(T, Cn, dtype=torch.bool)
    rnd = 0
    last = max(ckpts)
    while rnd < last and A > 0:
        torch.abs(Ma, out=g)
        torch.addcmul(d2ca, g, neg2da, out=g)
        torch.lt(Ma, 0.0, out=up)
        torch.where(up, ipos, ineg, out=ill)
        g.masked_fill_(ill, float("inf"))
        idx = g.argmin(dim=1, keepdim=True)
        fin = g.gather(1, idx) < 0.0
        dr = torch.where(Ma.gather(1, idx) < 0.0, 1.0, -1.0) * fin.float()
        v4a.scatter_add_(1, idx, dr)
        nv = v4a.gather(1, idx)
        ipos.scatter_(1, idx, nv >= 7.0)
        ineg.scatter_(1, idx, nv <= -7.0)
        coef = dr * da.gather(1, idx)
        torch.index_select(gw, 0, idx[:, 0], out=gb)
        torch.addcmul(Ma, gb, coef, out=Ma)
        rnd += 1
        if rnd in ckpts:
            vcur = v4.clone()
            vcur[rows] = v4a
            out[rnd] = J(vcur)
        m = fin[:, 0]
        if not bool(m.all()):
            if local:
                v4[rows[~m]] = v4a[~m]
            rows, Ma, v4a = rows[m], Ma[m], v4a[m]
            da, neg2da, d2ca = da[m], neg2da[m], d2ca[m]
            ipos, ineg = ipos[m], ineg[m]
            local = True
            A = rows.shape[0]
            g = torch.empty(A, Cn)
            gb = torch.empty(A, Cn)
            up = torch.empty(A, Cn, dtype=torch.bool)
            ill = torch.empty(A, Cn, dtype=torch.bool)
    return out


CKS = set(range(0, 2001, 100))
res = {}
for pair in TST:
    T_ = pair[0].shape[0]
    xs, v_gptq, unit = staged(pair)
    if T_ <= 32:
        continue
    tr16 = run_traj(xs, v_gptq, unit, gw16, gwf16, CKS)
    tr32 = run_traj(xs, v_gptq, unit, gw32, gwf32, CKS)
    ship_sweeps = (44 if T_ <= 256 else 20 if T_ <= 512 else 8)
    ship_rounds = ship_sweeps * SOL.REFINE_ROUNDS
    res[T_] = {"ship_rounds": ship_rounds, "tr16": tr16, "tr32": tr32}
    print(f"T={T_} ship_rounds={ship_rounds}", flush=True)
    print("  bf16:", {k: round(v, 7) for k, v in sorted(tr16.items())}, flush=True)
    print("  fp32:", {k: round(v, 7) for k, v in sorted(tr32.items())}, flush=True)

with open(os.path.join(C.HERE, "results_exp2b.json"), "w", encoding="utf-8") as fh:
    json.dump({str(k): v for k, v in res.items()}, fh, indent=1)
print("DONE")
