"""Side probe: is the T=1024 dip the refinement sweep cap?

_refine_act_values uses n_sweeps = 5 if T<=512 else 2 if T<=1024.  Re-score
the T=1024 calls with 5 sweeps (monkeypatched copy, everything else identical)
on one dist slice (spread=0.5, outlier_p=0.0, both N) of every C.

  python dev/decomp/probe_sweeps.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study as S  # noqa: E402


def _refine_act_values5(x, values, unit, gw, gwf):
    """Copy of solution._refine_act_values with n_sweeps=5 for all T<=1024."""
    v4 = torch.round(values / unit * 4.0)
    d = 0.25 * unit
    col2 = gw.diagonal()
    M = (v4 * d) @ gw - x @ gwf
    n_sweeps = 5
    for _ in range(n_sweeps):
        for _ in range(S._SOL.REFINE_ROUNDS):
            g = -2.0 * d * M.abs() + (d * d) * col2
            up = M < 0.0
            v4c = v4
            legal = torch.where(up, v4c < 7.0, v4c > -7.0)
            g = torch.where(legal & (g < 0.0), g, torch.full_like(g, float("inf")))
            dirn = torch.where(up, 1.0, -1.0)
            idx = g.argmin(dim=1, keepdim=True)
            fin = torch.isfinite(g.gather(1, idx))
            dr = dirn.gather(1, idx) * fin.float()
            v4.scatter_add_(1, idx, dr)
            M += (dr * d.gather(1, idx)) * gw[idx[:, 0]]
    return v4 * d


def main():
    resA = S.jload(S.RES_A)
    for C in S.CS:
        for N in S.NS:
            name = f"c{C}_n{N}_s0.5_o0.0"
            cpath = os.path.join(S.CACHE, f"{name}_refined.pt")
            if not os.path.exists(cpath):
                continue
            gi = next(g for g in S.iter_grid(None, None) if g[0] == name)
            _, seed, C_, N_, sp, op = gi
            group = S.make_group(seed, C_, N_, sp, op)
            import hif4 as H
            w_ref = H.dequantize_nvfp4(*group["weight"])
            w_std = S.V.deq(S.V.quant_alg1(w_ref.float()))
            cal = torch.load(cpath, weights_only=True)["cal"]
            st, wp = cal["activation_state"], cal["weight_params"]
            base = [c for c in resA[name]["variants"]["refined"]["cases"]
                    if c["T"] == 1024]
            orig = S._SOL._refine_act_values
            try:
                S._SOL._refine_act_values = _refine_act_values5
                t0 = time.perf_counter()
                with5 = [S.score_case(group["test_activation_list"][i], w_ref,
                                      w_std, wp, st, 10 ** 9)
                         for i in (3, 4)]
                dt = time.perf_counter() - t0
            finally:
                S._SOL._refine_act_values = orig
            line = [f"{name}: T1024 score pp "
                    f"base {base[0]['score'] * 100:.2f}/{base[1]['score'] * 100:.2f}"
                    f" -> sweeps5 {with5[0]['score'] * 100:.2f}/{with5[1]['score'] * 100:.2f}"
                    f" ({dt:.1f}s for 2 calls)"]
            print(line[0])
            sys.stdout.flush()


if __name__ == "__main__":
    main()
