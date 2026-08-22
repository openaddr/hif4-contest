"""Task 1: where does the decomp2 act_rerank prototype (2.6-2.9s at C=2048)
spend its time?  Instrumented copy of anatomy.act_rerank (reflip branch,
identical math) with per-section wall timers, run on cached decomp2 groups.

Usage: python dev/rerank/profile_proto.py [name ...]
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.dirname(HERE)
ROOT = os.path.dirname(DEV)
D2 = os.path.join(DEV, "decomp2")
sys.path.insert(0, DEV)
sys.path.insert(0, D2)
import anatomy as A      # noqa: E402
import study2 as S2      # noqa: E402

SOL = S2.sol()
TM = defaultdict(float)
CNT = defaultdict(int)


def _t(key, dt):
    TM[key] += dt
    CNT[key] += 1


def act_rerank_timed(x, v_in, unit_in, p_ship, st, cands, passes=3, mod=SOL):
    gw = st["gw"].float()
    gwf = st["gwf"].float()
    T, C = x.shape
    nb = C // 64
    v = v_in.clone()
    unit = unit_in.clone()
    sf_sel = p_ship["scale_factor"].reshape(T, nb).clone()
    lv2_sel = p_ship["scale_lv2"].reshape(T, nb, 8).clone()
    lv3_sel = p_ship["scale_lv3"].reshape(T, nb, 8, 2).clone()
    t0 = time.perf_counter()
    Jt = ((v @ gw) * v).sum(dim=1) - 2.0 * ((x @ gwf) * v).sum(dim=1)
    _t("init_Jt_matmuls", time.perf_counter() - t0)
    rows = torch.arange(T)
    t0 = time.perf_counter()
    B = x @ gwf
    _t("B=x@gwf_once", time.perf_counter() - t0)
    moved = -1
    for it in range(passes):
        t0 = time.perf_counter()
        M2 = v @ gw
        _t("pass_M2_matmul", time.perf_counter() - t0)
        moved = 0
        for b in range(nb):
            sl = slice(b * 64, (b + 1) * 64)
            t0 = time.perf_counter()
            V, sfV, l2V, l3V = A.block_values_batched(x[:, sl], cands)
            _t("cand_requant", time.perf_counter() - t0)
            K = V.shape[0] + 1
            t0 = time.perf_counter()
            d = V - v[:, sl].unsqueeze(0)
            dJ = torch.zeros(K, T)
            dJ[:K - 1] = 2.0 * torch.einsum('ktc,tc->kt', d, M2[:, sl]) \
                - 2.0 * torch.einsum('ktc,tc->kt', d, B[:, sl]) \
                + torch.einsum('ktc,dc,ktd->kt', d, gw[sl, sl], d)
            kstar = dJ.argmin(dim=0)
            _t("dJ_einsum", time.perf_counter() - t0)
            t0 = time.perf_counter()
            v_try = v.clone()
            unit_try = unit.clone()
            sf_t = sf_sel.clone()
            lv2_t = lv2_sel.clone()
            lv3_t = lv3_sel.clone()
            ks = kstar.clamp_max(K - 2)
            v_try[:, sl] = V[ks, rows, :]
            unit_try[:, sl] = A._unit_of(sfV[ks, rows], l2V[ks, rows], l3V[ks, rows])
            sf_t[:, b] = sfV[ks, rows]
            lv2_t[:, b] = l2V[ks, rows]
            lv3_t[:, b] = l3V[ks, rows]
            _t("clone_scatter", time.perf_counter() - t0)
            t0 = time.perf_counter()
            v_try = mod._refine_act_values(x, v_try, unit_try, gw, gwf)
            _t("FULL_refine", time.perf_counter() - t0)
            t0 = time.perf_counter()
            Jt_try = ((v_try @ gw) * v_try).sum(dim=1) \
                - 2.0 * ((x @ gwf) * v_try).sum(dim=1)
            _t("Jt_try_matmuls", time.perf_counter() - t0)
            better = Jt_try < Jt - 1e-9
            if better.any():
                moved += int(better.sum())
                Jt = torch.where(better, Jt_try, Jt)
                keep = better.unsqueeze(1)
                v = torch.where(keep, v_try, v)
                unit = torch.where(keep, unit_try, unit)
                sf_sel = torch.where(keep, sf_t, sf_sel)
                lv2_sel = torch.where(keep.unsqueeze(1), lv2_t, lv2_sel)
                lv3_sel = torch.where(keep.unsqueeze(1).unsqueeze(1), lv3_t, lv3_sel)
                t0 = time.perf_counter()
                M2 = v @ gw
                _t("accept_M2_matmul", time.perf_counter() - t0)
        if moved == 0:
            break
    return v, unit, sf_sel, lv2_sel, lv3_sel, {"moved_last": moved}


def profile(name):
    group = S2.build_group(name)
    cc = torch.load(os.path.join(S2.CACHE, f"{name}_ship.pt"), weights_only=True)
    st = cc["cal"]["activation_state"]
    pair = group["test_activation_list"][0]
    assert pair[0].shape[0] == 10
    ints = A.dynamic_internals(pair, st)
    torch.manual_seed(0)
    t0 = time.perf_counter()
    act_rerank_timed(ints["x"], ints["v1"], ints["unit"], ints["p"], st,
                     SOL.CAND_GRID, passes=3)
    tot = time.perf_counter() - t0
    print(f"\n{name}: total {tot:.2f}s")
    acc = 0.0
    for k in sorted(TM, key=TM.get, reverse=True):
        print(f"  {k:<22} {TM[k]:>7.3f}s  n={CNT[k]:>4}  "
              f"{TM[k] / max(tot, 1e-9) * 100:>5.1f}%")
        acc += TM[k]
    print(f"  {'(sum)':<22} {acc:>7.3f}s")
    TM.clear()
    CNT.clear()


def main():
    args = sys.argv[1:]
    if args:
        names = args
    else:
        names = ["c2048_n1024_s0.5_o0.002", "c2048_n8192_s0.9_o0.002"]
    for n in names:
        profile(n)


if __name__ == "__main__":
    main()
