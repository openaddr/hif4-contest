"""Measured speedup experiments against the CURRENT solution (v18), with
bit-identity verification (torch.equal on every output param tensor).

Fixes are applied as in-memory monkeypatches / textual patches of a LOADED
COPY of the solution source (example/solution/solution.py is never touched):

  f1  vectorized _quant_chunk  (candidate-batched; the lv2/lv3 inner search
      and candidate merge replicate the ORIGINAL tie semantics via strictly
      ordered torch.where -> bit-identical)
  f2  numpy _gptq_quantize_values (per-column elementwise ops in numpy
      sharing torch memory, cross-block matmul kept in torch MKL ->
      bit-identical)
  f3  act-GPTQ double-Cholesky skip (compute Ua only if the act-ordered
      Ua_o failed; textual patch of the calibration driver)
  f4  ROW_CHUNK / GPTQ_BLOCK sweeps (separate subcommand)
  f5  single-decomposition upper-Cholesky-of-inverse identity probe

Usage:
  C:/App/env/Python/python.exe dev/audit/exp_speed.py run c2048_n8192 ...
  C:/App/env/Python/python.exe dev/audit/exp_speed.py sweep
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUDIT = os.path.join(ROOT, "dev", "audit")
SOLUTION = os.path.join(ROOT, "example", "solution", "solution.py")
DATA_DIR = os.path.join(AUDIT, "data")

_SRC = open(SOLUTION, encoding="utf-8").read()
_vn = [0]


def load_variant(patch_src=None, **attrs):
    _vn[0] += 1
    p = os.path.join(AUDIT, f"_variant_{_vn[0]}.py")
    with open(p, "w", encoding="utf-8") as f:
        f.write(_SRC if patch_src is None else patch_src)
    spec = importlib.util.spec_from_file_location(f"_expvar_{_vn[0]}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def eq_params(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def eq_state(a, b):
    for k in set(a) | set(b):
        x, y = a.get(k), b.get(k)
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            if not (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
                    and torch.equal(x, y)):
                return False
        elif x != y:
            return False
    return True


# ===========================================================================
# f1: vectorized _quant_chunk (bit-identical candidate batching)
# ===========================================================================

SF_MIN_V = 2.0 ** -48
SF_MAX_V = 49152.0


def _qc_vec_impl(xb, wblk, grid, KB):
    ab = xb.abs()
    amax = ab.amax(dim=(2, 3, 4), keepdim=True)
    t = (amax / 7.0).clamp_min(1e-38)
    e0 = torch.floor(torch.log2(t)).squeeze(-1).squeeze(-1).squeeze(-1)  # (r,nb)
    K = len(grid)
    offs = torch.tensor([float(k) for k, _ in grid])
    sigs = torch.tensor([float(s) for _, s in grid])
    sf_all = (torch.exp2(e0.unsqueeze(-1) + offs) * sigs).clamp(SF_MIN_V, SF_MAX_V)
    abB = ab.unsqueeze(2)                    # (r,nb,1,8,2,4) view
    wbB = (wblk.unsqueeze(2) if wblk.dim() == 5
           else wblk.unsqueeze(0).unsqueeze(2))  # broadcastable (r?,nb,1,8,2,4)
    r, nb = e0.shape
    big_shape = (r, nb, KB, 8, 2, 4)
    tmp = torch.empty(big_shape, dtype=torch.float32)

    def run_batch(sf):
        kB = sf.shape[2]
        best_e2 = best_l2 = best_l3 = None
        for lv2_c in (1.0, 2.0):
            e3_list = []
            for lv3_c in (1.0, 2.0):
                unit = (sf.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                        * lv2_c * lv3_c)                     # (r,nb,kB,1,1,1)
                tgt = tmp[:, :, :kB] if kB < KB else tmp
                torch.div(abB, unit, out=tgt)
                tgt.mul_(4.0)
                tgt.round_()
                tgt.mul_(0.25)
                tgt.clamp_(0.0, 1.75)                        # mant
                tgt.mul_(unit)
                tgt.sub_(abB)
                tgt.pow_(2)
                tgt.mul_(wbB)
                e3_list.append(tgt.sum(dim=5))               # (r,nb,kB,8,2)
            take1 = e3_list[0] <= e3_list[1]
            e3 = torch.where(take1, e3_list[0], e3_list[1])
            lv3c = torch.where(take1, 1.0, 2.0)              # (r,nb,kB,8,2)
            e2 = e3.sum(dim=4)                               # (r,nb,kB,8)
            if best_e2 is None:
                best_e2 = e2
                best_l2 = torch.full_like(e2, lv2_c)     # (r,nb,kB,8)
                best_l3 = lv3c                            # (r,nb,kB,8,2)
            else:
                take2 = e2 < best_e2
                best_e2 = torch.where(take2, e2, best_e2)
                best_l2 = torch.where(take2, torch.full_like(e2, lv2_c), best_l2)
                best_l3 = torch.where(take2.unsqueeze(-1), lv3c, best_l3)
        return best_e2.sum(dim=3), best_l2, best_l3          # (r,nb,kB)

    err_best = sf_best = lv2_best = lv3_best = None
    for k0 in range(0, K, KB):
        sf = sf_all[:, :, k0:k0 + KB]
        err, l2, l3 = run_batch(sf)
        for kk in range(err.shape[2]):
            err_k = err[:, :, kk]
            if err_best is None:
                err_best = err_k
                sf_best = sf[:, :, kk]
                lv2_best = l2[:, :, kk]
                lv3_best = l3[:, :, kk]
            else:
                take = err_k < err_best
                take2 = take.unsqueeze(-1)
                take3 = take.unsqueeze(-1).unsqueeze(-1)
                err_best = torch.where(take, err_k, err_best)
                sf_best = torch.where(take, sf[:, :, kk], sf_best)
                lv2_best = torch.where(take2, l2[:, :, kk], lv2_best)
                lv3_best = torch.where(take3, l3[:, :, kk], lv3_best)

    sf5 = sf_best.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    lv2 = lv2_best.unsqueeze(-1).unsqueeze(-1)
    lv3 = lv3_best.unsqueeze(-1)
    unit = sf5 * lv2 * lv3
    mant = torch.clamp(torch.round(ab / unit * 4.0) / 4.0, 0.0, 1.75)
    return {"scale_factor": sf5, "scale_lv2": lv2, "scale_lv3": lv3,
            "sign": torch.sign(xb), "mant": mant}


# ===========================================================================
# f2: numpy _gptq_quantize_values (torch matmuls, numpy elementwise)
# ===========================================================================

def _gptq_np(x: torch.Tensor, unit: torch.Tensor, hinv: torch.Tensor,
             GB: int) -> torch.Tensor:
    R, C = x.shape
    W = x.clone()
    Q = torch.empty_like(W)
    unp = (unit if unit.is_contiguous() else unit.contiguous()).numpy()
    hnp = hinv.contiguous().numpy()
    npr_, npa_, npw_, npc_ = np.round, np.abs, np.where, np.clip
    one, mone = np.float32(1.0), np.float32(-1.0)
    for i1 in range(0, C, GB):
        i2 = min(i1 + GB, C)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        w1, q1, e1 = W1.numpy(), Q1.numpy(), E1.numpy()
        Hi = hnp[i1:i2, i1:i2]
        u = unp[:, i1:i2]
        last = i2 - i1 - 1
        for i in range(i2 - i1):
            w = w1[:, i]
            ui = u[:, i]
            m = npr_(npa_(w) / ui * 4.0)
            npc_(m, 0.0, 7.0, out=m)
            m *= 0.25
            s = npw_(w >= 0, one, mone)
            q = s * m * ui
            q1[:, i] = q
            d = Hi[i, i]
            if d < 1e-30:
                d = np.float32(1e-30)
            e1[:, i] = (w - q) / d
            if i < last:
                w1[:, i + 1:] -= e1[:, i][:, None] * Hi[i, i + 1:]
        Q[:, i1:i2] = Q1
        if i2 < C:
            W[:, i2:] -= E1 @ hinv[i1:i2, i2:]
            W[:, i1:i2] = W1
    return Q


# ===========================================================================
# f3: textual patch -- skip the redundant act-GPTQ Cholesky
# ===========================================================================

_OLD_F3 = """    if xh_pick is not None:
        Ha = q_used.T @ q_used
        Ua = _upper_cholesky_inv(Ha)
        if Ua is not None:
            order = torch.argsort(Ha.diagonal(), descending=True)
            Ua_o = _upper_cholesky_inv(Ha[order][:, order])
            if Ua_o is not None:
                Ua = Ua_o
            else:
                order = None
            p_r = _quantize_weighted(xh_pick, ones_w)"""
_NEW_F3 = """    if xh_pick is not None:
        Ha = q_used.T @ q_used
        order = torch.argsort(Ha.diagonal(), descending=True)
        Ua_o = _upper_cholesky_inv(Ha[order][:, order])
        if Ua_o is not None:
            Ua = Ua_o
        else:
            order = None
            Ua = _upper_cholesky_inv(Ha)
        if Ua is not None:
            p_r = _quantize_weighted(xh_pick, ones_w)"""


def build_f3():
    assert _SRC.count(_OLD_F3) == 1, "f3 anchor not found"
    return _SRC.replace(_OLD_F3, _NEW_F3)


# ===========================================================================
# harness
# ===========================================================================

def load_group(name):
    return torch.load(os.path.join(DATA_DIR, f"{name}.pt"),
                      weights_only=True, map_location="cpu")


def run_variant(sol, g):
    torch.manual_seed(0)
    t0 = time.perf_counter()
    out = sol.hif4_calibration_and_quantize_weight(
        g["weight"][0], g["weight"][1], g["calib_activation_list"])
    t_cal = time.perf_counter() - t0
    st = out["activation_state"]
    t_dyn = 0.0
    for pair in g["test_activation_list"]:
        t0 = time.perf_counter()
        sol.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
        t_dyn += time.perf_counter() - t0
    return out, t_cal, t_dyn


def _gptq_dispatch_factory(torch_fn, thresh):
    def fn(x, u, h):
        if x.shape[0] >= thresh:
            return torch_fn(x, u, h)
        return _gptq_np(x, u, h, 128)
    return fn


def cmd_run(names):
    base = load_variant()
    variants = [
        ("f1_KB4", load_variant(_quant_chunk=lambda a, b, g: _qc_vec_impl(a, b, g, 4))),
        ("f2_np", load_variant(_gptq_quantize_values=lambda x, u, h: _gptq_np(x, u, h, base.GPTQ_BLOCK))),
        ("f2d_np", load_variant(_gptq_quantize_values=_gptq_dispatch_factory(
            base._gptq_quantize_values, 2048))),
        ("f3_cholskip", load_variant(patch_src=build_f3())),
        ("combo", load_variant(
            patch_src=build_f3(),
            _quant_chunk=lambda a, b, g: _qc_vec_impl(a, b, g, 4),
            _gptq_quantize_values=_gptq_dispatch_factory(
                base._gptq_quantize_values, 2048))),
    ]
    print(f"{'variant':<12s} {'config':<12s} {'cal s':>7s} {'dyn s':>7s} {'d_cal':>7s} {'d_dyn':>7s} ident")
    for name in names:
        g = load_group(name)
        out_b, tc_b, td_b = run_variant(base, g)
        print(f"{'BASE':<12s} {name:<12s} {tc_b:7.2f} {td_b:7.2f} {'-':>7s} {'-':>7s} -")
        st_b = out_b["activation_state"]
        for tag, sol in variants:
            out_v, tc_v, td_v = run_variant(sol, g)
            ident = (eq_params(out_b["weight_params"], out_v["weight_params"])
                     and eq_state(st_b, out_v["activation_state"]))
            if ident:
                for pair in g["test_activation_list"]:
                    pb = base.hif4_dynamic_quantize_activation(pair[0], pair[1], st_b)
                    pv = sol.hif4_dynamic_quantize_activation(pair[0], pair[1], st_b)
                    ident = ident and eq_params(pb, pv)
            print(f"{tag:<12s} {name:<12s} {tc_v:7.2f} {td_v:7.2f} "
                  f"{tc_b - tc_v:+7.2f} {td_b - td_v:+7.2f} "
                  f"{'YES' if ident else 'NO'}")
        # drift check: rerun base
        out_b2, tc_b2, td_b2 = run_variant(base, g)
        print(f"{'BASE(2)':<12s} {name:<12s} {tc_b2:7.2f} {td_b2:7.2f} "
              f"  drift cal {tc_b2 - tc_b:+.2f} dyn {td_b2 - td_b:+.2f}")
        sys.stdout.flush()


def cmd_unit():
    """Randomized bit-identity stress of _qc_vec_impl vs original, including
    tie-heavy inputs (values exactly on the grid -> zero-error ties)."""
    base = load_variant()
    rng = np.random.default_rng(7)
    grids = [base.CAND_GRID, base.CAND_GRID_W]
    fails = 0
    for trial in range(40):
        r = int(rng.integers(1, 200))
        nb = int(rng.integers(1, 40))
        mode = trial % 4
        xb = torch.randn(r, nb, 8, 2, 4) * (2.0 ** float(rng.integers(-20, 8)))
        if mode == 1:   # tie storm: exact multiples -> many zero-error candidates
            xb = (torch.randn(r, nb, 8, 2, 4).round() * 0.5)
        elif mode == 2:  # zeros
            xb = torch.zeros(r, nb, 8, 2, 4)
            xb[:, :, :, :, 0] = 1.5
        wb = (torch.rand(r, nb, 8, 2, 4) + 0.1) if trial % 2 else torch.rand(nb, 8, 2, 4) + 0.1
        for grid in grids:
            a = base._quant_chunk(xb, wb, grid)
            b = _qc_vec_impl(xb, wb, grid, KB=4)
            ok = all(torch.equal(a[k], b[k]) for k in a)
            if not ok:
                fails += 1
                bad = [k for k in a if not torch.equal(a[k], b[k])]
                print(f"[unit] FAIL trial={trial} grid={len(grid)} r={r} nb={nb} "
                      f"mode={mode} bad={bad}")
    print(f"[unit] _qc_vec_impl: {80 - fails}/80 bit-identical")

    # numpy GPTQ stress, incl. tiny/bigger blocks and degenerate hinv diagonals
    base2 = load_variant()
    fails = 0
    for trial in range(30):
        R = int(rng.integers(1, 300))
        Cb = int(rng.integers(64, 600))
        x = torch.randn(R, Cb) * 3
        A = torch.randn(Cb, Cb)
        h = A.T @ A + torch.eye(Cb) * 0.5
        u = (torch.rand(R, Cb) + 0.5)
        for GB in (64, 128, 256):
            base2.GPTQ_BLOCK = GB
            a = base2._gptq_quantize_values(x, u, h)
            b = _gptq_np(x, u, h, GB)
            if not torch.equal(a, b):
                fails += 1
                d = (a - b).abs().max().item()
                print(f"[unit] GPTQ-NP FAIL trial={trial} GB={GB} maxdiff={d:.3e}")
    print(f"[unit] _gptq_np: {90 - fails}/90 bit-identical")


def cmd_sweep():
    """f4: ROW_CHUNK x GPTQ_BLOCK timing on weight-quant + dyn GPTQ.
    ROW_CHUNK is bit-identical (rows independent); GPTQ_BLOCK is not."""
    base = load_variant()
    g = load_group("c2048_n8192")
    w = base.dequantize_nvfp4(g["weight"][0], g["weight"][1]).float()
    ones = torch.ones(1, w.shape[1])
    print("[sweep] _quantize_weighted(8192x2048, 16-cand) ROW_CHUNK:")
    for rc in (256, 512, 1024, 2048, 4096):
        base.ROW_CHUNK = rc
        t0 = time.perf_counter()
        p1 = base._quantize_weighted(w, ones)
        t1 = time.perf_counter() - t0
        base.ROW_CHUNK = 2048
        t0 = time.perf_counter()
        p0 = base._quantize_weighted(w, ones)
        t0b = time.perf_counter() - t0
        ident = all(torch.equal(p0[k], p1[k]) for k in p0)
        print(f"  ROW_CHUNK={rc:5d}: {t1:6.2f}s (base {t0b:6.2f}s) ident={ident}")
    # GPTQ_BLOCK on weight GPTQ (c8192 weight is the heavy one) -> time only
    g8 = load_group("c8192_n8192")
    w8 = base.dequantize_nvfp4(g8["weight"][0], g8["weight"][1]).float()
    acts = [base.dequantize_nvfp4(aq, as_).float() for aq, as_ in g8["calib_activation_list"]]
    Hs = torch.zeros(8192, 8192)
    for a in acts[:-1]:
        Hs += a.T @ a
    U = base._upper_cholesky_inv(Hs)
    pw = base._quantize_weighted(w8, torch.ones(1, 8192))
    unit = base._params_unit_flat(pw)
    print("[sweep] weight _gptq_quantize_values(8192x8192) GPTQ_BLOCK:")
    ref = None
    for gb in (64, 128, 256, 512):
        base.GPTQ_BLOCK = gb
        t0 = time.perf_counter()
        q = base._gptq_quantize_values(w8, unit, U)
        t1 = time.perf_counter() - t0
        if ref is None:
            ref = q
        d = "0" if torch.equal(ref, q) else f"{(q - ref).abs().max().item():.2e}"
        print(f"  GPTQ_BLOCK={gb:4d}: {t1:6.2f}s maxdiff={d}")
    # dyn GPTQ overhead probe: numpy vs torch on (T, 8192)
    base.GPTQ_BLOCK = 128
    print("[sweep] dyn GPTQ (T, 8192) torch vs numpy:")
    for T in (10, 128, 512, 1024):
        x = torch.randn(T, 8192)
        u = torch.rand(T, 8192) + 0.5
        t0 = time.perf_counter()
        a = base._gptq_quantize_values(x, u, U)
        t1 = time.perf_counter() - t0
        t0 = time.perf_counter()
        b = _gptq_np(x, u, U, 128)
        t2 = time.perf_counter() - t0
        print(f"  T={T:5d}: torch {t1:6.3f}s numpy {t2:6.3f}s "
              f"ident={torch.equal(a, b)}")


def cmd_chol():
    """f5: can U (upper chol of H^-1) be had from ONE decomposition?"""
    torch.manual_seed(3)
    for n in (256, 1024):
        A = torch.randn(4 * n, n)
        H = A.T @ A
        d = H.diagonal().mean() * 0.1
        Hd = H + torch.eye(n) * d
        t0 = time.perf_counter()
        U_ref = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hd)), upper=True)
        t_ref = time.perf_counter() - t0
        # candidate: reverse-order Cholesky: chol(P H P) = Lq, U_cand = P Lq^T P
        P = torch.flip(torch.eye(n), [0])
        t0 = time.perf_counter()
        Lq = torch.linalg.cholesky(P @ Hd @ P)
        U_c = P @ Lq.T @ P
        t_c = time.perf_counter() - t0
        err = (U_c @ U_c.T - torch.linalg.inv(Hd)).norm() / U_ref.norm()
        match = torch.allclose(U_c, U_ref, atol=1e-4 * U_ref.abs().max())
        print(f"[chol] n={n}: ref {t_ref*1e3:.1f}ms cand {t_c*1e3:.1f}ms "
              f"rel-err(U U^T vs H^-1)={err:.2e} equal-to-ref={match}")
        print(f"        U U^T = H^-1 ? {torch.allclose(U_c @ U_c.T, torch.linalg.inv(Hd), rtol=1e-3)}; "
              f"U^T U = H^-1 ? {torch.allclose(U_c.T @ U_c, torch.linalg.inv(Hd), rtol=1e-3)}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    if mode == "run":
        cmd_run(sys.argv[2:] or ["c2048_n8192"])
    elif mode == "unit":
        cmd_unit()
    elif mode == "sweep":
        cmd_sweep()
    elif mode == "chol":
        cmd_chol()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
