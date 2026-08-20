"""Rotation-STRUCTURE ablation for the linear pipeline (offline, read-only w.r.t. solution.py).

Question: is the current block-diagonal Hadamard rotation (H*diag(sign)) the best
exact-invariant rotation structure? Variants (all exact orthogonal, shared by
weight and activation side so x.w is mathematically invariant; only the VALUES
seen by the quantizer change):

  V0  current:        R_b = H @ diag(d_b)                    (per 64-block, seed 777+b)
  V1  block perm:     R_b = P_b @ H @ diag(d_b)              (P_b fixed random 64-perm, seed 777+b)
  V2  difficulty sort: permute channels globally by difficulty
                       sqrt(E[x_j^2] * sum_i w_ij^2) DESC, then V0 on the
                       regrouped 64-blocks (P_diff @ blkdiag(H D))
  V3  double Hadamard: R_b = H @ diag(d_b) @ H               (dense orthogonal)

Scoring follows dev/diag3.py: for each test activation set, improvement of the
full pipeline (calibration + dynamic quantization) over the exact-paper alg1
baseline, in percentage points:
    score_pp = (mse_alg1 - mse_pipeline) / mse_alg1 * 100

Data: example/mini_sample/linear.pt (5 tests) + two synthetic groups from
dev/synth.make_linear_group.

Usage: /c/App/env/Python/python.exe dev/rotstruct.py [--quick]
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import hif4  # noqa: E402
import synth  # noqa: E402
import variants as V  # noqa: E402


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load_mod(os.path.join(ROOT, "..", "example", "solution", "solution.py"), "sol_rotstruct")
ROT_ORIG = S._rot_blocks  # keep the current pipeline's V0

# ---------------------------------------------------------------------------
# rotation variant machinery (deterministic, cached per block count)
# ---------------------------------------------------------------------------


def _hadamard64():
    H = torch.tensor([[1.0]])
    while H.shape[0] < 64:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / 8.0


_H64 = _hadamard64()
_RM_CACHE: dict = {}
_V2_PERM: torch.Tensor | None = None  # set per dataset before calibration


def _signs(nb):
    d = torch.empty(nb, 64)
    for b in range(nb):
        g = torch.Generator().manual_seed(777 + b)
        d[b] = (torch.rand(64, generator=g) < 0.5).float() * 2 - 1
    return d


def _rm0(nb):
    """V0 block matrices: H @ diag(d_b)."""
    if "v0" not in _RM_CACHE or _RM_CACHE["v0"].shape[0] != nb:
        _RM_CACHE["v0"] = _H64.unsqueeze(0) * _signs(nb).unsqueeze(1)
    return _RM_CACHE["v0"]


def _rm1(nb):
    """V1 block matrices: P_b @ H @ diag(d_b) (per-block fixed random row permutation)."""
    key = ("v1", nb)
    if key not in _RM_CACHE:
        idx = torch.empty(nb, 64, dtype=torch.long)
        for b in range(nb):
            g = torch.Generator().manual_seed(777 + b)
            idx[b] = torch.randperm(64, generator=g)
        _RM_CACHE[key] = _rm0(nb)[torch.arange(nb).unsqueeze(1), idx]
    return _RM_CACHE[key]


def _rm3(nb):
    """V3 block matrices: (H @ diag(d_b)) @ H  -- dense orthogonal."""
    key = ("v3", nb)
    if key not in _RM_CACHE:
        _RM_CACHE[key] = torch.matmul(_rm0(nb), _H64.unsqueeze(0))
    return _RM_CACHE[key]


def _apply(x, Rm):
    R, C = x.shape
    nb = C // 64
    xb = x.reshape(R, nb, 64)
    return torch.einsum("rbd,bde->rbe", xb, Rm).reshape(R, C)


def rot_v0(x):
    return _apply(x, _rm0(x.shape[1] // 64))


def rot_v1(x):
    return _apply(x, _rm1(x.shape[1] // 64))


def rot_v2(x):
    return rot_v0(x[:, _V2_PERM].contiguous())


def rot_v3(x):
    return _apply(x, _rm3(x.shape[1] // 64))


VARIANTS = [
    ("V0", lambda x: ROT_ORIG(x)),  # exact current pipeline function
    ("V1", rot_v1),
    ("V2", rot_v2),
    ("V3", rot_v3),
]


def difficulty_perm(lin):
    """Channel order by difficulty = sqrt(E[x_j^2] * sum_i w_ij^2) DESC.

    Computed from RAW calib activations + RAW weight; invariant to per-channel
    smoothing s (s^2 * 1/s^2 cancels), so it applies equally to the smoothed
    space where the rotation actually runs.
    """
    w_ref = hif4.dequantize_nvfp4(*lin["weight"]).float()
    ex = torch.zeros(w_ref.shape[1], dtype=torch.float32)
    n = 0
    for q, s in lin["calib_activation_list"]:
        a = hif4.dequantize_nvfp4(q, s).float()
        ex += (a * a).sum(dim=0)
        n += a.shape[0]
    ex /= max(n, 1)
    wcol = (w_ref * w_ref).sum(dim=0)
    diff = torch.sqrt(ex * wcol)
    return torch.argsort(diff, descending=True, stable=True)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def run_variant(rot, lin, v2_perm):
    """Full pipeline with S._rot_blocks monkeypatched to `rot`; returns per-test
    scores (pp vs alg1), calibration wall time, dynamic wall time, state flags."""
    global _V2_PERM
    _V2_PERM = v2_perm
    S._rot_blocks = rot
    torch.manual_seed(0)

    w_ref = hif4.dequantize_nvfp4(*lin["weight"]).float()
    w_std = V.deq(V.quant_alg1(w_ref.float()))

    t0 = time.perf_counter()
    cal = S.hif4_calibration_and_quantize_weight(*lin["weight"], lin["calib_activation_list"])
    t_cal = time.perf_counter() - t0
    w_play = hif4.hif4_dequantize(cal["weight_params"])
    state = cal["activation_state"]

    scores = []
    t0 = time.perf_counter()
    for q, s in lin["test_activation_list"]:
        x_ref = hif4.dequantize_nvfp4(q, s)
        ref = hif4.linear_ref(x_ref, w_ref)
        x_std = V.deq(V.quant_alg1(x_ref.float()))
        mse_std = ((hif4.linear_ref(x_std, w_std) - ref) ** 2).mean().item()
        p = S.hif4_dynamic_quantize_activation(q, s, state)
        x_play = hif4.hif4_dequantize(p)
        mse_play = ((hif4.linear_ref(x_play, w_play) - ref) ** 2).mean().item()
        scores.append((mse_std - mse_play) / mse_std * 100.0)
    t_dyn = time.perf_counter() - t0
    return scores, t_cal, t_dyn, state["mode"], state["g"], state.get("s") is not None


def self_check():
    # block orthogonality
    eye = torch.eye(64).unsqueeze(0)
    for nm, rm in (("v0", _rm0(5)), ("v1", _rm1(5)), ("v3", _rm3(5))):
        e = (torch.matmul(rm, rm.transpose(1, 2)) - eye).abs().max().item()
        assert e < 1e-4, (nm, e)
    # exact dot-product invariance on random tensors
    global _V2_PERM
    torch.manual_seed(0)
    x = torch.randn(64, 256)
    y = torch.randn(64, 256)
    _V2_PERM = torch.randperm(256)
    ref = x @ y.T
    for nm, rot in VARIANTS:
        e = ((rot(x) @ rot(y).T) - ref).abs().max().item()
        assert e < 1e-3, (nm, e)
    # V0 replica == original solution rotation
    e = (rot_v0(x) - ROT_ORIG(x)).abs().max().item()
    assert e < 1e-4, e
    print("self-check: orthogonality + exact invariance + V0 replica == OK", flush=True)


def main():
    quick = "--quick" in sys.argv
    self_check()

    datasets = [
        ("mini", torch.load(os.path.join(ROOT, "..", "example", "mini_sample", "linear.pt"),
                            weights_only=True, map_location="cpu")[0]),
        ("synA_out", synth.make_linear_group(4, 4096, 2048, spread=0.5, outlier_p=0.003)),
        ("synB", synth.make_linear_group(1, 2048, 1024, spread=0.3)),
    ]
    variants = VARIANTS[:2] if quick else VARIANTS
    results = {}

    for dname, lin in datasets:
        v2p = difficulty_perm(lin)
        print(f"\n=== dataset {dname}: W {tuple(lin['weight'][0].shape)}, "
              f"{len(lin['test_activation_list'])} tests ===", flush=True)
        for vname, rot in variants:
            scores, t_cal, t_dyn, mode, g, has_s = run_variant(rot, lin, v2p)
            results[(dname, vname)] = dict(scores=scores, t_cal=t_cal, t_dyn=t_dyn,
                                           mode=mode, g=g)
            detail = " ".join(f"t{i}={s:+6.2f}" for i, s in enumerate(scores))
            print(f"[{dname:8s}] {vname} mode={mode} g={g} cal={t_cal:5.1f}s "
                  f"dyn={t_dyn:5.1f}s avg={sum(scores)/len(scores):+6.2f}pp | {detail}",
                  flush=True)

    # ---------------- summary ----------------
    vnames = [v for v, _ in variants]
    print("\n" + "=" * 78)
    print("SCORE TABLE (pp improvement vs exact-alg1 baseline; higher is better)")
    header = f"{'variant':8s}" + "".join(f"{d:>22s}" for d, _ in datasets) + f"{'overall':>10s}"
    print(header)
    for vn in vnames:
        cells = []
        allsc = []
        for dname, _ in datasets:
            sc = results[(dname, vn)]["scores"]
            avg = sum(sc) / len(sc)
            base = sum(results[(dname, "V0")]["scores"]) / len(results[(dname, "V0")]["scores"])
            cells.append(f"{avg:+8.2f} ({avg-base:+5.2f})")
            allsc.extend(sc)
        print(f"{vn:8s}" + "".join(f"{c:>22s}" for c in cells)
              + f"{sum(allsc)/len(allsc):+10.2f}")

    print("\nTIMING (calibration s / dynamic s)")
    for vn in vnames:
        cells = []
        for dname, _ in datasets:
            r = results[(dname, vn)]
            cells.append(f"{r['t_cal']:5.1f}/{r['t_dyn']:5.1f}")
        print(f"{vn:8s}" + "".join(f"{c:>22s}" for c in cells))

    print("\nVERDICT (>= +0.3pp vs V0 on mini avg AND synth avg -> worth deploying)")
    for dname, _ in datasets:
        print(f"  {dname}: " + "  ".join(
            f"{vn}={sum(results[(dname, vn)]['scores'])/len(results[(dname, vn)]['scores']):+.2f}"
            for vn in vnames))
    for vn in vnames[1:]:
        mini_d = (sum(results[("mini", vn)]["scores"]) / len(results[("mini", vn)]["scores"])
                  - sum(results[("mini", "V0")]["scores"]) / len(results[("mini", "V0")]["scores"]))
        syn_all, syn0_all = [], []
        for dname in ("synA_out", "synB"):
            syn_all.extend(results[(dname, vn)]["scores"])
            syn0_all.extend(results[(dname, "V0")]["scores"])
        syn_d = sum(syn_all) / len(syn_all) - sum(syn0_all) / len(syn0_all)
        out_d = (sum(results[("synA_out", vn)]["scores"]) / len(results[("synA_out", vn)]["scores"])
                 - sum(results[("synA_out", "V0")]["scores"]) / len(results[("synA_out", "V0")]["scores"]))
        worth = "WORTH DEPLOYING" if (mini_d >= 0.3 and syn_d >= 0.3) else "not worth it"
        print(f"  {vn}: mini {mini_d:+.2f}pp | synth(outlier grp {out_d:+.2f}pp, "
              f"both {syn_d:+.2f}pp) -> {worth}")


if __name__ == "__main__":
    main()
