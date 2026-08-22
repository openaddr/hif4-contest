"""Shared harness for audit round 2 (v25).

Loads the CURRENT solution as baseline (never modified), provides variant
loading (monkeypatch / textual patch of an in-memory copy), data build/load,
and bit-identity comparators. Mirrors dev/audit/exp_speed.py conventions.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUDIT2 = os.path.join(ROOT, "dev", "audit2")
DATA_DIR = os.path.join(AUDIT2, "data")
RESULTS = os.path.join(AUDIT2, "results")
SOLUTION = os.path.join(ROOT, "example", "solution", "solution.py")
MINI = os.path.join(ROOT, "example", "mini_sample")
sys.path.insert(0, os.path.join(ROOT, "dev"))
import synth  # noqa: E402

_SRC = None
_vn = [0]

# name, C, N, calib tokens, test tokens (spread=0.5, w_spread=0.3 as in audit1)
CONFIGS = [
    ("c1024_n1024", 1024, 1024, (10, 128, 512, 1024), (10, 128, 512, 1024, 1024)),
    ("c2048_n8192", 2048, 8192, (10, 128, 512, 1024), (10, 128, 512, 1024, 1024)),
    ("c3072_n3072", 3072, 3072, (10, 128, 512, 1024), (10, 128, 512, 1024, 1024)),
    ("c4096_n4096", 4096, 4096, (10, 128, 512, 1024), (10, 128, 512, 1024, 1024)),
]


def src_text():
    global _SRC
    if _SRC is None:
        with open(SOLUTION, encoding="utf-8") as f:
            _SRC = f.read()
    return _SRC


def load_variant(patch_src=None, **attrs):
    _vn[0] += 1
    p = os.path.join(AUDIT2, f"_variant_{_vn[0]}.py")
    with open(p, "w", encoding="utf-8") as f:
        f.write(src_text() if patch_src is None else patch_src)
    spec = importlib.util.spec_from_file_location(f"_a2var_{_vn[0]}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def build(force=False):
    import time
    os.makedirs(DATA_DIR, exist_ok=True)
    for i, (name, C, N, cal_T, test_T) in enumerate(CONFIGS):
        path = os.path.join(DATA_DIR, f"{name}.pt")
        if os.path.exists(path) and not force:
            continue
        t0 = time.perf_counter()
        g = synth.make_linear_group(3100 + 7 * i, N, C, tokens=cal_T,
                                    spread=0.5, outlier_p=0.0, w_spread=0.3)
        g2 = synth.make_linear_group(9300 + 7 * i, N, C, tokens=test_T,
                                     spread=0.5, outlier_p=0.0, w_spread=0.3)
        g["test_activation_list"] = g2["test_activation_list"]
        torch.save(g, path)
        print(f"[build] {name}: {time.perf_counter() - t0:.1f}s, "
              f"{os.path.getsize(path) / 2 ** 20:.0f} MiB", flush=True)
    print("[build] done")


def load_group(name):
    return torch.load(os.path.join(DATA_DIR, f"{name}.pt"),
                      weights_only=True, map_location="cpu")


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


def run_linear(sol, g):
    """Calibration + all dynamic test calls; returns (out, params, t_cal, t_dyn)."""
    import time
    torch.manual_seed(0)
    t0 = time.perf_counter()
    out = sol.hif4_calibration_and_quantize_weight(
        g["weight"][0], g["weight"][1], g["calib_activation_list"])
    t_cal = time.perf_counter() - t0
    st = out["activation_state"]
    t_dyn = 0.0
    ps = []
    for pair in g["test_activation_list"]:
        t0 = time.perf_counter()
        p = sol.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
        t_dyn += time.perf_counter() - t0
        ps.append(p)
    return out, ps, t_cal, t_dyn


def check_linear(base, var, g, tag):
    """Full bit-identity check of variant vs base on one group."""
    ob, pb, _, _ = run_linear(base, g)
    ov, pv, _, _ = run_linear(var, g)
    ok = (eq_params(ob["weight_params"], ov["weight_params"])
          and eq_state(ob["activation_state"], ov["activation_state"])
          and all(eq_params(a, b) for a, b in zip(pb, pv)))
    print(f"[bitid] {tag}: {'PASS' if ok else 'FAIL'}")
    return ok
