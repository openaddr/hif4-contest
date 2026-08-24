"""Smoke test: base mode bit-identical to original; ff modes run, s sane."""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "dev"))
import synth  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ORIG = load(os.path.join(ROOT, "example", "solution", "solution.py"), "orig")
COPY = load(os.path.join(HERE, "solution.py"), "copy")

g = synth.make_linear_group(3, 1024, 1024, tokens=(128, 512), spread=0.5)

# --- 1) base bit-identity ---
torch.manual_seed(0)
o = ORIG.hif4_calibration_and_quantize_weight(*g["weight"], g["calib_activation_list"])
torch.manual_seed(0)
c = COPY.hif4_calibration_and_quantize_weight(*g["weight"], g["calib_activation_list"])
same = all(torch.equal(o["weight_params"][k], c["weight_params"][k]) for k in o["weight_params"])
s_same = torch.equal(o["activation_state"]["s"], c["activation_state"]["s"])
print("base bit-identical weight_params:", same, "| s equal:", s_same)
assert same and s_same

# --- 2) ff modes run ---
for mode in ("ff_icm", "ff_bal", "mag_scan"):
    COPY.SMOOTH_MODE = mode
    COPY.SMOOTH_DEBUG.clear()
    torch.manual_seed(0)
    t0 = time.perf_counter()
    r = COPY.hif4_calibration_and_quantize_weight(*g["weight"], g["calib_activation_list"])
    dt = time.perf_counter() - t0
    s = r["activation_state"]["s"]
    print(f"{mode}: cal {dt:.1f}s debug={COPY.SMOOTH_DEBUG} "
          f"s range [{s.min():.3f},{s.max():.3f}] std {s.log().std():.3f}")
    # dynamic call smoke
    p = COPY.hif4_dynamic_quantize_activation(g["test_activation_list"][0][0],
                                              g["test_activation_list"][0][1],
                                              r["activation_state"])
    ok = bool(torch.isfinite(p["mant"]).all())
    print(f"   dynamic ok={ok}")
COPY.SMOOTH_MODE = "base"
print("SMOKE OK")
