"""carry3 probe: carry vs refine ablation by float64-weight-hash buckets.

probe_cband readout: EXACT same score/time as rev4 -> no judge group's
computation changed -> failing groups are either (i) C<=2048 refined groups
(math guilty; C grid powers of 2 -> (2048,4096] band empty) or (ii) C=4096
groups whose dynamic R>1024 (refinement never ran for them; the Gram CARRY
alone breaks them). mini sample: C=2048, R<=1024 (refinement can run).

This probe splits linear groups 3-way (float64 hash -- the 1438 lesson):
  bucket 0: v14 exact       (no Gram carry, no refinement)
  bucket 1: Grams carried, refinement OFF   (pure carry test)
  bucket 2: Grams carried + refinement      (rev4 behavior)

Decode from which of the 6 failing groups still WA:
  fail in bucket 1 (and 2) -> CARRY guilty (S1) -> v16: bf16 Grams or drop
  fail in bucket 2 only    -> refinement MATH guilty (S2) -> v16: drop/guard
  none fail (0.1% chance)  -> interaction/nondeterminism, rethink
Attention groups untouched (v14 behavior).
"""
import os
import zipfile
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "example", "solution", "solution.py")
OUT_DIR = os.path.join(ROOT, "probe", "carry3_probe")
OUT_ZIP = os.path.join(ROOT, "dist", "probe_carry3.zip")

src = open(SRC, encoding="utf-8").read()

# 1) bucket definition right after calib entry (float64 sum: the 1438 lesson)
old = """    torch.manual_seed(0)  # deterministic calibration subsampling
    w = dequantize_nvfp4(weight_quant, weight_scale).float()"""
new = """    torch.manual_seed(0)  # deterministic calibration subsampling
    w = dequantize_nvfp4(weight_quant, weight_scale).float()
    _cb = int(w.double().abs().sum().item() * 1e3) % 3  # carry3 bucket"""
assert old in src, "seed anchor"
src = src.replace(old, new)

# 2) bucket 0 carries nothing
old = """    gw = gwf = None
    if C <= REFINE_MAX_C:"""
new = """    gw = gwf = None
    if C <= REFINE_MAX_C and _cb != 0:"""
assert old in src, "gram gate anchor"
src = src.replace(old, new)

# 3) refinement suppressed for bucket 1 via an inert bool flag in the state
old = """    # ---- lattice refinement on the final values (transformed space) ----
    if isinstance(activation_state, dict) and R <= REFINE_T_MAX:"""
new = """    # ---- lattice refinement on the final values (transformed space) ----
    if (isinstance(activation_state, dict) and R <= REFINE_T_MAX
            and activation_state.get("rz") is not True):"""
assert old in src, "dynamic refine anchor"
src = src.replace(old, new)

old = """        "gw": gw,
        "gwf": gwf,
    }"""
new = """        "gw": gw,
        "gwf": gwf,
        "rz": _cb == 1,  # carry3: carried but refinement suppressed
    }"""
assert old in src, "state dict anchor"
src = src.replace(old, new)

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "solution.py")
open(out_path, "w", encoding="utf-8").write(src)

spec = importlib.util.spec_from_file_location("probe_carry3", out_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import torch
p = mod.hif4_dynamic_quantize_v(torch.randn(4, 256), torch.rand(4, 16), 2, 128, None)
assert p["mant"].shape[-1] == 4
print("probe module OK")

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(out_path, "solution.py")
print("wrote", OUT_ZIP)
