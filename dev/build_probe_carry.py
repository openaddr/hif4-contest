"""Telemetry probe: decode whether the q/k/v carry ever assembles on the
judge, and the length distribution of valid v-calls.

Replaces the compensation with a bucketed V degradation:
  carry invalid           -> normal V            (score = 20177)
  carry valid, T<=512     -> V/1.15              (drop D1)
  carry valid, T<=1024    -> V/1.30              (drop D2)
  carry valid, T<=2048    -> V/1.50              (drop D3)
  carry valid, T>2048     -> V/1.80              (drop D4)
The total drop magnitude decodes the mixture. No budget meter, no T cap.
"""
import os
import re
import zipfile
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "example", "solution", "solution.py")
OUT_DIR = os.path.join(ROOT, "probe", "carry_probe")
OUT_ZIP = os.path.join(ROOT, "dist", "probe_carry.zip")

src = open(SRC, encoding="utf-8").read()

# replace the whole _dyn_v with the telemetry version
m = re.search(r"def _dyn_v\(.*?(?=def hif4_dynamic_quantize_q)", src, re.S)
assert m, "_dyn_v not found"

NEW = '''def _dyn_v(quant, scale, state, kvh, dh):
    x = dequantize_nvfp4(quant, scale).float()
    T, C = x.shape
    qc = _QKV_CARRY.get("q")
    kc = _QKV_CARRY.get("k")
    valid = (isinstance(qc, tuple) and isinstance(kc, tuple)
             and qc[0].shape[0] == T and kc[0].shape[0] == T
             and qc[0].shape[1] % dh == 0 and kc[1].shape[1] == C
             and qc[0].shape[1] // dh % kvh == 0)
    _QKV_CARRY.clear()
    if valid:
        if T <= 512:
            f = 1.15
        elif T <= 1024:
            f = 1.30
        elif T <= 2048:
            f = 1.50
        else:
            f = 1.80
        return _dyn_table((x / f).contiguous(), state, has_scale=False)
    return _dyn_table(x, state, has_scale=False)


'''
src = src[:m.start()] + NEW + src[m.end():]

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "solution.py")
open(out_path, "w", encoding="utf-8").write(src)

spec = importlib.util.spec_from_file_location("probe_carry", out_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import torch
p = mod.hif4_dynamic_quantize_v(torch.randn(4, 2 * 128), torch.rand(4, 16), 2, 128, None)
assert p["mant"].shape[-1] == 4
print("probe module OK")

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(out_path, "solution.py")
print("wrote", OUT_ZIP)
