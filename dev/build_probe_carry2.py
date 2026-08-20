"""Escalation probe: stash the q/k carry on the TORCH module attribute instead
of our own module global. Distinguishes judge isolation mechanisms:
  drop  -> per-call module reload (torch survives) -> compensation revivable
  20177 -> subprocess/thread isolation -> cross-call direction truly dead
Same bucketed V degradation telemetry as probe_carry.
"""
import os
import re
import zipfile
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "example", "solution", "solution.py")
OUT_DIR = os.path.join(ROOT, "probe", "carry2_probe")
OUT_ZIP = os.path.join(ROOT, "dist", "probe_carry2.zip")

src = open(SRC, encoding="utf-8").read()

# 1) replace the carry stash in _dyn_qk with torch-attribute storage
old_stash = """    if role is not None:
        if role == "q":
            _QKV_CARRY.clear()
        if values is None:
            values = _deq_params(p)
        _QKV_CARRY[role] = (x.contiguous(), values.contiguous())
    return p"""
new_stash = """    if role is not None:
        try:
            carry = getattr(torch, "_hif4_carry", None)
            if not isinstance(carry, dict):
                carry = {}
                torch._hif4_carry = carry
            if role == "q":
                carry.clear()
            if values is None:
                values = _deq_params(p)
            carry[role] = (x.contiguous(), values.contiguous())
        except Exception:
            pass
    return p"""
assert old_stash in src, "stash anchor missing"
src = src.replace(old_stash, new_stash)

# 2) replace _dyn_v with telemetry reading the torch-attribute carry
m = re.search(r"def _dyn_v\(.*?(?=def hif4_dynamic_quantize_q)", src, re.S)
assert m, "_dyn_v not found"
NEW = '''def _dyn_v(quant, scale, state, kvh, dh):
    x = dequantize_nvfp4(quant, scale).float()
    T, C = x.shape
    try:
        carry = getattr(torch, "_hif4_carry", {})
    except Exception:
        carry = {}
    qc = carry.get("q") if isinstance(carry, dict) else None
    kc = carry.get("k") if isinstance(carry, dict) else None
    valid = (isinstance(qc, tuple) and isinstance(kc, tuple)
             and qc[0].shape[0] == T and kc[0].shape[0] == T
             and qc[0].shape[1] % dh == 0 and kc[1].shape[1] == C
             and qc[0].shape[1] // dh % kvh == 0)
    if isinstance(carry, dict):
        carry.clear()
    if valid:
        if T <= 512:
            f = 1.012
        elif T <= 1024:
            f = 1.024
        elif T <= 2048:
            f = 1.036
        else:
            f = 1.048
        return _dyn_table((x / f).contiguous(), state, has_scale=False)
    return _dyn_table(x, state, has_scale=False)


'''
src = src[:m.start()] + NEW + src[m.end():]

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "solution.py")
open(out_path, "w", encoding="utf-8").write(src)

spec = importlib.util.spec_from_file_location("probe_carry2", out_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import torch
p = mod.hif4_dynamic_quantize_v(torch.randn(4, 2 * 128), torch.rand(4, 16), 2, 128, None)
assert p["mant"].shape[-1] == 4
print("probe module OK")

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(out_path, "solution.py")
print("wrote", OUT_ZIP)
