"""C=4096 bf16-Gram carry probe (envelope test between 48 and 192 MiB).

Proven state envelope: 48 MiB total (bf16 grams 32 + fp32 u_act 16 at C=2048)
passes since v16; 192 MiB (fp32 grams 128 + u_act 64 at C=4096) WA'd in
carry3. Unknown zone: 128 MiB (bf16 grams 64 + fp32 u_act 64 at C=4096).

This probe: linear groups with 2048 < C <= 4096 are hash-split (float64 sum,
the 1438 lesson): hash even -> carry bf16 grams + dynamic refinement (same
sweeps schedule); hash odd -> current v20 behavior (v14 path). C <= 2048
groups (incl. E3) are bit-identical to v20. Decode from new WA groups:
  none          -> 128 MiB envelope OK -> v21 extends to all C=4096 (+300-900)
  ~3 new WA     -> envelope < 128 MiB -> try 96 MiB (u_act bf16) or low-rank
  6 new WA      -> hard cap near 64 MiB
Timing: +10-15s online (refinement on ~3 groups), fits the 226s margin.
"""
import os
import zipfile
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "example", "solution", "solution.py")
OUT_DIR = os.path.join(ROOT, "probe", "c4096_probe")
OUT_ZIP = os.path.join(ROOT, "dist", "probe_c4096.zip")

src = open(SRC, encoding="utf-8").read()

old = """    gw = gwf = None
    if C <= REFINE_MAX_C:
        try:
            weight_params, q_used = _refine_weight_values(
                w_final, q_used, weight_params, acts_s, tf_final)
        except Exception:
            pass
        try:"""
new = """    gw = gwf = None
    # envelope probe: hash-even C<=4096 groups carry bf16 grams (128 MiB total
    # state incl u_act); hash-odd and C>4096 stay on the v20 path
    _e4 = (C <= REFINE_MAX_C
           or (C <= 4096 and int(w.double().abs().sum().item() * 1e3) % 2 == 0))
    if _e4:
        try:
            if C <= REFINE_MAX_C:
                weight_params, q_used = _refine_weight_values(
                    w_final, q_used, weight_params, acts_s, tf_final)
        except Exception:
            pass
        try:"""
assert old in src, "calibration gate anchor"
src = src.replace(old, new)

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "solution.py")
open(out_path, "w", encoding="utf-8").write(src)

spec = importlib.util.spec_from_file_location("probe_c4096", out_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import torch
p = mod.hif4_dynamic_quantize_v(torch.randn(4, 256), torch.rand(4, 16), 2, 128, None)
assert p["mant"].shape[-1] == 4
print("probe module OK")

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(out_path, "solution.py")
print("wrote", OUT_ZIP)
