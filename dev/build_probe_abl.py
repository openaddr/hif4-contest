"""Component-ablation probe: measures the ONLINE value of each pipeline
component by disabling one component per group (bucket from data hash).

Linear bucket = int(w.abs().sum() * 1e3) % 4:
  0 -> rotation forced OFF        (mode=0)
  1 -> act-GPTQ forced OFF        (no u_act/g)
  2 -> smoothing forced OFF       (s = ones)
  3 -> control (full pipeline)
Attention bucket = int(q.abs().sum() * 1e3) % 2:
  0 -> Q/K rotation forced OFF
  1 -> control

All ablations are calibration-side; the dynamic side follows via state.
Decode: probe score vs 20183 -> per-component online value
(each linear bucket ~12.5 groups ~62 cases; attention bucket ~25 groups).
"""
import os
import zipfile
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "example", "solution", "solution.py")
OUT_DIR = os.path.join(ROOT, "probe", "abl_probe")
OUT_ZIP = os.path.join(ROOT, "dist", "probe_ablation.zip")

src = open(SRC, encoding="utf-8").read()

# --- linear: bucket definition right after the calib entry ---
old = """    torch.manual_seed(0)  # deterministic calibration subsampling
    w = dequantize_nvfp4(weight_quant, weight_scale).float()"""
new = """    torch.manual_seed(0)  # deterministic calibration subsampling
    w = dequantize_nvfp4(weight_quant, weight_scale).float()
    _ab = int(w.abs().sum().item() * 1e3) % 4  # ablation bucket (probe)"""
assert old in src, "seed anchor"
src = src.replace(old, new)

# --- linear: bucket logic around the transform choice ---
old = """    # ---- transform choice: {0: none, 1: rotation} ----
    mode = 0
    Uw = None
    xh_pick = None
    if R > 64 and len(acts_s) >= 2 and acts_s[-1].shape[0] >= 8:"""
new = """    # ---- transform choice: {0: none, 1: rotation} ----
    mode = 0
    Uw = None
    xh_pick = None
    if _ab != 0 and R > 64 and len(acts_s) >= 2 and acts_s[-1].shape[0] >= 8:"""
assert old in src, "linear transform anchor"
src = src.replace(old, new)

# smoothing off for bucket 2: replace the alpha search result
old = """    s = torch.exp(logm * best_alpha)
    w_s = w / s
    acts_s = [a * s for a in acts_raw]"""
new = """    if _ab == 2:
        best_alpha = 0.0
    s = torch.exp(logm * best_alpha)
    w_s = w / s
    acts_s = [a * s for a in acts_raw]"""
assert old in src, "smoothing anchor"
src = src.replace(old, new)

# act-GPTQ off for bucket 1: gate the whole act-GPTQ block
old = """    # ---- activation-side GPTQ with act-order ----
    u_act = None
    gptq_act = 0
    order = None
    if xh_pick is not None:"""
new = """    # ---- activation-side GPTQ with act-order ----
    u_act = None
    gptq_act = 0
    order = None
    if xh_pick is not None and _ab != 1:"""
assert old in src, "actgptq anchor"
src = src.replace(old, new)

# --- attention: rotation bucket ---
old = """    loss_off = run(q, k)
    rot = 0
    if R is not None:"""
new = """    _abq = int(q.abs().sum().item() * 1e3) % 2
    loss_off = run(q, k)
    rot = 0
    if R is not None and _abq != 0:"""
assert old in src, "attn rot anchor"
src = src.replace(old, new)

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "solution.py")
open(out_path, "w", encoding="utf-8").write(src)

spec = importlib.util.spec_from_file_location("probe_abl", out_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import torch
p = mod.hif4_dynamic_quantize_v(torch.randn(4, 256), torch.rand(4, 16), 2, 128, None)
assert p["mant"].shape[-1] == 4
print("probe module OK")

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(out_path, "solution.py")
print("wrote", OUT_ZIP)
