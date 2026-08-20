"""Assemble the v9-linear decomposition probe:
current solution.py linear path + validated alg1 attention stubs.
Output: dist/probe_v9_linear.zip
"""
import os
import re
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "example", "solution", "solution.py")
OUT_DIR = os.path.join(ROOT, "probe", "v9_lin_probe")
OUT_ZIP = os.path.join(ROOT, "dist", "probe_v9_linear.zip")

src = open(SRC, encoding="utf-8").read()

ALG1 = '''

_E6M2_MIN = 2.0 ** -48
_E6M2_MAX = 49152.0


def _encode_e6m2(x):
    xc = x.clamp(min=1e-30)
    e = torch.floor(torch.log2(xc))
    m = torch.round(xc * (2.0 ** (2 - e)))
    m = torch.clamp(m, 4, 8)
    out = m * (2.0 ** (e - 2))
    return torch.clamp(out, _E6M2_MIN, _E6M2_MAX)


def _quantize_alg1(x):
    shape = tuple(x.shape)
    C = shape[-1]
    prefix = shape[:-1]
    xf = x.detach().float()
    xr = xf.reshape(*prefix, C // 64, 8, 2, 4)
    ax = xr.abs()

    v16 = ax.amax(dim=-1)
    v8 = v16.amax(dim=-1)
    vmax = v8.amax(dim=-1)

    sf = _encode_e6m2(vmax / 7.0)
    rec = torch.reciprocal(sf.float())
    e1_8 = (v8 * rec.unsqueeze(-1) >= 4.0).to(xf.dtype)
    e1_8_g = e1_8.unsqueeze(-1)
    e1_16 = ((v16 * rec.unsqueeze(-1).unsqueeze(-1) * (2.0 ** (-e1_8_g))) >= 2.0).to(xf.dtype)
    x_scaled = (
        xr * rec[..., None, None, None]
        * (2.0 ** (-e1_8[..., None, None]))
        * (2.0 ** (-e1_16[..., None]))
    )
    sign = torch.sign(x_scaled)
    qi = torch.floor(x_scaled.abs() * 4.0 + 0.5).clamp(0, 7)
    mant = qi / 4.0
    sign = torch.where(mant == 0, torch.zeros_like(sign), sign)
    return {
        "scale_factor": sf[..., None, None, None],
        "scale_lv2": (2.0 ** e1_8)[..., None, None],
        "scale_lv3": (2.0 ** e1_16)[..., None],
        "sign": sign,
        "mant": mant,
    }
'''

# 1) replace the attention calibration with a trivial stub
m = re.search(r"def hif4_calibration_attention\(.*?\n(?=def _attention_out)", src, re.S)
assert m, "calibration_attention not found"
src = src[:m.start()] + (
    "def hif4_calibration_attention(calib_qkv_list, q_num_heads, kv_num_heads, head_dim):\n"
    '    """Probe: alg1 attention (judge baseline)."""\n'
    '    return {"q_state": None, "k_state": None, "v_state": None}\n\n\n'
) + src[m.end():]

# 2) replace everything from _dyn_table to EOF (dyn helpers + wrappers) with alg1 versions
m = re.search(r"def _dyn_table", src)
assert m, "dyn block not found"
src = src[:m.start()] + (
    "def _dyn_alg1(quant, scale):\n"
    "    return _quantize_alg1(dequantize_nvfp4(quant, scale))\n\n\n"
    "def hif4_dynamic_quantize_q(q_quant, q_scale, q_num_heads, head_dim, q_state):\n"
    "    return _dyn_alg1(q_quant, q_scale)\n\n\n"
    "def hif4_dynamic_quantize_k(k_quant, k_scale, kv_num_heads, head_dim, k_state):\n"
    "    return _dyn_alg1(k_quant, k_scale)\n\n\n"
    "def hif4_dynamic_quantize_v(v_quant, v_scale, kv_num_heads, head_dim, v_state):\n"
    "    return _dyn_alg1(v_quant, v_scale)\n"
)

# hmm: _dyn_table was inside the replaced block; the block also contained wrappers.
# 3) insert ALG1 helpers before the attention calibration stub
anchor = "def hif4_calibration_attention(calib_qkv_list"
src = src.replace(anchor, ALG1 + "\n\n" + anchor, 1)

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "solution.py")
open(out_path, "w", encoding="utf-8").write(src)

# import smoke test
import importlib.util
spec = importlib.util.spec_from_file_location("probe_v9lin", out_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import torch
p = mod.hif4_dynamic_quantize_q(torch.randn(4, 16, 128), torch.rand(4, 16, 8), 4, 128, None)
assert p["mant"].shape[-1] == 4
print("probe module OK")

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(out_path, "solution.py")
print("wrote", OUT_ZIP)
