"""How predictable is the activation quantization error E = Xh - X?

Pool sizing for the W_eff idea:
  1) per-channel diagonal slope R^2 (C params, well-posed)
  2) top-k PCA of Xh rows predicting E (cross-channel linear structure)
  3) clamp statistics (systematic attenuation)
Fit on calib[:-1], report R^2 on held-out calib[-1] (honest).
"""
import sys, os
import importlib.util

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "example", "solution"))


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load_mod(os.path.join(ROOT, "..", "example", "solution", "solution.py"), "sol")
sys.path.insert(0, ROOT)
import hif4  # noqa: E402

torch.manual_seed(0)
lin = torch.load(os.path.join(ROOT, "..", "example", "mini_sample", "linear.pt"),
                 weights_only=True, map_location="cpu")[0]
wq, ws = lin["weight"]
calib = lin["calib_activation_list"]
w_ref = hif4.dequantize_nvfp4(wq, ws).float()

out = S.hif4_calibration_and_quantize_weight(wq, ws, calib)
state = out["activation_state"]
mode, s = state["mode"], state["s"]


def tf(x):
    x = x * s
    if mode == 1:
        return S._rot_blocks(x)
    return x


X = [tf(S.dequantize_nvfp4(*a).float()) for a in calib]
Xh = []
for a in calib:
    p = S.hif4_dynamic_quantize_activation(a[0], a[1], state)
    Xh.append(S._deq_params(p).contiguous())

Xf = torch.cat(X[:-1])
Ef = torch.cat([xh - x for xh, x in zip(Xh[:-1], X[:-1])])
Xv = X[-1]
Ev = Xh[-1] - X[-1]

# --- 3) clamp stats (recompute units per block on the held-out) ---
p_hold = S.hif4_dynamic_quantize_activation(calib[-1][0], calib[-1][1], state)
unit_hold = S._params_unit_flat(p_hold).abs()
clamped = (Xv.abs() > 1.75 * unit_hold + 1e-12)
print(f"clamped elements: {100*clamped.float().mean().item():.2f}%")
e2 = (Ev ** 2)
print(f"MSE share from clamped elems: {100*(e2*clamped).sum().item()/e2.sum().item():.1f}%")

# --- 1) diagonal per-channel slope ---
num = (Xf * Ef).sum(dim=0)
den = (Xf * Xf).sum(dim=0).clamp_min(1e-20)
alpha = num / den
res = Ev - alpha * Xv
r2_diag = 1 - (res ** 2).sum().item() / (Ev ** 2).sum().item()
print(f"per-channel diagonal R^2 on hold-out: {100*r2_diag:+.2f}%")

# --- 2) cross-channel: predict E from top-k PCA directions of Xf ---
Xc = Xf - Xf.mean(0, keepdim=True)
U, Sv, Vt = torch.linalg.svd(Xc, full_matrices=False)
V = Vt.T  # (C, k)
proj = Xv @ V  # hold-out features
for k in (16, 64, 256):
    F = proj[:, :k]
    # ridge fit Ef = F @ B, small ridge
    Ftr = (Xf - Xf.mean(0, keepdim=True)) @ V[:, :k]
    A = Ftr.T @ Ftr + torch.eye(k) * (Ftr ** 2).mean() * 0.1
    B = torch.linalg.solve(A, Ftr.T @ Ef)
    pred = F @ B
    r2 = 1 - ((Ev - pred) ** 2).sum().item() / (Ev ** 2).sum().item()
    print(f"PCA-{k:3d} cross-channel R^2 on hold-out: {100*r2:+.2f}%")
