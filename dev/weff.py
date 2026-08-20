"""Size the W_eff (LS-corrected weight target) opportunity on mini linear data.

Current weight GPTQ minimizes ||Xh (What - W)||^2; the true objective is
||Xh What^T - X W^T||^2. The LS-optimal effective weight is
    W_eff = W_s @ (X^T Xh) @ (Xh^T Xh)^-1   (transformed space, damped)
We measure: output MSE on the 5 TEST activations, current vs W_eff-corrected,
with the correction fitted only on calib[:-1] (hold-out discipline).
"""
import sys, os, time
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
tests = lin["test_activation_list"]
w_ref = hif4.dequantize_nvfp4(wq, ws).float()

t0 = time.time()
out = S.hif4_calibration_and_quantize_weight(wq, ws, calib)
print(f"calib {time.time()-t0:.1f}s  mode={out['activation_state']['mode']}")
wp = out["weight_params"]
state = out["activation_state"]
q_used = S._deq_params(wp).contiguous()

# --- reproduce transformed space ---
mode = state["mode"]
s = state["s"]
def tf(x):
    x = x * s
    if mode == 1:
        return S._rot_blocks(x)
    return x

# original (unquantized) calib activations in transformed space
acts = [S.dequantize_nvfp4(*a).float() for a in calib]
Xs = [tf(a) for a in acts]
W_s = tf(w_ref)

# quantized calib activations through the REAL dynamic pipeline
Xh = [S.deq_params_p if False else None]  # placeholder
Xh = []
for a in calib:
    p = S.hif4_dynamic_quantize_activation(a[0], a[1], state)
    Xh.append(S._deq_params(p).contiguous())

# clamp statistics: how systematic is the activation error?
E = torch.cat([xh - x for xh, x in zip(Xh, Xs)])
Xc = torch.cat(Xs)
print(f"act err: rel MSE {100*(E*E).mean().item()/(Xc*Xc).mean().item():.3f}%  "
      f"corr(E, Xh) global: {torch.corrcoef(torch.stack([E.flatten(), Xc.flatten()]))[0,1].item():.4f}")

# --- fit on calib[:-1], H = Xh^T Xh, M = X^T Xh ---
Hh = torch.zeros_like(W_s[:, :1].T @ W_s[:, :1]).squeeze(0)  # CxC
C = W_s.shape[1]
Hh = torch.zeros(C, C)
M = torch.zeros(C, C)
for x, xh in zip(Xs[:-1], Xh[:-1]):
    Hh += xh.T @ xh
    M += x.T @ xh
damp = 0.01 * Hh.diagonal().mean()
Hh_d = Hh + torch.eye(C) * damp
try:
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Hh_d))
except Exception as e:
    print("cholesky fail", e)
    raise SystemExit
W_eff = (W_s @ M.T) @ Hinv

# quantify correction size
print(f"||dW||/||W|| = {(W_eff - W_s).norm().item() / W_s.norm().item():.4f}")

# --- evaluate on the 5 TEST activations (true end metric) ---
def eval_weights(q_w):
    tot = 0.0
    for pair in tests:
        x_ref = S.dequantize_nvfp4(*pair).float()
        ref = x_ref @ w_ref.T
        p = S.hif4_dynamic_quantize_activation(pair[0], pair[1], state)
        xq = S._deq_params(p)
        mse = ((xq @ q_w.T - ref) ** 2).mean().item()
        tot += mse
    return tot / len(tests)

mse_cur = eval_weights(q_used)
print(f"\ncurrent    output MSE {mse_cur:.4e}")

# quantize W_eff with the same units + GPTQ (Uw rebuilt from Hh)
unit = S._params_unit_flat(wp)
U = S._upper_cholesky_inv(Hh / Hh.diagonal().mean())  # normalized-ish
for gamma in (1.0, 0.5):
    Wg = W_s + gamma * (W_eff - W_s)
    q_eff = S._gptq_quantize_values(Wg, unit, U)
    mse_eff = eval_weights(q_eff)
    print(f"W_eff g={gamma:.1f} output MSE {mse_eff:.4e}  "
          f"({100*(1-mse_eff/mse_cur):+.1f}% vs current)")
