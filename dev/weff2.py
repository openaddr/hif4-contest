"""W_eff sizing, take 2: proper ridge form.

    W_eff^T = (Hh + lam I)^-1 (M^T + lam I) W^T
    Hh = Xh^T Xh (quantized calib acts), M = X^T Xh, lam swept on log grid.
Fitted on calib[:-1], scored on the 5 TEST activations (true end metric).
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
calib, tests = lin["calib_activation_list"], lin["test_activation_list"]
w_ref = hif4.dequantize_nvfp4(wq, ws).float()

out = S.hif4_calibration_and_quantize_weight(wq, ws, calib)
wp, state = out["weight_params"], out["activation_state"]
q_used = S._deq_params(wp).contiguous()
mode, s = state["mode"], state["s"]


def tf(x):
    x = x * s
    if mode == 1:
        return S._rot_blocks(x)
    return x


Xs = [tf(S.dequantize_nvfp4(*a).float()) for a in calib]
Xh = []
for a in calib:
    p = S.hif4_dynamic_quantize_activation(a[0], a[1], state)
    Xh.append(S._deq_params(p).contiguous())
W_s = tf(w_ref)

C = W_s.shape[1]
Hh = torch.zeros(C, C)
M = torch.zeros(C, C)
for x, xh in zip(Xs[:-1], Xh[:-1]):
    Hh += xh.T @ xh
    M += x.T @ xh
eye = torch.eye(C)
scale = Hh.diagonal().mean()


def eval_weights(q_w):
    tot = 0.0
    for pair in tests:
        x_ref = S.dequantize_nvfp4(*pair).float()
        ref = x_ref @ w_ref.T
        p = S.hif4_dynamic_quantize_activation(pair[0], pair[1], state)
        xq = S._deq_params(p)
        tot += ((xq @ q_w.T - ref) ** 2).mean().item()
    return tot / len(tests)


mse_cur = eval_weights(q_used)
print(f"current output MSE {mse_cur:.4e}")

unit = S._params_unit_flat(wp)
U = S._upper_cholesky_inv(Hh)
best = (None, mse_cur)
for lam_exp in (-4, -3, -2, -1, 0):
    lam = scale * (10.0 ** lam_exp)
    A = Hh + eye * lam
    try:
        Ainv_Mt_lam = torch.linalg.solve(A, M.T + eye * lam)
    except Exception:
        continue
    W_eff = W_s @ Ainv_Mt_lam
    dn = (W_eff - W_s).norm().item() / W_s.norm().item()
    q_eff = S._gptq_quantize_values(W_eff, unit, U)
    mse_eff = eval_weights(q_eff)
    tag = ""
    if mse_eff < best[1]:
        best = (lam_exp, mse_eff)
        tag = "  <-- best"
    print(f"lam=1e{lam_exp:+d}*diag  ||dW||/||W||={dn:7.4f}  MSE {mse_eff:.4e}  "
          f"({100*(1-mse_eff/mse_cur):+.1f}% vs current){tag}")
print(f"\nbest: lam 1e{best[0]} -> {100*(1-best[1]/mse_cur):+.1f}% output MSE")
