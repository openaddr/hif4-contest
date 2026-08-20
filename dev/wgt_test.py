"""Does W-column-norm weighting of the activation search reduce output MSE?

Plain (no act-GPTQ) comparison on mini linear, v9 solution:
  A) activation search with uniform weights   (current)
  B) activation search weighted by clamp(wcol^2/mean, 0.25, 4)
Also the weight-side symmetric test:
  C) weight search weighted by clamp(act_col_energy/mean, 0.25, 4)
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
calib, tests = lin["calib_activation_list"], lin["test_activation_list"]
w_ref = hif4.dequantize_nvfp4(wq, ws).float()

out = S.hif4_calibration_and_quantize_weight(wq, ws, calib)
wp, state = out["weight_params"], out["activation_state"]
mode, s = state["mode"], state["s"]
q_used = S._deq_params(wp).contiguous()


def tf(x):
    x = x * s
    if mode == 1:
        return S._rot_blocks(x)
    return x


# ---- variant B: W-column-norm weighted activation search ----
wcol = (w_ref ** 2).sum(dim=0)  # importance of each input channel
wgt = (wcol / wcol.mean()).clamp(0.25, 4.0)


def eval_act(q_w, weighted):
    tot = 0.0
    for pair in tests:
        x_ref = S.dequantize_nvfp4(*pair).float()
        ref = x_ref @ w_ref.T
        x = tf(x_ref)
        p = S._quantize_weighted(x, wgt.unsqueeze(0) if weighted else torch.ones(1, x.shape[1]))
        xq = S._deq_params(p)
        tot += ((xq @ q_w.T - ref) ** 2).mean().item()
    return tot / len(tests)


mse_a = eval_act(q_used, False)
mse_b = eval_act(q_used, True)
print(f"A uniform act search : {mse_a:.4e}")
print(f"B wcol-weighted      : {mse_b:.4e}  ({100*(1-mse_b/mse_a):+.1f}%)")

# ---- variant C: act-energy weighted WEIGHT search (affects units pre-GPTQ) ----
energy = torch.zeros(w_ref.shape[1])
n = 0
for a in calib:
    ad = S.dequantize_nvfp4(*a).float() * s
    energy += (ad * ad).sum(dim=0)
    n += ad.shape[0]
wgt_w = ((energy / n) / (energy / n).mean()).clamp(0.25, 4.0)
W_s = tf(w_ref)
p_w = S._quantize_weighted(W_s, wgt_w.unsqueeze(0))
q_w2 = S._deq_params(p_w).contiguous()
mse_c = eval_act(q_w2, False)
print(f"C energy-weighted W  : {mse_c:.4e}  ({100*(1-mse_c/mse_a):+.1f}%)  [A uses uniform-W search]")
# combined
mse_bc = eval_act(q_w2, True)
print(f"B+C combined         : {mse_bc:.4e}  ({100*(1-mse_bc/mse_a):+.1f}%)")
