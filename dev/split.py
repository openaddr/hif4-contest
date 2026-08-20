"""Error budget split on mini linear (v9): weight-side vs activation-side."""
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


W_s = tf(w_ref)
tot = wqsw = wasw = 0.0
for pair in tests:
    x_ref = S.dequantize_nvfp4(*pair).float()
    ref = x_ref @ w_ref.T
    xs = tf(x_ref)
    p = S.hif4_dynamic_quantize_activation(pair[0], pair[1], state)
    xq = S._deq_params(p)
    tot += ((xq @ q_used.T - ref) ** 2).mean().item()
    wqsw += ((xs @ q_used.T - ref) ** 2).mean().item()   # exact acts, quant W
    wasw += ((xq @ W_s.T - ref) ** 2).mean().item()      # quant acts, exact W
n = len(tests)
print(f"total            {tot/n:.4e}  (100%)")
print(f"weight-side only {wqsw/n:.4e}  ({100*wqsw/tot:.0f}%)")
print(f"act-side only    {wasw/n:.4e}  ({100*wasw/tot:.0f}%)")
