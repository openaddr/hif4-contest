"""v14 candidates: (B) weight clipping before quantization, (C) Hessian
shrinkage toward diagonal for better calib->test transfer. Mini evaluation
with the true output metric (tests against x @ W_ref^T).
"""
import sys, os, importlib.util
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "example", "solution"))
sys.path.insert(0, ROOT)


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load_mod(os.path.join(ROOT, "..", "example", "solution", "solution.py"), "sol")
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
N, C = W_s.shape

# ---------- B: weight clipping (fit on calib[:-1], evaluate on tests) ----------
acts_s = [tf(S.dequantize_nvfp4(*a).float()) for a in calib]
Hs = torch.zeros(C, C)
for a in acts_s[:-1]:
    Hs += a.T @ a
Uw = S._upper_cholesky_inv(Hs)


def eval_q(q_w):
    tot = 0.0
    for pair in tests:
        x_ref = S.dequantize_nvfp4(*pair).float()
        ref = x_ref @ w_ref.T
        p = S.hif4_dynamic_quantize_activation(pair[0], pair[1], state)
        xq = S._deq_params(p)
        tot += ((xq @ q_w.T - ref) ** 2).mean().item()
    return tot / len(tests)


base = eval_q(q_used)
print(f"baseline (current pipeline)        {base:.4e}")

unit_w = S._params_unit_flat(wp)
for c in (4.0, 3.0, 2.5, 2.0):
    sig = W_s.std(dim=1, keepdim=True)
    Wc = W_s.clamp(-c * sig, c * sig)
    qc = S._gptq_quantize_values(Wc, unit_w, Uw)
    m = eval_q(qc)
    print(f"clip c={c:.1f} (global sigma)      {m:.4e}  ({100*(1-m/base):+.1f}%)")

# ---------- C: Hessian shrinkage for act-GPTQ ----------
Ha = q_used.T @ q_used
Ua = S._upper_cholesky_inv(Ha)
order = torch.argsort(Ha.diagonal(), descending=True)
for delta in (0.25, 0.5):
    diag = Ha.diagonal()
    Hs_shr = (1 - delta) * Ha + delta * torch.diag(diag)
    Ua_s = S._upper_cholesky_inv(Hs_shr)
    order_s = torch.argsort(Hs_shr.diagonal(), descending=True)
    tot = 0.0
    for pair in tests:
        x_ref = S.dequantize_nvfp4(*pair).float()
        ref = x_ref @ w_ref.T
        xs = tf(x_ref)
        p_r = S._quantize_weighted(xs, torch.ones(1, C))
        unit_x = S._params_unit_flat(p_r)
        xo = xs[:, order_s]
        q = S._gptq_quantize_values(xo, unit_x[:, order_s], Ua_s)
        q0 = torch.empty_like(q)
        q0[:, order_s] = q
        tot += ((q0 @ q_used.T - ref) ** 2).mean().item()
    print(f"act-GPTQ shrinkage delta={delta:.2f}    {tot/len(tests):.4e}  ({100*(1-tot/len(tests)/base):+.1f}%)")
