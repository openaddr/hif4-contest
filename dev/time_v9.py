"""v9 timing on mini_sample (single linear + single attn group; v8 local
reference: linear cal 4.86s/dyn 1.75s, attn cal 2.72s/dyn 4.53s)."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "example", "solution"))
import torch
import solution as S

ROOT = os.path.join(os.path.dirname(__file__), "..", "example", "mini_sample")
torch.manual_seed(0)

lin = torch.load(os.path.join(ROOT, "linear.pt"), weights_only=True, map_location="cpu")[0]
t0 = time.time()
cal = S.hif4_calibration_and_quantize_weight(*lin["weight"], lin["calib_activation_list"])
t1 = time.time()
for pair in lin["test_activation_list"]:
    S.hif4_dynamic_quantize_activation(pair[0], pair[1], cal["activation_state"])
t2 = time.time()
print(f"linear: cal {t1-t0:.2f}s  dyn/call {(t2-t1)/len(lin['test_activation_list']):.2f}s "
      f"({len(lin['test_activation_list'])} calls)")

at = torch.load(os.path.join(ROOT, "attn.pt"), weights_only=True, map_location="cpu")[0]
keys = list(at.keys())
print("attn keys:", keys[:8])
qh = at["q_num_heads"]
kvh = at["kv_num_heads"]
dh = at["head_dim"]
samples = at["calib"]
tests = at["test"]
t0 = time.time()
st = S.hif4_calibration_attention(samples, qh, kvh, dh)
t1 = time.time()
n = 0
for smp in tests:
    S.hif4_dynamic_quantize_q(*smp["q"], qh, dh, st["q_state"])
    S.hif4_dynamic_quantize_k(*smp["k"], kvh, dh, st["k_state"])
    S.hif4_dynamic_quantize_v(*smp["v"], kvh, dh, st["v_state"])
    n += 1
t2 = time.time()
print(f"attn:   cal {t1-t0:.2f}s  dyn/call {(t2-t1)/max(n,1):.2f}s ({n} calls)")
