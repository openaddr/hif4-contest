"""Experiment A: extend ALPHA_GRID with a stronger smoothing point 0.75.

Monkeypatch S.ALPHA_GRID = (0.0, 0.15, 0.3, 0.5, 0.75) at runtime (the alpha
loop reads the module global at call time), run the mini linear pipeline,
score the 5 tests vs the quant_alg1 baseline, and diff against the current
baseline scores. Also report which alpha the search picked.
"""
from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "dev")
import exp_common as E  # noqa: E402

NEW_GRID = (0.0, 0.15, 0.3, 0.5, 0.75)
OLD_GRID = (0.0, 0.15, 0.3, 0.5)

std = E.std_baseline()

# --- reference run (old grid) with timing ---
torch.manual_seed(0)
S_old = E.S.ALPHA_GRID
E.S.ALPHA_GRID = OLD_GRID
out_old, dt_old = E.run_pipeline()
sc_old = E.score(out_old["weight_params"], out_old["activation_state"], std)

# --- extended grid run ---
torch.manual_seed(0)
E.S.ALPHA_GRID = NEW_GRID
t0 = time.perf_counter()
out_new = E.S.hif4_calibration_and_quantize_weight(*E.LIN["weight"], E.LIN["calib_activation_list"])
dt_new = time.perf_counter() - t0
sc_new = E.score(out_new["weight_params"], out_new["activation_state"], std)
E.S.ALPHA_GRID = S_old  # restore

# --- which alpha was picked? recompute logm and match against s ---
acts_raw = [E.S.dequantize_nvfp4(aq, as_).float() for aq, as_ in E.LIN["calib_activation_list"]]
abs_sum = sum(a.abs().sum(dim=0) for a in acts_raw)
n_tok = sum(a.shape[0] for a in acts_raw)
m = (abs_sum / max(n_tok, 1)).clamp_min(1e-12)
logm = m.log()
logm = logm - logm.mean()
s_new = out_new["activation_state"]["s"]
for alpha in NEW_GRID:
    if torch.allclose(torch.exp(logm * alpha), s_new, atol=1e-6):
        print(f"picked alpha (new grid): {alpha}")
        break
else:
    print("picked alpha (new grid): <no match — likely 0.75? check manually>")

print()
E.report("A: old grid (baseline reproduce)", sc_old, dt_old, base=list(sc_old))
E.report("A: grid + 0.75", sc_new, dt_new, base=list(sc_old))
