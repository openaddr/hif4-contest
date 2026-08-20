"""Experiment B: GPTQ damping sweep. GPTQ_DAMP in (0.003, 0.005, 0.01, 0.02, 0.05).

_upper_cholesky_inv reads GPTQ_DAMP as a module global at call time, so
setting S.GPTQ_DAMP before calibration is enough. 0.01 is the shipped value
(= baseline). Calibration is wrapped in time.perf_counter; the pipeline is
fully deterministic (manual_seed(0) inside), so score diffs are pure damp
effects. Note damp also affects _upper_cholesky_inv calls in the mode-choice
proxy and the activation-side GPTQ — that is the intended global sweep.
"""
from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "dev")
import exp_common as E  # noqa: E402

DAMPS = (0.003, 0.005, 0.01, 0.02, 0.05)
orig = E.S.GPTQ_DAMP

std = E.std_baseline()

# baseline timing (damp=0.01 is the shipped default; run it twice, report 2nd)
torch.manual_seed(0)
_, dt_warm = E.run_pipeline()
torch.manual_seed(0)
out_base, dt_base = E.run_pipeline()
sc_base = E.score(out_base["weight_params"], out_base["activation_state"], std)
E.report("B: damp=0.01 baseline", sc_base, dt_base, base=list(sc_base))
print(f"  (warmup run time: {dt_warm:.2f}s)")

results = []
try:
    for damp in DAMPS:
        E.S.GPTQ_DAMP = damp
        torch.manual_seed(0)
        t0 = time.perf_counter()
        out = E.S.hif4_calibration_and_quantize_weight(*E.LIN["weight"], E.LIN["calib_activation_list"])
        dt = time.perf_counter() - t0
        sc = E.score(out["weight_params"], out["activation_state"], std)
        E.report(f"B: damp={damp}", sc, dt, base=list(sc_base))
        results.append((damp, sc, dt))
finally:
    E.S.GPTQ_DAMP = orig

print()
print("=== Experiment B summary (baseline mean %.4f, t=%.2fs) ===" % (sum(sc_base) / 5, dt_base))
for damp, sc, dt in results:
    print(f"damp={damp:<6} mean={sum(sc) / 5:+.4f}  diff={(sum(sc) / 5 - sum(sc_base) / 5) * 100:+.2f}pp  calib={dt:.2f}s")
