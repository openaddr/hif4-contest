## v1 - 2026-08-19 18:56:02
- artifact: dist/solution_v1.zip
- note: v1: greedy hierarchy + E6M2 scale search (8 cands), output-error-weighted; local mini_sample total +6.66/10 cases (linear ~+0.78, attn ~+0.55)

## v2 - 2026-08-19 19:25:27
- artifact: dist/solution_v2.zip
- note: v2: FIX root cause of -30091: sf candidate grid re-anchored at absmax/7 (was 2^floor(log2 amax), 2-3.5x too large, crushing small sub-blocks). + 8-cand weighted search, lv2/lv3 MSE refinement, AWQ smoothing (linear), q/k paired smoothing, V positional weights. Local vs norm7 baseline: linear +81%, attn +34%/case; self_check 22/22; est 92s local full-run

## v3 - 2026-08-19 19:31:26
- artifact: dist/solution_v3.zip
- note: v3: attention recalibrated - gradient-sensitivity per-element weight tables (Q/K/V) with gamma search + auto fallback to uniform (gamma=0 chosen on mini_sample, self-guarding); keeps amax/7 anchored 8-code search + lv refinement + q/k smoothing. Local vs norm7: linear +79%, attn +38%/case (v2 was +34). Ref-solution calibration: online ~= 0.71x local => est +29000

## v4 - 2026-08-19 19:50:46
- artifact: dist/solution_v4.zip
- note: v4: Q/K per-head Hadamard rotation (exact attention invariant, per-group on/off decided on calib, kills outlier structure); linear weight-guidance clamped to [0.25,4]x mean; dropped beta smoothing + gradient-gamma machinery (overfit culprits per synthetic ablation); fixed local eval baseline bug (baseline activations now quantized). Synthetic: lin +22.7%/worst +19.0, attn +30.1%/worst +19.8 (v3: 18.3/-1.9, 15.8/9.0). Mini: +6.31/10. Runtime local ~107s

