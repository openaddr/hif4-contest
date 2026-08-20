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

## v5 - 2026-08-19 21:38:10
- artifact: dist/solution_v5.zip
- note: v5: GPTQ error compensation, hold-out guarded (H from calib[:-1], eval on calib[-1]): weight side 0.086x RTN on real mini data (11x); activation side via quantized-weight Gram; V via attention-prob Gram per length. Dropped channel-energy weighting (hurt output MSE 1.46x). Exact alg1 baseline confirmed by probe(=0). mini: lin +90-97%, attn +29-42%; synth worst-case: lin +28%/worst +19.6 (guards reject GPTQ there). Est judge time ~170s/300s

## v6 - 2026-08-19 22:25:03
- artifact: dist/solution_v6.zip
- note: v6: + block-diagonal 64-dim random-Hadamard rotation for linear (exact matmul invariant, Gaussianizes intra-block outliers; on/off chosen per group by GPTQ-level proxy on weight subsample — RTN-level proxy proved too noisy). Rotation+GPTQ synergistic on mini (weight-side 0.22x alg1) and synth. Fixed diag3 eval bug (player activations were unquantized). mini: lin +81-85%/case, attn +29-42%; synth: lin +42%/worst +35, attn +9%. Judge time est 165-218s/300s

## v7 - 2026-08-19 22:41:08
- artifact: dist/solution_v7.zip
- note: v7: + Q/K logit-space GPTQ (per-head 256x256 Hessians Q^TQ/K^TK from calib, hold-out guarded; targets softmax(Q dK^T + dQ K^T) linearization). mini attn +41%->+55%/case (t4 3.42e-4->2.64e-4); synth unchanged (guards reject on iid). v6 online was 19229 @221.6s; est judge ~233s/300s

## v8 - 2026-08-19 23:22:05
- artifact: dist/solution_v8.zip
- note: v8: linear transform now 3-way guarded {rotation | channel permutation | none} via GPTQ-level subsample proxy (permutation = correlation-preserving outlier isolation, wins on flat/hard regimes where rotation suboptimal); activation GPTQ act-ordered. mini lin +81-85%, attn +41-55% (attn unchanged); synth lin mean 41.4%/worst 27.1 (worst improved from 34.8). Decomp probe: online lin +67.5%, attn +11.9% -> v8 targets linear. Judge est ~245s/300s

## v9 - 2026-08-20 00:09:36
- artifact: dist/solution_v9.zip

- note: v9: sf candidates now ranked by EXACT refined error (6-cand R6 grid x jointly-optimal lv tree) instead of greedy-lv ranking — pure-Gaussian element-wise gain over alg1 6.6%->11.9% (near the 12.1% grid optimum); greedy ranking provably overshoots sf. Dropped v8's permutation proxy (online ~0 gain, refunds ~12s). mini +6.54->+6.65; synth lin mean 41.7->45.0/worst 27.1->37.9, attn mean 9.2->14.5/worst -20.8->-4.9. Timing A/B: +6.5s online est -> ~255s/300s. v8 online was 19883 @265.9s
## v9 - 2026-08-20 09:25:26
- artifact: dist/solution_v9.zip

- note (v9 rev2, pre-upload): fixed in-sample guard leak in attention calibration. V-GPTQ guard evaluated on `big` (largest calib sample) while its P-Gram was built INCLUDING that sample -> accepted overfit GPTQ-V that hurt test output MSE by 2.8-17.3% on judge-like random data. All attention guards (rotation, V-GPTQ, Q/K-GPTQ) now evaluate on calib[-1] (true hold-out); Grams/Hessians from calib[:-1] only. Synth attn mean 14.5->21.9, worst -4.9->+14.0. mini +6.65->+6.71; attn dyn 0.91->0.79s/call (V-GPTQ skipped). zip rebuilt before first upload
- note: 20169.294 @ 289.143s online was the FIRST v9 build (refined ranking, no guard fix). The guard-fixed rev2 zip (current dist/solution_v9.zip, verified) is still untested online; it isolates the V-GPTQ hold-out guard fix vs the 20169 baseline. Time risk: first v9 already at 289/300 -> v10 must cut >=30s regardless of score outcome
