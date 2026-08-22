# Decomposition: where the remaining linear error lives (v21)

32 synthetic groups (dev/synth.py): C{512,1024,2048,4096} x N{1024,8192} x
spread{0.5,0.9} x outlier{0,0.002}; calib T=(10,128,512,1024), test
T=(10,128,512,1024,1024), shared channel gains. Score mirrors dev/diag3.py
(exact paper Alg.1 baseline). refined = lattice forced at all C (hash-even
branch); unrefined = v14 path. beta=0 twin parity: bit-identical, 32/32.

## Table i - score (pp) by test-T bucket
| T    | refined | unrefined |
|------|---------|-----------|
| 10   | 59.63   | 41.85     |
| 128  | 58.95   | 41.15     |
| 512  | 58.33   | 41.04     |
| 1024 | 52.07   | 40.76     |
T=10 is NOT worse - it is the BEST bucket; T=1024 is worst (-6..7pp).
Unrefined is flat in T => the dip is the lattice stage: sweep depth drops
5->2 at T=1024 (_refine_act_values). Probe (probe_sweeps.py, 8 groups):
5 sweeps at T=1024 recovers +1.1..+10.4pp, mean +6.2pp, cost +0.5-3s/call.

## Table ii - score (pp) by (C, refined?) [T10/T128/T512/T1024 | all]
| C    | refined                     | unrefined | delta |
|------|-----------------------------|-----------|-------|
| 512  | 47.8/47.8/47.8/42.6 | 45.7 | 31.8      | +13.9 |
| 1024 | 59.1/59.3/58.7/51.8 | 56.1 | 41.1      | +15.1 |
| 2048 | 64.4/62.7/61.2/55.5 | 59.9 | 44.1      | +15.7 |
| 4096 | 67.2/66.0/65.6/58.4 | 63.2 | 47.5      | +15.7 |
Residual error is largest at small C (C=512: player MSE still 54% of std);
lattice refinement is worth ~14-16pp at EVERY C. Attribution (mse_act/mse_w):
0.76 @C512 -> ~1.0 @C>=2048 - act and weight error are equal co-contributors,
weight side dominates at small C. Ship hash parity: 1/8 C=4096 groups even.

## T=10 anchor stats (max(test_absmax)/calib_block_max, transformed space)
mean 0.30-0.34, p50 0.27-0.32, p90 0.51-0.62; 80-89% of blocks < 0.5; the
prior moves 98-100% of anchors at beta>=0.5. The 10-token block max sits ~3x
below the calib max - harmless today: the quantizer anchors per row, every
row-block max is representable at its own anchor, no clipping to trade.

## Beta sweep (score delta pp vs refined control; 32 groups)
| beta | dT10   | dT128  | dT512  | dT1024 | worst case |
|------|--------|--------|--------|--------|------------|
| 0.5  | -88.9  | -30.4  | -18.7  | -16.3  | -1236      |
| 0.7  | -120.5 | -46.2  | -28.0  | -23.4  | -1517      |
| 1.0  | -203.1 | -85.5  | -54.2  | -47.2  | -2227      |
per-C dT10 @beta=.5: C512 -335, C1024 -17, C2048 -3.4, C4096 +0.25 (noise).
Uniformly negative: per-token scale (randn(T,1) gains) makes the pooled calib
block max ~3x a typical row's absmax; the raised anchor multiplies sf (grid
step) for nearly every row-block with no offsetting gain; C=512 hit hardest.

## VERDICT: NO-SHIP the calib block-absmax sf-prior
Online value (250 cases = 50 grp x [1xT10,1xT128,1xT512,2xT1024]): beta .5
-> -8.5k pp, .7 -> -12.1k pp, 1.0 -> -21.9k pp. No beta in {0.5,0.7,1.0} is
shippable; the small-T-anchor premise is empirically false.

## Actionable instead (measured here)
1. Small-C weight side (act/w=0.76 at C=512): weight error dominates there
   (but see E3 verdict below - unreachable via sweep depth on synthetic).
2. Refined-at-4096 is +15.7pp for the 7/8 hash-odd groups (state/WA risk).

# Sweep curves (sweep2.py; results_C/D.json; baseline = v22 ship s5/r20)
Method: string-patched solution copies (probe_sweeps pattern), cached cal
states (v21->v22 diff touched only the dynamic n_sweeps line; parity
0.00e+00 vs results_A at T<=512), 8 groups/C, judge = local/4.8, 0.28x
transfer, 250 refined online calls (50/50/50 x T10/128/512 + 100 x T1024).

## Act sweep curve: score pp (mean over 8 groups x T buckets; gains are
T-UNIFORM, so per-T rows collapse - e.g. C2048 d(s5->s12) = +6.5/+6.7/+6.4/
+6.4 pp for T=10/128/512/1024; no per-T tuning is warranted)
| C  | s5   | s8   | s12  | pp/sweep 5-8 | 8-12 |
|----|------|------|------|--------------|------|
| 512  | 47.4 | 49.0 | 49.9 | +0.53 | +0.24 |
| 1024 | 58.4 | 61.3 | 63.2 | +0.95 | +0.47 |
| 2048 | 62.4 | 66.0 | 68.9 | +1.20 | +0.72 |
Concave but NOT flat at 12: the marginal halves from the 5-8 to the 8-12
step yet stays +0.2pp/sweep (C512) to +0.7pp (C2048); larger C pays most.

## Rounds vs sweeps: SAME loop body -> only sweeps*rounds matters.
s10 vs s5r40 bit-identical (mse rel 0.00e+00, all 5 cases of a C512 group);
s5r40 sits between s8 (160 iters) and s12 (240) on the curve as predicted.
Raise n_sweeps, NOT REFINE_ROUNDS: rounds also multiplies E3 cal cost.

## Timing (local s per dynamic call, C=2048 = worst refined C)
| T    | s5  | s8  | s12 |
|------|-----|-----|-----|
| 512  | 0.57 | 0.77 | 1.06 |
| 1024 | 1.18 | 1.63 | 2.24 |
Online price vs ship s5: s8 +7s blended / +12s worst-case all-C2048;
s12 +17s blended / +29s worst-case. Both fit the ~50s margin (248s ship).

## VERDICT: n_sweeps 5 -> 12 (uniform, all T<=1024), REFINE_ROUNDS=20 kept.
Expected ONLINE points at 0.28x transfer: s12 +321 (uniform C mix) to +455
(all-C2048); s8 fallback +187..+251 at +7..12s. Artifact 248 -> ~265-277s.

## E3 weight sweeps at small C: DEAD AXIS on synthetic.
w1/w2/w3 bit-identical scores (0.00pp) at C in {512,1024}: the hold-out
guard reverts E3 in 25/25 synthetic groups (16 small-C + 8 C2048 + the one
hash-even C4096; hold1/hold0 = 1.06-1.31, refined consistently WORSE on
hold-out). Extra sweeps only burn cal time: C512 +0.33s, C1024 +0.7s local
(w1->w3 mean). Keep REFINE_W_SWEEPS=1 - it does fire on the real mini
sample (v20 diag3 +7.25), so do not zero it on synthetic evidence alone.

