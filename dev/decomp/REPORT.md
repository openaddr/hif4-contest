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
row-block max is representable at its own anchor, so there is no clipping
benefit to trade against grid coarseness.

## Beta sweep (score delta pp vs refined control; 32 groups)
| beta | dT10   | dT128  | dT512  | dT1024 | worst case |
|------|--------|--------|--------|--------|------------|
| 0.5  | -88.9  | -30.4  | -18.7  | -16.3  | -1236      |
| 0.7  | -120.5 | -46.2  | -28.0  | -23.4  | -1517      |
| 1.0  | -203.1 | -85.5  | -54.2  | -47.2  | -2227      |
per-C dT10 @beta=.5: C512 -335, C1024 -17, C2048 -3.4, C4096 +0.25 (noise).
Uniformly negative: per-token scale (randn(T,1) gains) makes the pooled
calib block max ~3x a typical row's absmax; the raised anchor multiplies sf
(grid step) for nearly every row-block with no offsetting gain. C=512 is hit
hardest (fewest channels, smallest weight-err share).

## VERDICT: NO-SHIP the calib block-absmax sf-prior
Online value (250 cases = 50 grp x [1xT10,1xT128,1xT512,2xT1024]): beta .5
-> -8.5k pp, .7 -> -12.1k pp, 1.0 -> -21.9k pp. No beta in {0.5,0.7,1.0} is
shippable; the small-T-anchor premise is empirically false.

## Actionable instead (measured here)
1. T=1024 sweep cap 2->5: +6.2pp/case mean => ~+620pp online (100 calls),
   cost +50-300s online => needs a time budget (3-4 sweeps compromise).
2. Small-C weight side (act/w=0.76 at C=512): weight error dominates there.
3. Refined-at-4096 is +15.7pp for the 7/8 hash-odd groups (state/WA risk).
