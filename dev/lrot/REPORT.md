# Data-fitted orthogonal rotation (mode-2 prototype) vs fixed Hadamard

4 calib-fitted candidates, fit on smoothed calib[:-1] (exact orthogonal, sign/det
fixed; verified allclose(R@R.T, I)): Ra PCA(desc), Rb PCA@blkHadamard hybrid,
Rc round-robin variance-equalized PCA, Rd eigvecs of act-Hessian. FULL pipeline
(ship config) with `_rot_blocks` monkeypatched; pipeline's own holdout mode
proxy left in charge. Harness: dev/lrot/study_lrot.py; mini (C2048/N8192) +
12 synthetic groups (C{1024,2048,4096} x spread{.5,.9} x outlier{0,.002}, N
alternating 1024/8192, seeds 4200+13i, calib T=(10,128,512,1024), test
T=(10,128,512,1024,1024)). Score = (mse_alg1 - mse_play)/mse_alg1, pp/case.

## Score table (mean pp over 5 test cases per group)
| case                  |  none | hadam |   Ra |   Rb |   Rc |   Rd |
|-----------------------|------:|------:|-----:|-----:|-----:|-----:|
| mini (C2048)          | 76.76 | 86.41 | 76.76| 76.76| 76.76| 76.76|
| c1024 s.5 o0          | 81.17 | 82.20 | 81.90| 82.16| 82.21| 82.15|
| c1024 s.9 o0          | 81.37 | 82.36 | 81.96| 82.20| 81.99| 82.01|
| c1024 s.5 o.002       | 33.74 | 48.88 | 35.57| 36.05| 33.40| 35.70|
| c1024 s.9 o.002       | 32.87 | 43.47 | 28.18| 29.15| 25.71| 28.17|
| c2048 s.5 o0          | 89.79 | 90.04 | 89.80| 89.81| 89.95| 89.88|
| c2048 s.9 o0          | 89.24 | 89.61 | 89.45| 89.42| 89.50| 89.43|
| c2048 s.5 o.002       | 40.46 | 52.33 | 42.46| 42.77| 41.19| 42.34|
| c2048 s.9 o.002       | 41.04 | 50.96 | 40.46| 40.92| 39.19| 40.51|
| c4096 s.5 o0          | 80.13 | 80.85 | 80.61| 80.52| 80.63| 80.61|
| c4096 s.9 o0          | 79.44 | 80.35 | 79.98| 80.03| 80.09| 79.97|
| c4096 s.5 o.002       | 46.42 | 54.89 | 46.79| 46.97| 46.45| 46.81|
| c4096 s.9 o.002       | 47.38 | 54.02 | 48.86| 49.00| 48.45| 48.82|
| MEAN                  | 63.06 | 68.95 | 63.29| 63.52| 62.73| 63.32|

Delta vs hadamard: Ra -5.66 / Rb -5.43 / Rc -6.22 / Rd -5.63 pp/case;
wins 0/13 groups (Rc 1 tie). Clean groups (o0): tie (-0.0..-0.3pp).
Outlier groups (o0.002): -7..-15pp. On mini the pipeline's own proxy DECLINED
all R (mode=0 -> score == none; Hadamard accepted, +9.7pp over none).

## Holdout guard (task 4; gate failed, computed anyway)
R fit on calib[:-1], scored on calib[-1] vs hadamard: Ra -6.97 / Rb -6.70 /
Rc -7.52 / Rd -6.96 pp, wins 0/13. No out-of-sample survival anywhere.
NOTE: the shipped proxy only compares none-vs-rotated; it accepted R on 12/13
groups -- an unguarded mode=2 ship would have shipped the harmful R there.

## Root cause (why fitting does not transfer)
- synth draws FRESH channel gains per activation sample: calib-vs-test top
  eigenspaces are near-orthogonal (principal-angle cos 0.003-0.005 at k=64,
  C2048), and the spectrum top is nearly flat (lam1/lam64 ~ 2). The fitted
  eigenvectors are sample-specific noise; Hadamard is eigenbasis-agnostic.
- With outliers (o0.002) calib outliers define eigendirections that never
  recur; Hadamard's uniform spreading is the robust choice (R ~ none+1-2pp,
  Hadamard ~ none+10-15pp).

## Costs (would-be, had it shipped)
- Fit at calibration: eigh(C,C) 0.06/0.27/1.97 s at C=1024/2048/4096
  (measured; SVD of X 0.21/0.38/0.98 s). ~+0.1-0.9 s/group -> est +5-15 s
  online over ~20-30 linear groups.
- Dynamic per call: rotation-only delta full-R vs Hadamard (bench, median):
  C1024: -0.5..+4 ms; C2048: +0.6..+19 ms; C4096: +3.9..+79 ms (T=10..1024).
  End-to-end dynamic calls: 1.36 s (had) vs 1.36-1.43 s (R) at C4096 mean.
  Est +3-8 s online for 250 calls (judge ~2x on memory-bound shapes).
- State: current 8/32/128 MiB at C=1024/2048/4096. +R fp32 -> 12/48/192,
  +R bf16 -> 10/40/160 MiB. C=4096 BUSTS the ~128 MiB envelope even in bf16;
  only C<=2048 fits (bf16 R also loses exact orthogonality, ~2e-3 defect).

## VERDICT: NO-SHIP
Every fitted construction loses to the fixed Hadamard on every group class;
the ship gate (>0.3pp on BOTH mini and synthetic) fails at -5.4..-6.2pp.
Expected online value: ~0 with the holdout guard (correctly declines
everywhere; only the fit/eigh time is wasted), roughly -150..-400 online
(5-6pp/case x 250 linear cases, 1pp ~ 25-30 online) if shipped unguarded.
The learned-rotation vein behaves like the attention P-proxy: sample-specific
second-order statistics do not transfer, even on structured-looking data.
Direction closed at PCA/hybrid/equalize/Hessian-fit levels; only a rotation
fit on a statistic that provably persists across samples would reopen it.
