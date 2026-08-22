# caw: cancellation-aware weight quantization - NO-SHIP

solution.py untouched; patched module exec'd from source (dev/caw/study.py).
Ship-bit-identity of the off-patch verified (selftest). Seeds/scoring = decomp
conventions (study2 iter_grid, quant_alg1 baseline, joint x_play/w_play MSE).

## Algebra (task 1)
Per dynamic call (transformed space): D = v q^T - x w^T = Dq*q^T + x(q-w)^T,
Dq = v-x on the flip lattice (v4 in [-7,7], step d=0.25*unit). Continuous box:
Dq_rc in [-A_rc, A_rc], A_rc = min(k, 7 - |v4_rc|*sign(alpha_c))*d (alpha_c =
<w_c-q_c, q_c>/||q_c||^2; a |v4|=7 element is blocked only when the required
correction pushes it further from zero). Uncancelable objective
J_u(q) = min_{|Dq|<=A} ||Dq q^T + x(q-w)^T||_F^2; reachable set = box image
under q^T (zonotope, C generators in R^N).
(a) diag: Gram-diag Q^T Q: J_u^diag = sum_rc max(0,|(X(Q-W)^T Q)_rc| - A_rc
g_c)^2/g_c. Wired into GPTQ: feedback error e -> e - beta*alpha*q_i,
beta = sum_r min(|x_ri alpha|,A_ri)^2 / sum_r (x_ri alpha)^2.
(b) box-relaxed: FISTA projected gradient on the full box (120 iters).
(c) oracle: actual _refine_act_values on 128 holdout rows, fp32 Grams.

## Fidelity a/b vs c (4 groups x variants rtn/gptq/caw{0.5,1,2})
- diag rank-corr vs oracle 0.97, box 0.85, PLAIN (no projection) 1.00: the
  uncancelable projections add no ranking signal beyond plain holdout MSE.
- level: (pred-floor)/(oracle-floor) box 0.22 (credits 2-4x more
  cancellation than lattice greedy achieves), diag 4.2x (over-residual).
- k in {0.5,1,2} quanta INDISTINGUISHABLE: measured alpha ~ 1e-3..1e-2, so
  0.5 quanta already covers the whole self-aligned error except |v4|=7 rows
  (k-free saturation block). beta_mean 0.78-0.88.
- mechanistic finding: the self-aligned (q_i-direction) fraction of a
  column's rounding error is tiny (~random angle), so the prescribed
  feedback-shrink is a near no-op: caw == gptq to 3-4 digits. The observed
  50-75% weight-error cancellation (decomp2) is JOINT (zonotope over all C
  generators), which no per-column-separable surrogate sees; embedding it
  needs the final Q (chicken-egg in GPTQ's sequential loop).

## End-to-end (12 groups: C{1024,2048,4096} x spread{.5,.9} x o{0,.002},
N alternating 1024/8192; k=1; guard = post-refinement holdout, E3 rows)
guard accept 0/12 -> caw ships nothing; scores BIT-IDENTICAL to ship:
| C | T10 ship/caw | T128 | T512 | T1024 | all | dcal |
|----|--------------|------|------|-------|-----|------|
|1024| 67.7/67.7 | 67.5/67.5 | 66.2/66.2 | 61.7/61.7 | 64.9/64.9 (+0.0) | +1.2s |
|2048| 76.5/76.5 | 75.6/75.6 | 70.8/70.8 | 65.6/65.6 | 70.8/70.8 (+0.0) | +2.2s |
|4096| 81.2/81.2 | 79.0/79.0 | 74.4/74.4 | 48.8/48.8 | 66.4/66.4 (+0.0) | +5.3s |
by T: 75.1/74.0/70.5/58.7 pp, delta +0.0 at every bucket.
mini_sample linear (N=8192,C=2048): joint-conv 92.7/92.7/94.3/93.6/94.2 pp,
identical; guard reject (caw 0.3% worse). FORCED (unguarded) caw would cost
mean -21.3/-18.6/-15.8 pp per case (worst -42) at C=1024/2048/4096: the
guard is load-bearing. Root cause: on synth groups the ship plain-holdout
guard already keeps RTN over GPTQ post-refinement (fidelity: oracle rtn <
gptq in 4/4 groups; caw ~ gptq), consistent with decomp2's weight re-anchor
0/24 holdout -- weight-side calib re-optimization overfits.

## Timing (local, per group; judge ~2-4.8x)
stage 2.08s mean (duplicate caw GPTQ loop 1.99s; at N=8192 up to 4.2s) +
guard 1.06s (refinement on 22 holdout rows). Added cal: +1.2/+2.2/+5.3s by
C. Dynamic side untouched (identical states when rejected). Over the <2s
C=2048 target as built; shipping without the duplicate q_g computation
(stage-off pass-through) would halve it.

## VERDICT: NO-SHIP
- holdout accept 0/12 (+mini reject); online expected value = 0 pp over the
  250-call suite (bit-identical); forced variant ~ -20 pp/case.
- cost if shipped anyway: +2-25s cal per group online for nothing.
- the decomp2 motivation stands (cancellation is real) but the cheap
  per-column diag surrogate captures ~none of it; the box-relaxation is
  un-embeddable in GPTQ's column loop and miscalibrated 4.5x. A viable
  version needs a joint-span objective (e.g. tr(E H E^T P_perp(Q)) iterated
  over partial Q, or candidate search scored by the actual refinement --
  the oracle path IS deployable at guard cost ~1s/group) AND a group
  regime where GPTQ itself beats RTN on holdout.
Artifacts: study.py (selftest/fit/suite/suitef/mini/rep), results_{fit,
suite,mini}.json, cache/ (gitignored, no .pt in git).
