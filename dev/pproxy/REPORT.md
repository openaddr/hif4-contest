# Calibration-fitted attention-P proxy for V refinement — VERDICT: NO-SHIP

Design: carry per-R time-Grams G_R[hv] = sum_{h in group} P_cal,h^T P_cal,h
(P = softmax(q_cal k_cal^T / sqrt(dh)) from the calibration sample with R rows;
bf16 in v_state, keyed by the incoming v row count) and refine v values
against the Gram image of the residual with the solution's flip machinery.
Variants: single (1 calib sample/R), meanP = E[P]^T E[P], EptP = E[P^T P];
fit on calib[:-1] (hold-out discipline). Regimes: same (test==calib sample,
upper bound), diff (q/k = 0.8*calib + 0.6*fresh, P-cos 0.45-0.70 ~= mini judge
0.27-0.76), shift (fresh draws, spread*2.25). R-match verified: mini
calib_R = test_R = {10,128,512,1024(2x)}; synA/B/C (make_attn_group) all True.

## 3x3 table (dscore pp vs solution baseline, sw6 / sw2; mean over 5 calls/group)
| proxy | same | diff (realistic) | shift | mini judge (real data) | oracle sw6 |
|---|---|---|---|---|---|
| single | -17.1 (-7.5) | -33.0 (-11.6) | -26.3 (-10.0) | -2.2 (-0.9) | same +1.4 / diff +2.2 / shift +0.4 / mini +7.8 |
| meanP  |  -4.7 (-1.1) | -12.6 (-4.8)  | -25.4 (-10.0) | -2.2 (-0.9) | |
| EptP   |  -4.4 (-0.9) | -10.0 (-3.8)  | -22.2 (-9.4)  | -2.2 (-0.9) | |
Every cell negative, every regime, every sweep depth. Mini (the only real
judge sample): -1.2/-1.7/-2.3/-3.7/-2.1 pp at sw6 per case vs oracle
+6.4..+9.4 (oracle mean +7.80 == dev/attn_refine, machinery consistent).

## Why it fails (measured)
1. Machinery is correct: where the Gram is EXACT (test==fitted calib sample,
   gcos=1.00) the proxy equals the oracle (+4.97 vs +5.04, +0.29 vs +0.27,
   +0.74 vs +0.79, +0.44 vs +0.41 pp).
2. The time-Gram is SAMPLE-SPECIFIC, not a distribution statistic: two
   same-distribution samples give near-orthogonal Grams (gcos 0.01-0.16 for
   the non-fitted twin / hold-out cases at R>=128). meanP/EptP average
   quasi-orthogonal Grams -> mostly the shared diagonal-mass skeleton; that
   is still not the test call's Gram (mini gcos 0.49-0.87).
3. Mis-specified quadratic form actively hurts: proxy removes 20.8-25.2% of
   ITS OWN objective while RAISING the true exact-P error J_P by +7.5%
   (T=512) / +12.5% (T=1024) on mini; oracle removes 23-28% (jp_check.py).
   Greedy top-1 keeps flipping while any proxy gain exists, so mis-flips
   accumulate ~ with the misaligned Gram energy. Same pathology as flat-P,
   only smaller magnitude. No online guard can detect this (measuring true
   J_P needs this call's q/k).

## Algebra (per-flip gain, verified vs brute force to 1e-9)
out-err_h = P_h D_hv, D = v_hat - v (R,C), C = kvh*dh; J = sum_hv sum_{c in
cols(hv)} d_c^T G_hv d_c. Flip (r,c) by delta = +-d, d = 0.25*unit:
  dJ = 2*delta*M[r,c] + delta^2 * G_hv[r,r],   M[:,cols(hv)] = G_hv @ D[:,cols(hv)]
s* = -sign(M[r,c]), best gain -2d|M[r,c]| + d^2 G[r,r] (== _flip_sel with
col2 = G[r,r]). M is the (R,C) Gram image maintained by LEFT multiplication
(time mixing) -- NOT D @ (C,C); kvh matmuls (R,R)@(R,dh), R^2*C flops
(~1.1 GFLOP @ R=1024,C=512; measured 5-14 ms); per flip a rank-1 COLUMN
update M[:,c] += delta*G_hv[:,r] (O(R)); selection = top-1 per COLUMN
(columns independent). (R,R) intermediates: the bf16 Grams themselves.

## Timing (local, per v-call) / state
C=512 kvh=2, sw2/sw6 ms @ R=10/128/512/1024: 22/56, 102/305, 224/552, 355/973.
C=1024 kvh=8: 250/1936, 525/1356, 1243/4880. M-init 0.3-14.4 ms (negligible);
the round loop is memory-bound (judge ~2x). State: kvh*R^2*2 B per distinct R
= 5.06 MiB (mini cfg), 20.25 MiB (kvh=8) -- inside the 128 MiB envelope, moot.

## Recommendation
NO-SHIP. Expected online on judge-like data (mini/regime-b): -2.2pp/case
(sw6) x 250 cases ~= -550 points (sw2: ~-220); not one tested variant or
regime is positive except the unattainable exact-Gram case. The +7.4-7.8pp
oracle remains reachable ONLY with the call's own q/k at v-time (dead: judge
call isolation). P-proxy direction now closed at three levels (flat, fitted,
exact); any reopen needs a per-call q/k side-channel, which does not exist.
Run: dev/pproxy/proto.py (~7 min, writes results.json); jp_check.py for the
J_P mechanism. No .pt committed.
