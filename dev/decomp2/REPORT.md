# decomp2: residual decomposition, CURRENT config (v30)
solution.py moved 24/12/5 -> 32/14/6 mid-study (calibration bit-identical to
v29; one re-run group scored identically = converged). Seeds/scoring
identical to dev/decomp study.py (exact Alg.1 baseline). study2.py
(pop/t2048/rep), anatomy.py (task 3); results_*.json + cache/ (no .pt in git).

## Task 1 - ship score (pp): by T (40 groups C{512..8192}) and by C
refined = grams carried (C<=2048 or hash-even 4096; 25/33 C<=4096 groups).
| T | all | refined | act/w |   | C | T10 | T1024 | all | dt/case | act/w |
|----|-----|---------|-------|---|------|------|-------|--------|-------
| 10 | 59.6 | 65.1 | 0.99 |   | 512 | 50.7 | 47.5 | 49.7 | 0.37s | 0.81 |
| 128 | 59.3 | 65.4 | 0.94 |   | 1024 | 65.7 | 58.7 | 62.7 | 0.78s | 0.95 |
| 512 | 57.1 | 62.1 | 0.91 |   | 2048 | 75.0 | 63.3 | 68.9 | 1.82s | 1.01 |
| 1024 | 53.9 | 57.3 | 0.89 |   | 4096 | 53.6 | 50.4 | 51.8 | 1.08s | 1.00 |
                                 | 8192 | 52.8 | 49.6 | 50.5 | 1.39s | 1.00 |
Small-T deep tiers work (T=10/128 best buckets, +8 over T=1024; the T=1024
lag is the 5-6-sweep tail). Weight side relatively dominant at small C.

## Population gaps (2a/2b/2c)
2a C=4096 forced-_e4 (8 groups; ship==all4096 on the 1/8 even, ==nohash 7/8):
  all4096 80.97/79.86/73.87/66.69 (T10/128/512/1024) vs nohash 48.57/47.55/
  47.30/46.99 -> gap +32.4/+32.3/+26.6/+19.7 pp per case (T-mix mean +26.1).
  Cost local: cal 9.6->14.1 s/group, dynamic 0.74->3.25 s/call.
2b T=2048 (independent seeded draw; pop cal states reused): ship T2048 sits
  15-20pp BELOW T1024 (mse 1.3-2.1x); REFINE_T_MAX->4096 recovers:
  C512 +15.8, C1024 +18.7, C2048 +19.1, hash-even C4096 +27.4 pp/case
  (C4096 mean +3.4 = 7/8 gram-less). dt 0.2/0.5/1.2/2.3s -> 0.6/1.4/3.3/2.6s.
2c C=8192 (never refined) 50.5 all-T vs C=2048 refined 68.9 -> 18.4pp gap.
  LOCKED: bf16 grams 268 MiB + u_act 268 MiB >> 128 MiB pass / 192 MiB fail
  envelope; needs a structurally smaller carry, not a gate change.

## Task 3 - residual anatomy within refined calls (T=10; 24 C<=2048 groups)
Dynamic-call replication bit-exact 24/24. Shares of mse_play: act 5.47x,
weight 5.33x, cross -10.79x: the additive split is DEAD at deep tiers -- the
flip lattice optimizes against q_used (carried Grams), so the act residual
systematically CANCELS weight error; what survives cancellation is the target.
(i) weight-side: co-equal contributor (act/w 0.9-1.0 refined; 0.81 at C512).
(ii) act lattice-converged: 193/240 rows have NO improving single flip after
  the ship tier. Clean groups (o=0): 120/120 rows, 12/12 bit-identical at 96
  sweeps. Outlier groups (o=.002): 73/120 rows, 1/12 bit-identical; s96
  leaves +1.8pp/case -> the T<=32 tier is NOT converged on outlier inputs
  (deepening ~free: 56->170 ms local/call).
(iii) grid-lock: SMALL. Act per-block grid re-rank vs the exact carried-Gram
  objective (deployable online, re-flip acceptance): clean +0.15pp, outlier
  +3.3pp (max +9.8) of the T=10 case; 16-cand grid no better than 6 (+3.1).
  Weight grid re-anchor upper bound (exhaustive per-block re-search vs exact
  calib objective, 2 passes): in-sample fit-J -11.1% BUT hold-out x1.27 WORSE
  in 24/24 groups and test d -18..-22pp at every T -- pure fit-row overfit;
  a deployable guard rejects 24/24. The grid is NOT the binding constraint.

## Ranked pots (250-call online suite; rough, transfer 1.0x small-T / 0.28x T1024)
1. C=4096 hash-odd refinement (2a): +26.1pp/case synthetic = ~+144pp per
   affected group -> +720..1440 online at f4096=10..20% of 50 groups. Cost:
   +4.5s cal + 2.5s/call local (judge ~x2), state stays at the shipped
   128 MiB hash-even envelope; time budget must come from sweep re-tiering.
2. REFINE_T_MAX 1024->2048 (2b): +16..27pp per T=2048 call, +0.4..2.2s
   local/call. ZERO if the judge has no T>1024 linear calls (stated mix says
   none) -- verify with one instrumented run before spending budget.
3. Outlier-targeted small-T polish: deeper T<=32 tier when flips remain
   (+1.8pp/case, ~free) + act grid re-rank vs carried Grams (+3.3pp per T=10
   case on outlier groups, ~0 on clean; slim to 1 pass, 2.6-2.9s measured is
   too slow). Ceiling ~+50..150 online at ~0-10s total if ~half the online
   groups are outlier-like.

Dead ends re-confirmed: weight grid re-anchor (0/24 hold-out), act re-rank on
clean inputs, sf-prior, E3 deepening, C8192 grams (state size).
