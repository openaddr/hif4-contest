# rerank: slimming the decomp2 outlier-group grid re-rank (10x attempt)

fast.py + profile_proto.py; solution.py untouched; decomp2 groups/cache reused.

## Where the prototype's 2.6-2.9s went (profile_proto.py, C=2048 T=10)
91% FULL_refine (96 per-block _refine_act_values, 32-40 sweeps x 20 rounds)
| 4% redundant (T,C)@(C,C) Jt_try matmuls | 2% candidate re-quant dispatch.
MECHANISM REDISCOVERY: the dJ candidate ranking is dead weight -- after ship
refinement the incumbent is locally optimal, every candidate's plain dJ > 0,
argmin = identity row on 97-100% of tries, and the clamp maps it to the FIXED
coarsest candidate (sf 2^e0*1.25*4, lv2=lv3=2). ~100% of accepted swaps are
that path: the value is per-block re-grid + re-refinement (basin hopping on
the carried-Gram J), and a try crosses below the incumbent J only after
~80-200 top-1 flips (measured trajectories).

## Slim version (fast.py)
(a) mask: v4 pinned at +-7 -> measured DENSE (~97% of pairs, clean groups
too): correct detector, but the sparsity premise fails at 64-block size.
(b) only masked pairs perturbed, one 64-block per row per iteration (rows are
independent in J and in the greedy flips -> prototype order preserved, all
rows batched).  (c) M = v@gw - x@gwf maintained exactly (rank-64 perturb
updates + ship _rounds_np in-place; rejected tries restore the full saved
row): zero big matmuls.  (d) batched dJ (use_dj=True) measured to change
nothing (quirk dominance) -> default swaps to the fixed candidate.
(e) short re-flip (ref_sweeps) + exact per-row J acceptance, as prototype.
Deploy: sweeps 4 (C<=1024) / 1 (C=2048), 3 passes, pair budget 240/256
(global top by pinned-count + per-row cap bounding the iteration count).

## Timing (median of 3 reps, outlier groups, re-rank stage, local)
T=10: C512 112ms | C1024 271ms | C2048 272ms (264-299) -- target <=300ms OK
T=128: C512 175ms | C1024 315ms | C2048 259ms -- target <=600ms OK
mean vs prototype 2.47s -> 162ms = x17 (T=10, all 24 groups).

## Score reproduction (T=10, same-run prototype; decomp2's +3.33pp was v30 --
v31's deeper tiny-T tier cut the headroom to +2.85pp)
outlier mean: proto +2.85 -> fast +1.67 (59%).  By C: 512 96%, 1024 79%,
2048 31%.  clean +0.15 -> +0.03 (no harm).  Depth knob at C=2048: s1 0.30s
=31%, s4 0.72s =84-99%, s16 2.6s =100% of prototype d_pp.
Grid match on re-ranked blocks: mean 92.4% (min 64% C=2048 outlier; C512
96-100%, C1024 90.6-100%).  Of 496 mismatches only 31 are objective ties
(<1e-6 rel): real decision divergence; even at prototype depth (s16, 2.6s)
C=2048 match stays 88-100% -> the 95% gate is unreachable at C=2048 at ANY
budget <= prototype cost.

## Gates: 3a match>=95% FAIL (92.4%); 3b +3.3pp FAIL at budget (59%; 31% at
C=2048); 4 timing PASS (both buckets, all C).

## VERDICT: NO-SHIP at the 0.3s budget
The 10x is real (x17) but amputates the mechanism: the gain lives in deep
per-try re-refinement the budget cannot fund at C=2048.  If shipped anyway
(+1.67pp x ~50 small-T cases x 0.3-1.0 transfer) ~ +25-85 online -- below
the +50-150 ceiling and overlapping the FREE deeper tiny-T tier (+1.8pp,
decomp2).  Revisit if ~0.7-1s/call at C=2048 becomes affordable (s4 there
= 84-99% reproduction); at C<=1024 the budget already fits with 79-96%.
