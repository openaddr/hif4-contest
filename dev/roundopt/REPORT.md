# roundopt: lattice round-loop optimization -- SHIPPED (patched copy)

## What shipped (idea 2 + pass-reduction; ideas 1/3 measured dead)
The round loop is **per-row independent**: `M += coef[:,None]*gw[idx]` touches
row r with row r's own (idx[r], coef[r]) only, so a row with no legal
improving flip (dr=0) is value-unchanged and **frozen forever** (verified:
flip rounds per row are a strict prefix 0..K_r, 160/160 calls). Shipped into
`dev/roundopt/patched/solution.py` (ship file untouched; `make_patch.py`):
1. `_rounds_active`: compact to the flipping set each round (monotone
   shrink), stop when empty; identical per-row op order -> identical flips.
2. Pass-reduced round, ~13 -> ~8 (A,C) memory passes (loop is memory-bound):
   `g = d2col + |M|*(-2d)` in ONE addcmul (no-FMA on this CPU, bit-equal);
   legality as illegal-masks `v4>=7 / v4<=-7` with INF fill only where
   ILLEGAL and `fin = g_sel<0` (selection provably identical: negatives sort
   below positives -> same argmin whenever a flip exists; no-flip rounds are
   value no-ops); `M += coef*gw[idx]` via addcmul.
3. numpy tail at A<=32 rows (torch dispatch-bound), numpy `_rounds_np`
   flattened + early exit when a round flips nothing.
- Idea 1 (heap) dead: every ACTIVE row flips EVERY round (prefix property),
  so there is no selection to defer and all C gains of a flipping row change
  every round (rank-1 touches the whole row).
- Idea 3 (batching M updates) dead: same reason -- next flip needs updated M.
- Weight-side `_refine_weight_values` chunk loop -> same `_rounds_active`.

## Timing (median per dynamic call, realistic captured inputs; 32 groups)
refine-only `_refine_act_values` ship -> v2n (this is the hot loop):
| T | C=1024 | C=2048 |
|---|---|---|
| 10  | 45.7 -> 35.3 ms (1.31x) | 80.4 -> 76.1 ms (1.06x) |
| 128 | 471 -> 235 ms (2.00x)   | 609 -> 400 ms (1.52x) |
| 512 | 427 -> 335 ms (1.28x)   | 944 -> 639 ms (1.48x) |
|1024 | 414 -> 284 ms (1.45x)   | 946 -> 620 ms (1.53x) |
Whole dynamic call e2e (8 groups + mini_sample): 1.11-2.85x per call
(mean ~1.35x; best at T=128 where the 640-round tier collapses to a
handful of active rows: active row-rounds 67.6% at T=128, 91.9% at T=512,
100% at T=1024 -- the pass-reduction carries the large-T win).
Calibration (weight loop): 2.9 -> 2.6 s at N=8192 (no regression).

## Bit-identity evidence
- 160/160 captured refine calls (32 groups: C{1024,2048} x N{1024,8192} x
  spread{0.5,0.9} x outlier{0,0.002} x 2 seeds, test T=10/128/512/1024/1024):
  identical (round,row,col,dr) flip TRACE vs an instrumented replica that
  itself torch.equals the ship module output; final v4*d torch.equal.
- e2e: ship vs patched calibrate bit-equal (weight_params, q_used-derived
  state) and all 40 synthetic + 5 mini_sample/linear.pt dynamic outputs
  torch.equal (`verify_e2e.py`, log e2e.log: "E2E PASS").
- self_check 22/22 on dev/roundopt/patched.
- addcmul == mul_+add_ bit-identity verified on CPU (no FMA re-assoc).

## Online estimate
Round loop ~53% of the linear pipeline (audit3). Local refine savings:
~0.6 s/group (C=1024), ~1.2 s/group (C=2048) per 5-call group; ~30-40 s
local across a judge-like 50-group linear mix + weight side. Memory-bound
judge pricing ~2x local -> **est 15-25 online seconds saved** (at 40-70
pts/10s: ~+60-175 points of night-tier headroom, e.g. deeper sweeps).

## Integration diff (what to apply to solution.py when shipping)
- `_rounds_np`: flatten loops + `if not fin.any(): break` (op order kept).
- insert `_rounds_active` (see make_patch.py ROUNDS_ACTIVE_SRC).
- `_refine_act_values` T>32 loop -> 1 call: `_rounds_active(M, v4, d, neg2d,
  d2col, gw, n_sweeps * REFINE_ROUNDS)`.
- `_refine_weight_values` chunk loop -> `_rounds_active(Ac, v4c, d[i1:i2],
  neg2d, d2col, Gxx, REFINE_W_SWEEPS * REFINE_ROUNDS)`.
Artifacts: capture.py (input capture), core.py (verified twins + traces),
analyze.py (gate), bench.py, verify_e2e.py, make_patch.py, patched/.
