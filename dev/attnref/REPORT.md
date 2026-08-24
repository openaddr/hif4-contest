# Attention-side lattice refinement — VERDICT: NO-SHIP

Mechanism explored: run the linear-side greedy top-1 lattice refinement on the
dynamic attention Q/K calls, weighted by the calibration per-head logit-space
Hessians (Hk for Q, Hq for K), with bf16 H carried in the state and a hold-out
guard in calibration. V-side uniform sweep re-confirmed as a theorem no-op.

- Baseline: `example/solution/solution.py` (v33), copied to `dev/attnref/solution.py`.
  Mainline untouched. Mechanism flag-gated, defaults OFF (bit-identity verified).
- Harness: `dev/attnref/bench.py` (modes ident / vcheck / mini / synth / rounds).
- Raw data: `dev/attnref/results.json`. Prior V-path verdict: `REPORT_vpath_0821.md`.
- Python: C:\App\env\Python\python.exe, CPU torch, local timings (judge ≈ local/2
  for memory-bound round loops per the v23/24 postmortem; /4.8 for other phases).

## 1. Verdict

**NO-SHIP.** All three SHIP gates fail:

| Gate | Required | Measured |
|---|---|---|
| mini guarded avg | ≥ +1.5 pp/case | **+1.249 pp/case** (cases: +0.92/+0.95/+1.63/+1.13/+1.62) |
| synth buckets ≥ +2pp | ≥ 1 bucket | **0/7 buckets**; unguarded −8.1 … −24.9 pp/case (guard rejects all 7) |
| typical call cost | ≤ 15–25 ms online | Q-refine @T=512/1024 = 100–470 ms local at r20 (**50–235 ms+ online**); only T≤128 tiers fit |

Judge-realistic expectation: judge attention data is iid-like (transfer 0.22;
Q/K GPTQ guards reject on iid synth — v7 history, reconfirmed here: gq=0 on all
synth shapes). On iid-like data the hold-out guard REJECTS refinement (verified
5/5 synth shapes incl. outlier/spread regimes, 0 false accepts) → expected
online value ≈ 0, while the guard's calibration cost (+0.3 s local/group at sw4
≈ +3 s online over 50 groups) is paid unconditionally — against a ~+5 s
remaining buyable budget.

## 2. Mechanism (implemented in dev/attnref/solution.py)

- `q_state["H"] = Hk`, `k_state["H"] = Hq` (bf16, kvh×dh×dh; fp32 Grams in
  state WA'd on the judge → bf16 mandatory). State +2·kvh·dh² B/role:
  mini (2,256) +256 KiB/role; worst realistic (32,128) +1 MiB/role. No
  envelope risk (attention states stay ≪ 128 MiB).
- `_refine_attn_heads(x, values, unit, H, num_heads, sweeps)`: greedy top-1
  coordinate descent over all B·T rows (row = (head, token); rows independent
  → exact CD), active-set compaction mirroring `_rounds_active`, rank-1 Gram
  updates. Objective per kv-group J = ||dq @ K_calib^T||² (Q) /
  ||dk @ Q_calib^T||² (K) — identical algebra to the linear side with
  gw = gwf = H. GQA head mapping h → h·kvh//qh.
- `_dyn_qk` refines the FINAL values (post-GPTQ when gq=1, else table values).
- Calibration hold-out guard: strided (≤512 rows; row-independent ⇒ faithful)
  last-calib-sample evaluation of the FULL attention output MSE, sequential
  q-then-k, accept only on improvement. Guard rejects → rf=0 and H not carried
  → dynamic path bit-identical to v33.
- Config: `ATTN_REFINE_Q_SWEEPS / K / V`, `ATTN_REFINE_GUARD`,
  `ATTN_REFINE_GUARD_MAXT`, `ATTN_REFINE_FORCE_H`, `ATTN_GPTQ_ENABLE`,
  `_REF_ATTN_ROUNDS` (rounds-per-sweep override).

## 3. Value measurements

### 3.1 mini (real structured data; qh16/kvh2/dh256; T=10..1024)

Baseline (v33 path) avg score +0.5092/case vs exact-alg1 judge baseline.

| config | dscore pp/case (t0..t4, T=10/128/512/1024/1024) | avg |
|---|---|---|
| Q-refine only (sw4) | +0.231 +0.248 +0.699 +0.666 +0.691 | +0.507 |
| K-refine only (sw4) | +0.622 +0.570 +0.938 +0.500 +0.909 | +0.708 |
| Q+K (sw4) | +0.916 +0.954 +1.630 +1.131 +1.616 | **+1.249** |
| Q+K guarded (sw4..32) | identical (guard accepts at every depth) | +1.249 |
| {table+refine}, no GPTQ (sw4) | +11.81 +11.96 +14.40 +10.97 +13.11 vs table base | +12.45 |

Absolute: {GPTQ+refine} 0.5217 > {refine only} 0.5142 > {GPTQ only} 0.5092 >
{table only} 0.3898. Refinement alone ≈ GPTQ (substitutes), stacking adds
+1.25pp over the GPTQ path. **Sweep curve is exactly flat 4→32** (identical
values; rows freeze within ~1–3 sweeps — the loop early-exits, so extra sweeps
cost nothing). Gains are T-uniform (+0.9…+1.6pp) while cost scales with T.

### 3.2 synthetic (7 shapes, spread 0.4–0.7, outlier variants)

Unguarded Q+K refinement (sw4; identical at sw8):

| shape (qh/kvh/dh, T set) | avg dscore pp/case | per-case |
|---|---|---|
| a16_4_64 (10..1024) | **−8.06** | +4.19 −18.35 −4.70 −13.37 |
| a32_8_128 (10..1024) | **−8.84** | +14.65 −22.34 −14.87 −12.78 |
| a8_8_128 (10..1024) | **−14.10** | −14.69 −11.28 −6.16 −24.29 |
| a16_2_256 (10..1024) | **−13.44** | −1.81 −17.53 −18.47 −15.92 |
| a4_2_64 (10..2048) | **−9.92** | −10.46 −9.38 −7.05 −12.81 |
| a16_8_64 (10..2048) | **−10.29** | −10.72 −11.73 −8.55 −10.15 |
| a32_8_128_sp (10..1024, spread .7 + outliers) | **−24.88** | −67.15 −13.60 −11.45 −7.32 |

**Guarded: rf rejected (rf=None → no H carried, no cost, bit-identical
dynamic path) on 7/7 shapes; accepted on mini.** Rejection verified genuine
(unguarded run of the same shape forces rf=4 fine).

### 3.3 J-split diagnostic (why): fit-Hessian vs fresh-test-Hessian objective

| data | J_cal removed | J_test removed | verdict |
|---|---|---|---|
| mini, Q@T=1024 | +18.1% | **+14.7%** | real cross-channel structure transfers |
| synth a32_8_128 | +40.5% | **−31.3%** | calib-Gram off-diagonals are sampling noise |
| synth a16_4_64 | +31.8% | **−22.1%** | same |

On iid-like data the per-head Gram off-diagonals are pure calibration noise;
greedy flips exploit them, J_cal falls, the TRUE objective rises. This is the
4th instance of the STRATEGY §7 theorem "sample-fitted objectives never
transfer" (P-proxy, fitted rotations, cross-call cache, now Hessian refinement).

## 4. Cost measurements (local ms; judge ≈ /2 for the round loop)

### 4.1 Baseline dynamic calls (v33 path, mini shape)

T=10/128/512/1024/1024: q 49/157/440/678/879, k 43/52/112/168/142,
v 8/24/32/46/35. (Dyn-call timing variance band is wide (±30%) — identical
values scored 440–879 ms; the dedicated micro numbers below are authoritative
for the refine increment.)

### 4.2 Q-refine micro (post-GPTQ values, bf16 H, sw=1, median of 3)

| T | r2 | r4 | r8 | r20 | J removed (mini) |
|---|---|---|---|---|---|
| 10 | 5 | 5 | 8 | 17 | 7.3→16.8% |
| 128 | 12 | 29 | 44 | 57 | 7.9→18.0% |
| 512 | 57 | 103 | 195 | 212 | 8.0→18.1% |
| 1024 | 82 | 155 | 279 | 473 | 7.9→18.0% |

- Cost ∝ (heads·T·dh) per round as expected; r2 captures ~44% of the J-value
  at ~17% of r20 cost (the cost-value frontier knob).
- Deployed sw4 (early-exit) ≈ 2–3× the r20 column: q@T=1024 ≈ 0.9–1.3 s local
  (dyn-call delta), q@T=512 ≈ 0.46 s, q@T=128 ≈ 20–70 ms, q@T=10 ≈ 12 ms.
  K-side ≈ kvh/qh of Q cost.
- Online per-call gate (15–25 ms) is met ONLY by T≤128 tiers at small depth.
- Calibration guard cost: mini +0.3 s local at sw4 (hold strided to 512 rows;
  hold T0=1024), +0.7–1.3 s at sw32 ⇒ ≈ +3 s online across 50 groups at sw4.
- Calibration H accumulation: unchanged (Hq/Hk already computed for GPTQ; the
  bf16 conversion is negligible).

## 5. Interaction decomposition (task item 5)

- mini {table+refine, no GPTQ}: +12.45 pp over table base → absolute 0.5142.
- mini {table+GPTQ+refine}: +1.25 pp over GPTQ base → absolute 0.5217 (best).
- GPTQ has NOT eaten the whole refinement gain (stack still +1.25pp over the
  GPTQ path) but the two are near-substitutes: refine-alone recovers 97% of
  the GPTQ+refine score at similar-or-higher dynamic cost. On synth the
  {table+GPTQ+refine} cell is unreachable by guard (GPTQ itself is rejected
  there, gq=0 on all 7 shapes — same iid pattern as v7 history).

## 6. V-side (task item 1c) — theorem no-op re-confirmed

0 changed elements at sweeps {4,8,16} on every mini case (T=10..1024).
Wv = I (no output projection) ⇒ uniform-weight objective is per-element
separable ⇒ RTN table values are already optimal ⇒ every flip gain ≥ 0.
Matches REPORT_vpath_0821.md; no further V-side machinery built (per
coordinator directive 2026-08-24).

## 7. Safety/format verification

- Flags-off bit-identity vs v33: calibration state, all q/k/v dynamic outputs
  `torch.equal` on all 5 mini cases (mode `ident`).
- Flags-on: `validate_frozen_state` clean (bf16 H allowed), params clean,
  mant grid legal (multiples of 0.25 in [0,1.75]), official
  `example/self_check.py` on `dev/attnref`: **22/22 PASSED**.
- Deterministic: no RNG in guard or refinement; guarded calibration for mini
  accepted at every sweep depth with identical outputs (sw4..32).

## 8. Risk list (if anyone reopens this)

1. **Unguarded deployment on iid-like judge data: −8…−25 pp/case × 250
   attention cases ≈ −2,000…−6,000 online.** The hold-out guard is the only
   barrier; verified 0/7 false accepts on synth, but its false-accept rate on
   the judge's actual mix is unobservable offline.
2. Guard calibration cost is unconditional (~+3 s online at sw4) even when
   every group rejects — against a ~+5 s total remaining budget.
3. Memory-bound cost model (judge ≈ local/2) has precedent from the linear
   side only; if judge attention runs at /4.8 the large-T cost halves but is
   still ≥ 10× over the gate.
4. bf16-H vs fp32-H guard mismatch (guard evaluates fp32, deployment refines
   with bf16-rounded H): negligible in measurement (mini accepted; J-curves
   indistinguishable), but a nonzero determinism-of-acceptance edge.

## 9. Reopen triggers (recorded, not acted on)

- Any evidence that judge attention calibration carries real cross-channel
  structure (e.g., a bucketed probe showing Q/K-GPTQ acceptance/value online).
  Then: guard already protects iid groups; ship shape = small-T tier
  (T≤128, r4–r8: ≤ 44 ms local ≈ 6–22 ms online/call) at sw1–2.
- Deeper rounds never pay: J-curve saturates ~r20; sweeps beyond convergence
  are free but valueless. The only real knobs are rounds and the T-tier.
