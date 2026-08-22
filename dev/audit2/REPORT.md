# CPU-time audit round 2 (dev/audit2, v25, 2026-08-22)

Method: prof2.py/cProfile on unmodified v25; fixes as textual patches of a
loaded copy (bundle.py, solution.py untouched); interleaved A/B, medians of 3;
bit-identity gate 8/8 PASS (4 synth configs incl. both C>2048 hash branches,
2 extra seeds, real mini linear, real mini attn with q/k/v carry replay).
Pricing: savings are memory-bound (elementwise/gather rounds) -> online =
local/2; matmul dedupes -> /4.8. All savings are structural (both regimes).

## Where v25 time actually goes now (c2048_n8192, cProfile)
calib 5.9s: wGPTQ(torch,R>2048) 1.67 | _refine_weight 1.96 | _quant 0.89 |
gptq-np 0.29 | chol 0.17. dyn 3.6s: _refine_act 2.48 (71%!) | gptq-np 0.77 |
_quant 0.28. Attn mini: calib 1.28 + dyn 3.46 (v-comp 1.73, q 1.4).
KEY: _flip_sel recomputes (-2*d) and (d*d)*col2 -- both (T,C)-sized -- every
round, plus 2 (T,C) bound comparisons; rounds are the top consumer.

## Candidates
| # | item | local s saved (measured) | est online s | bit-id | verdict |
|---|---|---|---|---|---|
| C4 | act-refine round opt: hoist -2*d,(d*d)*col2; cache v4 bounds (scatter-maintained); dirn at (T,1); in-place/out= | -23..31% of _refine_act (T=512/1024); dyn -0.37 (c1024) .. -0.73 (c2048)/grp | ~8-10 | Y (30 unit + e2e) | SHIP |
| C3 | weight-refine: same round opt + chunk-outer restructure (rows independent -> same flip seq) + REFINE_W_CHUNK 2048->1024 | -23..26% of sweep (N8192xC2048 1.63->1.25; chunk sweep: 1024 fastest) | ~4-6 | Y (chunk-invariance proven+measured) | SHIP |
| E8 | numpy round loop for T<=32 dyn refinement (torch dispatch-bound there) | T=10,C2048: 0.084->0.029 (-65%); loses >=T=64 (crossover 32/64) | ~1.5-2 | Y (24 tie-storm unit + e2e) | SHIP |
| C2 | _quant_chunk_vec threshold 4M->2M | -6..-20% on flipped shapes (T1024xC2048 dyn, 512x4096, 2048-row w-chunks C>=1024); ties below 2M | ~0.5-1 (attn 0-1.7 incl.) | Y (16 shapes + e2e) | SHIP |
| C5 | dedupe xh_pick@w_final.T (ref/ref2), hoist a_big@w[rows].T out of alpha loop | ~0.02-0.1/grp | ~0.2-0.4 | Y (same operands) | SHIP (free) |
| C1 | numpy GPTQ dispatch boundary 2048->2560 | numpy wins R=2560 by 3-8%, loses >=3072 (all C) | ~0.1 | Y | skip (negligible) |

## Measured dead ends (do not retry)
- in-place torch GPTQ column loop (R>2048): 0.3-1.3% = noise.
- numpy _gptq_quantize_batched (attn q/k/v): 2.6x SLOWER at B=16 (torch
  elementwise is multithreaded; numpy is not). Wins only B=2,T<=512 (~22ms).
- out=-buffered numpy GPTQ columns: -9..-16% (slower).
- copysign for q=s*m*ui: diverges on w=-0.0 (NVFP4 carriers can be -0.0).
- gw.float() cross-call caching: judge clones the state per call (identity
  changes) -- useless; storing fp32 grams in-state would double state size.

## Bundle e2e (bundle.py, medians of 3 interleaved)
| config | calib save | dyn save | total/grp | bit-id |
|---|---|---|---|---|
| c1024_n1024 | +0.05 | +0.37 | 0.41 | Y |
| c2048_n8192 | +0.38 | +0.73 | 1.09 | Y |
| c3072_n3072 (hash-odd, no refine) | -0.01 | +0.06 | 0.08 | Y |
| c4096_n4096 (hash-even) | +0.52 | +0.56 | 0.94 | Y |
| real mini linear | -- | -- | 1.45 | Y |
| attn mini | -- | -- | ~0.04 | Y |

## Recommended bundle: C4+C3+E8+C2+C5 -> ~17 s online (range 14-19)
Mix {C1024:12, C2048:16, C4096:14 (7 hash-even), C8192:8} + 50 attn, N~2-4C
(refine savings scale with N; measured at N=C..4C): ~32-35 local s, ~97%
memory-bound -> /2 -> **~16-18 online s**, daytime and deep-night alike.
On top: round-1 bundle; artifact ~274s -> ~257s daytime (well under 288),
~248 -> ~231 deep-night.
Risks: all items bit-identical by construction + gate; E8 adds a numpy path
(numpy 2.5.1 proven in judge env by round-1 f2; argmin first-min matches
torch CPU, verified incl. all-inf tie rounds). REFINE_W_CHUNK=1024 keeps
peak RSS ~unchanged (smaller row blocks, per-chunk (1024,C) buffers).
Confidence: C4/C3 high (multiple shapes, real data), E8 high-ish (unit+e2e),
C2/C5 medium (small magnitudes).
