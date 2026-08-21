# CPU-time audit of solution v18 (dev/audit, 2026-08-20)

Method: `prof.py` drives the UNMODIFIED solution on seeded synthetic groups
(`dev/synth.py`; data regenerated via `prof.py build`). Phase attribution via
`sol_phases.py`, an instrumented re-implementation whose outputs are
`torch.equal`-verified BIT-IDENTICAL to the original first. Box: 20-core
Windows, torch 2.13 CPU, **10 threads (default)**, episodic +/-15% load drift
-> all A/B numbers are interleaved-rep medians. Online s = local / 4.8.

## Per-phase timing (local s per group; online ~= /4.8)

Calib tokens (10,128,512,1024), test (10,128,512,1024,1024):
| phase | c1024n1024 | c2048n8192 | c4096n4096 | c8192n8192 | attn mini |
|---|---|---|---|---|---|
| calib total | 0.96 | 4.66 | 7.38 | 30.3 | 1.53 |
| weight GPTQ (values loop) | 0.23 | 1.75 | 2.20 | 9.24 | - |
| transform choice (gram+2xchol+proxy) | 0.38 | 0.80 | 2.15 | 7.57 | - |
| weight quant (_quant_chunk 6-cand) | 0.05 | 1.12 | 1.33 | 5.31 | - |
| act-GPTQ search (gram+2xchol+gptq) | 0.2 | 0.55 | 1.4 | 7.5 | - |
| alpha search / dequant / rest | 0.1 | 0.4 | 0.3 | 1.0 | 1.3 (gq_guard 0.88) |
| dynamic total (5 test calls) | 1.12 | 2.23 | 3.65 | 8.16 | 3.03 |
| -> dyn GPTQ (torch column loop) | 0.75 | 1.18 | 2.76 | 6.86 | 0.6 (qk+v) |
cProfile (c2048 calib): _gptq_quantize_values 53% cum, _quantize_weighted 25%,
_upper_cholesky_inv 5%; the GPTQ loop emits ~16 torch ops/column x C columns.

## Top-5 consumers, 50-linear-group workload (assumed mix, see below)
Mix assumption: {C=1024:12, C=2048:16, C=4096:14, C=8192:8}, N~2-4C, test lens
(10,128,512,1024,1024); with attn 50x mini-like this reproduces the observed
online 286s (+~100s fixed) within ~5%, so the mix is plausible.
| # | consumer | local s (50 grp) | online s |
|---|---|---|---|
| 1 | dyn GPTQ column loop (dyn.gptq, all C) | ~180 | ~37 |
| 2 | weight GPTQ (cal.weight_gptq, N x C^2 flops) | ~150 | ~31 |
| 3 | act-GPTQ search chol/gram + xform chol | ~110 | ~23 |
| 4 | weight quant _quant_chunk | ~80 | ~17 |
| 5 | attn calib gq_guard + dyn v-compensate | ~65 | ~14 |

## Measured speedups (all BIT-IDENTICAL: torch.equal on every param/state/dyn tensor)

Fixes (impl + harness in exp_speed.py / bench_e2e.py; solution file untouched):
- f1 vectorized `_quant_chunk`: batch sf-candidates (KB=2) along a new dim,
  one fp32 scratch, sequential `torch.where` merges replicate exact tie
  semantics; 80/80 randomized unit cases + 4 synth configs + real mini
  bit-identical. Apply when R*C >= 4M (below that torch is faster).
- f2 numpy `_gptq_quantize_values` for R <= 2048 rows: per-column elementwise
  in numpy sharing torch memory, cross-block matmul stays torch MKL; 90/90
  unit + end-to-end bit-identical. At R=8192 numpy is 1.5x SLOWER -> dispatch
  on R (weight GPTQ at N>2048 stays torch).
- f3 act-GPTQ Cholesky skip: order + Ua_o first, Ua only if Ua_o failed
  (damped-PSD chol essentially never fails). Saves one `_upper_cholesky_inv`.

| config | before (cal+dyn) | after | saved local | saved online | identical |
|---|---|---|---|---|---|
| c1024n1024 | 2.45 | 1.71 | 0.74 | 0.15 | YES |
| c2048n8192 (=mini shape) | 7.80 | 6.44 | 1.36 | 0.28 | YES |
| c4096n4096 | 11.5 | 9.0 | 2.52 | 0.53 | YES |
| c8192n8192 | 46.8 | 37.6 | 9.22 | 1.92 | YES |
| real mini linear | 8.61 | 7.24 | 1.37 | 0.29 | YES |
Micro (medians, identical outputs): wquant -19..-21%; dyn GPTQ -34..-68%
(T=10..1024); chol skip 0.08/0.45/2.84 s at C=2048/4096/8192.

## Recommended bundle: f1 + f2 + f3 (exactly as measured above)
- 50-group saving ~140 local s => **~29 s online** (25-33 across plausible
  mixes; attn adds ~1-2 s). Bit-identical outputs -> zero score risk;
  numpy 2.5.1 is in the judge env package list.
- Risks: (a) f2 numpy round/clip/where semantics verified equal incl.
  negatives/zeros; BLAS-sensitive matmuls were kept in torch; (b) f3's
  fallback recomputes Ua so the common path is provably identical;
  (c) f1 scratch = 1/8 of chunk (~134 MB at 2048x8192), inside RSS envelope.
- Funds the T=1024 lattice refinement extension (+13 s online est.) ~2x over.

## Dead ends (measured, do not retry)
- GPTQ_BLOCK 64 saves only ~4% on weight GPTQ and CHANGES numerics; 256/512
  are 1.3x/2x slower. Keep 128. ROW_CHUNK 2048 near-optimal (512 only ~11%
  on c8192 weight quant; subsumed by f1 tuned at 2048).
- Single-decomp Cholesky-of-inverse identity: reverse-chol candidate satisfies
  neither U^T U nor U U^T of H^-1; also slower. 3-decomp path is minimal.
- Gram dtype/order (gw/gwf, Ha): already 240-530 GFLOPS fp32 syrk-class; only
  dtype changes would help and those break bit-identity.
