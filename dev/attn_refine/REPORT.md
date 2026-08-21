# Attn V-path output-projected lattice refinement — VERDICT: NO-SHIP

## Transformed-space analysis (task 1)
- v is quantized in the RAW dequantized-NVFP4 fp32 space: x = dequantize_nvfp4(v).float(),
  (T, C), C = kvh*dh = 512 here. v_state is ALWAYS None (no smoothing s; `_dyn_table`
  has_scale=False) and V is NEVER rotated (only Q/K use `_make_R`). No channel transform.
- Wv in that space = IDENTITY: the judge's attention has NO output projection
  (task book + `_attention_out`/`hif4.attn_ref` = softmax(qk^T/sqrt(dh)) @ v).
  kv-head column blocks are disjoint, so G = Wv^T Wv = I_C and the uniform-row-weight
  objective ||(v_hat-v)@Wv^T||^2 degenerates to plain ||v_hat-v||_F^2.
- Consequence (proof): online the judge isolates calls -> the q/k carry never assembles
  (probe_carry verdict) -> v values are round-to-nearest on the searched grid. With G=I,
  flip gain g = -2*d*|M| + d^2 >= 0 because |M| = |v_hat - v| <= d/2: NO improving flip
  exists. Measured: 0/524288 elements changed, all depths, all cases.
- The real V coupling is across TIME (P mixes rows), not channels; the per-row
  linear-side machinery cannot represent it under G=I. Exact-P refinement needs a (T,T)
  per-head Gram and top-1 per COLUMN (columns independent in J_P) = the oracle below.

## Score table (dscore vs solution baseline, pp of case score; mini attn 5 cases)
baseline: carry +0.552/+0.535/+0.620/+0.628/+0.630 ; plain +0.471/+0.435/+0.532/+0.548/+0.560

| variant | path | t0 | t1 | t2 | t3 | t4 | avg |
|---|---|---|---|---|---|---|---|
| uniform sw2/4/6 | plain (=ONLINE) | 0 | 0 | 0 | 0 | 0 | 0.00 |
| uniform sw2 | carry (local only) | -3.94 | -4.29 | -4.86 | -5.18 | -4.87 | -4.63 |
| uniform sw4 | carry | -4.35 | -4.91 | -5.61 | -6.21 | -5.77 | -5.37 |
| uniform sw6 | carry | -4.35 | -5.04 | -5.60 | -6.12 | -5.71 | -5.36 |
| oracle-P sw2 | plain | +6.39 | +6.15 | +7.74 | +8.35 | +8.16 | +7.36 |
| oracle-P sw6 | plain | +6.39 | +6.15 | +7.89 | +9.40 | +9.19 | +7.80 |
| oracle-P sw6 | carry | -0.65 | -1.80 | +0.55 | +2.11 | +2.33 | +0.51 |
| flat-P sw2 | plain | -14.4 | -34.5 | -61.5 | -104.6 | -78.8 | -58.8 |

Objectives: uniform on carry removes 15.7-65.4% of the PROXY ||dv||^2 but INCREASES the
true exact-P V-error J_P by 0.1-7.5% (anti-correlated: it undoes the local GPTQ
compensation). Oracle removes 15-28% of true J_P (plain). V-error removed online by the
shippable design: exactly 0% (no-op proof).

## Timing (local, per v-call, C=512)
uniform refine: sw2 7-71 ms, sw4 14-134 ms, sw6 25-224 ms (T=10..1024). oracle-P: sw2
14-254 ms, sw6 39-478 ms. T-adaptive (linear rule 5/2) sits inside [2,6]; curves
saturate by sw4 (uniform) / sw6 (oracle) — depth is not the lever.

## State cost
G = Wv^T Wv = I is analytic -> 0 bytes carried (v_state stays None). A literal bf16
C x C Gram would be 2*C^2 B = 512 KiB @ C=512 per attention group (general: 2C^2 B).

## Recommendation
NO-SHIP the uniform-row-weight V refinement:
1. Online it is a provable no-op (0 changed elements; identity Wv + RTN values).
2. Where it does fire (local compensate path) it LOSES 4.6-5.4pp/case.
3. Flat-P / calibrated-P proxies are catastrophically wrong on structured mini data.
Expected online value: 0 under 1x..7x transfer (if the carry ever revived, approx
-1160..-1350 online at 1x — actively dangerous). Upper bound for any future P-aware
variant (oracle; unshippable today since the v-call receives only v and cross-call
carry is dead on the judge): +7.4..+7.8pp/case local -> ~+1850..+1950 online at 1x,
~+13k..+14k at 7x. Reopen only if q/k ever become available at v-time. Direction closed.

Run: dev/attn_refine/proto.py (deterministic, torch.manual_seed(0), ~2 min, writes
results.json; sanity: loop reproduces SOL._refine_act_values(x,v,u,I,I) bitwise).
