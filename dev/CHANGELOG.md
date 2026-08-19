## v1 - 2026-08-19 18:56:02
- artifact: dist/solution_v1.zip
- note: v1: greedy hierarchy + E6M2 scale search (8 cands), output-error-weighted; local mini_sample total +6.66/10 cases (linear ~+0.78, attn ~+0.55)

## v2 - 2026-08-19 19:25:27
- artifact: dist/solution_v2.zip
- note: v2: FIX root cause of -30091: sf candidate grid re-anchored at absmax/7 (was 2^floor(log2 amax), 2-3.5x too large, crushing small sub-blocks). + 8-cand weighted search, lv2/lv3 MSE refinement, AWQ smoothing (linear), q/k paired smoothing, V positional weights. Local vs norm7 baseline: linear +81%, attn +34%/case; self_check 22/22; est 92s local full-run

