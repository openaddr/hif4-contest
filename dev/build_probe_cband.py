"""C-band probe: separates state-size vs refinement-math as the v15 WA cause.

rev4 facts: gate C<=4096 shipped, E3 out, sweeps 5/2 -> SAME 6 linear groups
WA (18205). C>4096 path is bit-identical to v14 (which passed everything),
so the 6 failing groups are C<=4096 refined groups. Two surviving causes:
  S1 state size: gw+gwf = 2*C^2*4B = 128 MiB at C=4096 (v14's proven max
     state was u_act = 64 MiB) -> judge envelope broken by the bigger state.
  S2 refinement math corrupts specific judge data (fp32 drift / ill-conditioned
     Gram) until MSE beats the baseline and the case is marked WA.

This probe splits the refined range by C (deterministic on shape, no hash):
  C <= 2048           : carry Grams + FULL refinement   (state <= 32 MiB)
  2048 < C <= 4096    : carry Grams, SKIP refinement    (state = 128 MiB)
  C > 4096            : v14 behavior                    (no Grams)

Decode from which of the 6 known failing groups still WA:
  all 6 fail      -> they are C<=2048 -> refinement math guilty (S2);
                     128 MiB carry proved harmless in the same run.
  all 6 pass      -> they are in (2048,4096]; carry at 128 MiB is innocent,
                     refinement math at big C was the killer (S2 at big C);
                     this config is directly shippable as v16.
  mixed           -> failing ones are <=2048 (S2), passing ones big-C.
  new groups fail -> nondeterminism / something else entirely.
"""
import os
import zipfile
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "example", "solution", "solution.py")
OUT_DIR = os.path.join(ROOT, "probe", "cband_probe")
OUT_ZIP = os.path.join(ROOT, "dist", "probe_cband.zip")

src = open(SRC, encoding="utf-8").read()

# 1) new constant beside the other refinement knobs
old = "REFINE_W_ROWS = 2048        # calib rows feeding the weight objective"
new = """REFINE_USE_C = 2048         # C-band probe: dynamic refinement cap. Grams
                            # are still carried up to REFINE_MAX_C, so
                            # (USE_C, MAX_C] is a carry-only band.
REFINE_W_ROWS = 2048        # calib rows feeding the weight objective"""
assert old in src, "constants anchor"
src = src.replace(old, new)

# 2) dynamic refinement gated by the use-band
old = """    # ---- lattice refinement on the final values (transformed space) ----
    if isinstance(activation_state, dict) and R <= REFINE_T_MAX:"""
new = """    # ---- lattice refinement on the final values (transformed space) ----
    # C-band probe: refine only C <= REFINE_USE_C; (USE_C, MAX_C] carries the
    # Grams in the state but skips refinement (state-size vs math ablation).
    if (isinstance(activation_state, dict) and R <= REFINE_T_MAX
            and C <= REFINE_USE_C):"""
assert old in src, "dynamic refine anchor"
src = src.replace(old, new)

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "solution.py")
open(out_path, "w", encoding="utf-8").write(src)

spec = importlib.util.spec_from_file_location("probe_cband", out_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
import torch
p = mod.hif4_dynamic_quantize_v(torch.randn(4, 256), torch.rand(4, 16), 2, 128, None)
assert p["mant"].shape[-1] == 4
print("probe module OK")

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(out_path, "solution.py")
print("wrote", OUT_ZIP)
