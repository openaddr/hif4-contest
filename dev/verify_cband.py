"""Verify the C-band probe behaves exactly as designed on synthetic groups.

Bands: C<=2048 refines (output differs from a no-refine variant),
2048<C<=4096 carries Grams but must be bit-identical to no-refine.
"""
import importlib.util
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dev"))
import synth  # noqa: E402

PROBE = os.path.join(ROOT, "probe", "cband_probe", "solution.py")


def load(name, use_c):
    src = open(PROBE, encoding="utf-8").read()
    src = src.replace("REFINE_USE_C = 2048", f"REFINE_USE_C = {use_c}")
    assert f"REFINE_USE_C = {use_c}" in src
    spec = importlib.util.spec_from_file_location(name, PROBE)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["_src"] = src
    import types
    code = compile(src, PROBE, "exec")
    ns = {"__name__": name, "__file__": PROBE}
    exec(code, ns)
    return types.SimpleNamespace(**ns)


def run_case(name, C, N, calib_T, test_T, spread, outp, wspread):
    probe = load("probe_mod", 2048)
    noref = load("noref_mod", 0)
    g = synth.make_linear_group(1234 + C, N, C, tokens=calib_T,
                                spread=spread, outlier_p=outp, w_spread=wspread)
    g2 = synth.make_linear_group(4321 + C, N, C, tokens=test_T,
                                 spread=spread, outlier_p=outp, w_spread=wspread)
    torch.manual_seed(0)
    t0 = time.perf_counter()
    out = probe.hif4_calibration_and_quantize_weight(
        g["weight"][0], g["weight"][1], g["calib_activation_list"])
    t_cal = time.perf_counter() - t0
    st = out["activation_state"]
    gw = st.get("gw")
    has_gw = isinstance(gw, torch.Tensor)
    st_mb = (sum(v.numel() * v.element_size() for v in st.values()
                 if isinstance(v, torch.Tensor)) / 2 ** 20) if has_gw else -1
    diffs = []
    times = []
    for pair in g2["test_activation_list"]:
        t0 = time.perf_counter()
        a = probe.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
        ta = time.perf_counter() - t0
        b = noref.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
        nd = int((a["mant"] != b["mant"]).sum()) + int((a["sign"] != b["sign"]).sum())
        diffs.append(nd)
        times.append(ta)
    total = a["mant"].numel() + a["sign"].numel()
    print(f"[{name}] C={C} N={N} cal={t_cal:.1f}s gw_carried={has_gw} "
          f"state={st_mb:.0f}MiB dyn_t={[f'{t:.2f}' for t in times]} "
          f"elem_diff_vs_norefine={diffs} / {total}")
    return has_gw, diffs


ok = True
gw, diffs = run_case("small(refine)", 1024, 1024, (10, 128, 512), (128, 512),
                     0.8, 0.002, 0.9)
ok &= gw and sum(diffs) > 0
gw, diffs = run_case("mid(carry-only)", 4096, 2048, (10, 128, 512), (128, 512),
                     0.5, 0.0, 0.3)
ok &= gw and sum(diffs) == 0
print("BAND CHECK:", "PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
