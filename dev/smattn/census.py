"""Secondary task: ff_bal (linear-side free-form smoothing) acceptance census
across the 15 stress_data synthetic configs.  Answers: was the judge's +207
with only ~21% mini-predicted acceptance limited by the guard rejecting
(structure-poor groups) or by weak per-group value?

For each stress config: calibrate dev/smooth/solution.py (SMOOTH_MODE ff_bal)
on its calib list; report guard accept/reject, j_base/j_cand, s stats; and
the real test-side delta vs SMOOTH_MODE=base.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "dev"))
import hif4  # noqa: E402
import variants as V  # noqa: E402

SMOOTH_DIR = os.path.join(ROOT, "dev", "smooth")
spec = importlib.util.spec_from_file_location(
    "_smooth_sol", os.path.join(SMOOTH_DIR, "solution.py"))
SOL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SOL)

STRESS = os.path.join(ROOT, "dev", "stress_data")


def score(group, arm):
    SOL.SMOOTH_MODE = "base" if arm == "base" else arm
    SOL.SMOOTH_GUARD = True
    SOL.SMOOTH_DEBUG.clear()
    torch.manual_seed(0)
    cal = SOL.hif4_calibration_and_quantize_weight(
        *group["weight"], group["calib_activation_list"])
    w_ref = hif4.dequantize_nvfp4(*group["weight"])
    w_std = V.deq(V.quant_alg1(w_ref.float()))
    w_play = hif4.hif4_dequantize(cal["weight_params"])
    rows = []
    for pair in group["test_activation_list"]:
        x_ref = hif4.dequantize_nvfp4(*pair)
        ref = hif4.linear_ref(x_ref, w_ref)
        x_std = V.deq(V.quant_alg1(x_ref.float()))
        mse_std = ((hif4.linear_ref(x_std, w_std) - ref) ** 2).mean().item()
        p = SOL.hif4_dynamic_quantize_activation(pair[0], pair[1],
                                                 cal["activation_state"])
        mse_play = ((hif4.linear_ref(hif4.hif4_dequantize(p), w_play) - ref) ** 2).mean().item()
        rows.append((mse_std - mse_play) / mse_std * 100.0)
    s = cal["activation_state"].get("s")
    sstats = None
    if s is not None:
        sstats = [round(float(x), 3) for x in (s.min(), s.max(), s.log().std())]
    return sum(rows) / len(rows), dict(SOL.SMOOTH_DEBUG), sstats, cal["activation_state"].get("mode")


def main():
    names = sorted(os.listdir(STRESS))
    out = []
    for name in names:
        path = os.path.join(STRESS, name, "linear.pt")
        if not os.path.exists(path):
            continue
        group = torch.load(path, weights_only=True, map_location="cpu")[0]
        try:
            m_base, dbg_b, _, mode = score(group, "base")
            m_ff, dbg_f, sstats, mode = score(group, "ff_bal")
        except Exception as exc:
            rec = {"cfg": name, "error": repr(exc)}
            print(json.dumps(rec), flush=True)
            out.append(rec)
            continue
        rec = {"cfg": name, "accept": dbg_f.get("accepted"),
               "j": [dbg_f.get("j_base"), dbg_f.get("j_cand")],
               "s_stats": sstats, "mode": mode,
               "base_pp": round(m_base, 3), "ff_pp": round(m_ff, 3),
               "delta_pp": round(m_ff - m_base, 3)}
        print(json.dumps(rec), flush=True)
        out.append(rec)
    with open(os.path.join(HERE, "census.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    acc = sum(1 for r in out if r.get("accept"))
    print(f"ACCEPT {acc}/{len(out)}")


if __name__ == "__main__":
    main()
