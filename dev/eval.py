"""Local scoring harness: baseline vs solution on mini_sample.

Score per case = (MSE_STD - MSE_PLAYER) / MSE_STD, matching the task book.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import hif4  # noqa: E402


def load_solution(solution_dir):
    spec = importlib.util.spec_from_file_location(
        "solution", os.path.join(solution_dir, "solution.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    sol = load_solution(os.path.join(os.path.dirname(__file__), "..", "example", "solution"))
    root = os.path.join(os.path.dirname(__file__), "..", "example", "mini_sample")

    t0 = time.time()
    total_score = 0.0
    n_cases = 0

    # ---------------- Linear ----------------
    linear = torch.load(os.path.join(root, "linear.pt"), weights_only=True, map_location="cpu")
    for gi, g in enumerate(linear):
        w_ref = hif4.dequantize_nvfp4(*g["weight"])
        t_cal = time.time()
        cal = sol.hif4_calibration_and_quantize_weight(*g["weight"], g["calib_activation_list"])
        t_cal = time.time() - t_cal

        w_std = hif4.hif4_dequantize(hif4.hif4_quantize_standard(w_ref.float()))
        w_play = hif4.hif4_dequantize(cal["weight_params"])

        for ti, pair in enumerate(g["test_activation_list"]):
            x_ref = hif4.dequantize_nvfp4(*pair)
            ref = hif4.linear_ref(x_ref, w_ref)
            mse_std = ((hif4.linear_ref(x_ref, w_std) - ref) ** 2).mean().item()

            t_dyn = time.time()
            p = sol.hif4_dynamic_quantize_activation(pair[0], pair[1], cal["activation_state"])
            t_dyn = time.time() - t_dyn
            x_play = hif4.hif4_dequantize(p)
            mse_play = ((hif4.linear_ref(x_ref, w_play) - ref) ** 2).mean().item()

            s = (mse_std - mse_play) / mse_std
            total_score += s
            n_cases += 1
            print(f"[linear g{gi} t{ti}] MSE_std={mse_std:.6e} MSE_play={mse_play:.6e} "
                  f"score={s:+.4f}  (cal {t_cal:.2f}s dyn {t_dyn*1000:.0f}ms)")

    # ---------------- Attention ----------------
    attn = torch.load(os.path.join(root, "attn.pt"), weights_only=True, map_location="cpu")
    for gi, g in enumerate(attn):
        qh, kvh, dh = g["q_num_heads"], g["kv_num_heads"], g["head_dim"]
        t_cal = time.time()
        cal = sol.hif4_calibration_attention(g["calib"], qh, kvh, dh)
        t_cal = time.time() - t_cal

        for ti, sample in enumerate(g["test"]):
            q_ref = hif4.dequantize_nvfp4(*sample["q"])
            k_ref = hif4.dequantize_nvfp4(*sample["k"])
            v_ref = hif4.dequantize_nvfp4(*sample["v"])
            ref = hif4.attn_ref(q_ref, k_ref, v_ref, qh, kvh, dh)

            q_std = hif4.hif4_dequantize(hif4.hif4_quantize_standard(q_ref.float()))
            k_std = hif4.hif4_dequantize(hif4.hif4_quantize_standard(k_ref.float()))
            v_std = hif4.hif4_dequantize(hif4.hif4_quantize_standard(v_ref.float()))
            mse_std = ((hif4.attn_ref(q_std, k_std, v_std, qh, kvh, dh) - ref) ** 2).mean().item()

            t_dyn = time.time()
            pq = sol.hif4_dynamic_quantize_q(sample["q"][0], sample["q"][1], qh, dh, cal["q_state"])
            pk = sol.hif4_dynamic_quantize_k(sample["k"][0], sample["k"][1], kvh, dh, cal["k_state"])
            pv = sol.hif4_dynamic_quantize_v(sample["v"][0], sample["v"][1], kvh, dh, cal["v_state"])
            t_dyn = time.time() - t_dyn
            out = hif4.attn_ref(
                hif4.hif4_dequantize(pq), hif4.hif4_dequantize(pk), hif4.hif4_dequantize(pv),
                qh, kvh, dh,
            )
            mse_play = ((out - ref) ** 2).mean().item()

            s = (mse_std - mse_play) / mse_std
            total_score += s
            n_cases += 1
            print(f"[attn   g{gi} t{ti}] MSE_std={mse_std:.6e} MSE_play={mse_play:.6e} "
                  f"score={s:+.4f}  (cal {t_cal:.2f}s dyn {t_dyn*1000:.0f}ms)")

    print(f"\nTOTAL score = {total_score:+.4f} over {n_cases} cases "
          f"(elapsed {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
