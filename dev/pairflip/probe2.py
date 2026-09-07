"""probe2: cooperative pair-flip upper bound on shared-structure synthetic
groups (dev/smooth/exp_smooth.py generator; judge-like structured regime).

Groups: c2048_shared (mini-shape twin), c1024_shared (small-C), and the
outlier variant c2048_shared_outl -- one seed each, ship arms identical to
probe1 (scan@ship, ship+pair1, deep, deep+pair).

Usage: python dev/pairflip/probe2.py [--out results_probe2.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "dev"))
sys.path.insert(0, os.path.join(ROOT, "dev", "smooth"))
sys.path.insert(0, HERE)

import pf                    # noqa: E402
import exp_smooth as E       # noqa: E402

torch.set_num_threads(max(1, os.cpu_count() - 2))

GROUPS = [
    ("c2048_shared", dict(N=2048, C=2048, spread=0.5, outlier_p=0.0,
                          w_spread=0.3, share=1.0)),
    ("c1024_shared", dict(N=2048, C=1024, spread=0.5, outlier_p=0.0,
                          w_spread=0.3, share=1.0)),
    ("c2048_shared_outl", dict(N=2048, C=2048, spread=0.5, outlier_p=0.002,
                               w_spread=0.3, share=1.0)),
]


def summarise_scan(st):
    rb = st["row_best"]
    neg = rb[torch.isfinite(rb)]
    return {
        "rows_A_neg": st["rows_A_neg"],
        "cand_rows": st["cand_rows"],
        "coop_rows": st["coop_rows"],
        "Bneg_pairs": st["Bneg_pairs"],
        "applied_rows": st["applied_rows"],
        "sum_dJ": st["sum_dJ"],
        "coop_dJ_p50": float(neg.median()) if neg.numel() else None,
        "coop_dJ_min": float(neg.min()) if neg.numel() else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results_probe2.json"))
    args = ap.parse_args()

    SOL = pf.load_sol(name="_pairflip_sol2")
    out = []
    for gname, kw in GROUPS:
        seed = 5100 + 131 * 0 + (sum(map(ord, gname)) % 977)
        torch.manual_seed(0)
        group = E.make_shared_group(seed, **kw)
        Wp = group["weight"]
        N = Wp[0].shape[0]
        torch.manual_seed(0)
        t0 = time.perf_counter()
        cal = SOL.hif4_calibration_and_quantize_weight(
            *Wp, group["calib_activation_list"])
        t_cal = time.perf_counter() - t0
        w_ref = pf.hif4.dequantize_nvfp4(*Wp)
        w_play = pf.hif4.hif4_dequantize(cal["weight_params"])
        st_state = cal["activation_state"]
        print(f"== {gname} seed={seed} cal={t_cal:.1f}s "
              f"mode={st_state.get('mode')} g={st_state.get('g')} "
              f"gw={st_state['gw'].dtype}", flush=True)

        for i, pair in enumerate(group["test_activation_list"]):
            x_ref, ref, mse_std = pf.case_mse_std(w_ref, pair)
            ctx = pf.make_ctx(SOL, pair, st_state)
            T, C = ctx["T"], ctx["C"]
            rec = {"group": gname, "seed": seed, "case": i, "T": T, "C": C,
                   "N": N, "mse_std": mse_std}
            pp_ship, _ = pf.pp_of_params(ctx["p_ship"], w_play, ref, mse_std)
            rec["pp_ship"] = round(pp_ship, 3)

            M, A = pf.single_stats(ctx)
            t0 = time.perf_counter()
            st0 = pf.pair_scan(ctx, M, A)
            rec["scan_at_ship"] = summarise_scan(st0)
            rec["scan_at_ship"]["scan_s"] = round(time.perf_counter() - t0, 2)
            rec["amin_p50"] = float(st0["row_amin"].median())

            v4_ship = ctx["v4"].clone()
            if st0["applied_rows"] > 0:
                pf.apply_picks(ctx, M, st0["picks"])
                p_s1 = SOL._values_to_params(ctx["v4"] * ctx["d"],
                                             ctx["p_base"])
                pp_s1, _ = pf.pp_of_params(p_s1, w_play, ref, mse_std)
            else:
                pp_s1 = pp_ship
            rec["pp_ship_pair1"] = round(pp_s1, 3)
            rec["pred_dpp_ship_pair1"] = round(
                (-st0["sum_dJ"]) / (T * N) / mse_std * 100.0, 4)
            ctx["v4"] = v4_ship

            t0 = time.perf_counter()
            rs, ts, conv = pf.reconverge_singles(SOL, ctx, max_sweeps=256)
            rec["deep_sweeps"] = rs
            rec["deep_s"] = round(time.perf_counter() - t0, 2)
            rec["deep_converged"] = conv
            p_deep = SOL._values_to_params(ctx["v4"] * ctx["d"], ctx["p_base"])
            pp_deep, _ = pf.pp_of_params(p_deep, w_play, ref, mse_std)
            rec["pp_deep"] = round(pp_deep, 3)

            t0 = time.perf_counter()
            stats, tps, dJ_pairs, dJ_singles = pf.pair_machine(SOL, ctx)
            rec["pair_sweeps"] = len(stats)
            rec["pair_s"] = round(time.perf_counter() - t0, 2)
            rec["pair_total_dJ"] = dJ_pairs
            rec["pair_unlocked_singles_dJ"] = dJ_singles
            p_final = SOL._values_to_params(ctx["v4"] * ctx["d"],
                                            ctx["p_base"])
            pp_dp, _ = pf.pp_of_params(p_final, w_play, ref, mse_std)
            rec["pp_deep_pair"] = round(pp_dp, 3)
            rec["pred_dpp_pair"] = round(
                (-(dJ_pairs + dJ_singles)) / (T * N) / mse_std * 100.0, 4)

            out.append(rec)
            print(f"[{gname} case {i} T={T}] ship={pp_ship:.2f} "
                  f"ship+p1={pp_s1:.2f}({rec['pred_dpp_ship_pair1']:+.4f}) "
                  f"deep={pp_deep:.2f} deep+pair={pp_dp:.2f}"
                  f"({rec['pred_dpp_pair']:+.4f}) "
                  f"coop@ship={st0['coop_rows']}/{T} "
                  f"A_neg@ship={st0['rows_A_neg']}/{T}", flush=True)
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
