"""probe1: cooperative pair-flip upper bound on mini REAL test (linear).

Arms per test case (all offline, ship module = dev/pairflip/solution.py):
  ship        pp of the unmodified v42 dynamic path
  (scan@ship) convergence margins + cooperative-pair stats at the ship point
  deep        greedy singles continued to full convergence (fp32 grams)
  deep+pair   pair_machine from the deep point (iterated 2-opt + single
              re-convergence until exhaustion)
  ship+pair   pair_machine directly from the ship point (no deepening)

Cross-check: predicted dpp from exact dJ (dJ_sum / (T*N) / mse_std * 100)
against the actual end-to-end rescoring of the modified grid.

Usage: python dev/pairflip/probe1.py [--out results_probe1.json] [--quick]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import torch

import pf

ROOT = pf.ROOT
torch.set_num_threads(max(1, os.cpu_count() - 2))


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
        "coop_dJ_p90": float(neg.quantile(0.1)) if neg.numel() else None,
        "coop_dJ_min": float(neg.min()) if neg.numel() else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(pf.HERE, "results_probe1.json"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    SOL = pf.load_sol()
    mini = torch.load(os.path.join(ROOT, "example", "mini_sample", "linear.pt"),
                      weights_only=True, map_location="cpu")[0]
    W, CAL, TST = (mini["weight"], mini["calib_activation_list"],
                   mini["test_activation_list"])
    N = W[0].shape[0]

    cache = os.path.join(pf.HERE, "cache", "mini_cal.pt")
    if os.path.exists(cache):
        cal = torch.load(cache, weights_only=True)
        print("cal: loaded cache", flush=True)
    else:
        torch.manual_seed(0)
        t0 = time.perf_counter()
        cal = SOL.hif4_calibration_and_quantize_weight(*W, CAL)
        print(f"cal: fresh {time.perf_counter()-t0:.1f}s", flush=True)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save({"weight_params": cal["weight_params"],
                    "activation_state": cal["activation_state"]}, cache)

    w_ref = pf.hif4.dequantize_nvfp4(*W)
    w_play = pf.hif4.hif4_dequantize(cal["weight_params"])
    st_state = cal["activation_state"]

    out = []
    for i, pair in enumerate(TST):
        x_ref, ref, mse_std = pf.case_mse_std(w_ref, pair)
        ctx = pf.make_ctx(SOL, pair, st_state)
        T, C = ctx["T"], ctx["C"]
        rec = {"case": i, "T": T, "C": C, "N": N, "mse_std": mse_std}

        pp_ship, mse_ship = pf.pp_of_params(ctx["p_ship"], w_play, ref, mse_std)
        rec["pp_ship"] = round(pp_ship, 3)

        # scan at the ship point
        t0 = time.perf_counter()
        M, A = pf.single_stats(ctx)
        st0 = pf.pair_scan(ctx, M, A)
        st0["scan_s"] = round(time.perf_counter() - t0, 2)
        rec["scan_at_ship"] = summarise_scan(st0)
        rec["scan_at_ship"]["scan_s"] = st0["scan_s"]
        rec["amin_p50"] = float(st0["row_amin"].median())
        print(f"[case {i} T={T}] ship pp={pp_ship:.2f} "
              f"rows_A_neg={st0['rows_A_neg']}/{T} "
              f"coop_rows@ship={st0['coop_rows']} "
              f"sum_dJ@ship={st0['sum_dJ']:.3e} scan={st0['scan_s']}s", flush=True)

        # ship + ONE pair sweep (append-only variant, ship-tier cost)
        v4_ship = ctx["v4"].clone()
        J_ship = pf.J_raw(ctx)
        if st0["applied_rows"] > 0:
            pf.apply_picks(ctx, M, st0["picks"])
            p_s1 = SOL._values_to_params(ctx["v4"] * ctx["d"], ctx["p_base"])
            pp_s1, _ = pf.pp_of_params(p_s1, w_play, ref, mse_std)
            rec["pp_ship_pair1"] = round(pp_s1, 3)
            rec["pred_dpp_ship_pair1"] = round(
                (-st0["sum_dJ"]) / (T * N) / mse_std * 100.0, 4)
        else:
            rec["pp_ship_pair1"] = pp_ship
            rec["pred_dpp_ship_pair1"] = 0.0
        ctx["v4"] = v4_ship                      # restore ship grid
        rec["J_check_ship"] = pf.J_raw(ctx) - J_ship   # ~0.0 after restore
        print(f"[case {i}] ship+pair1 pp={rec['pp_ship_pair1']:.3f} "
              f"(d={rec['pp_ship_pair1']-pp_ship:+.4f} "
              f"pred={rec['pred_dpp_ship_pair1']:+.4f})", flush=True)

        # deep singles
        J_before_deep = pf.J_raw(ctx)
        t0 = time.perf_counter()
        rs, ts, conv = pf.reconverge_singles(SOL, ctx, max_sweeps=256)
        rec["deep_sweeps"] = rs
        rec["deep_s"] = round(time.perf_counter() - t0, 2)
        rec["deep_converged"] = conv
        rec["deep_dJ"] = pf.J_raw(ctx) - J_before_deep
        p_deep = SOL._values_to_params(ctx["v4"] * ctx["d"], ctx["p_base"])
        pp_deep, mse_deep = pf.pp_of_params(p_deep, w_play, ref, mse_std)
        rec["pp_deep"] = round(pp_deep, 3)
        rec["pred_dpp_deep"] = round(
            (-rec["deep_dJ"]) / (T * N) / mse_std * 100.0, 4)
        print(f"[case {i}] deep sweeps={rs} ({rec['deep_s']}s, conv={conv}) "
              f"pp={pp_deep:.2f} (d={pp_deep-pp_ship:+.3f} "
              f"pred={rec['pred_dpp_deep']:+.4f})", flush=True)

        # pair machine from the deep point
        t0 = time.perf_counter()
        stats, ts, dJ_pairs, dJ_singles = pf.pair_machine(SOL, ctx, verbose=True)
        rec["pair_stats"] = [summarise_scan(s) for s in stats]
        rec["pair_s"] = round(time.perf_counter() - t0, 2)
        rec["pair_total_dJ"] = dJ_pairs
        rec["pair_unlocked_singles_dJ"] = dJ_singles
        p_final = SOL._values_to_params(ctx["v4"] * ctx["d"], ctx["p_base"])
        pp_dp, mse_dp = pf.pp_of_params(p_final, w_play, ref, mse_std)
        rec["pp_deep_pair"] = round(pp_dp, 3)
        dJ_gain = -(dJ_pairs + dJ_singles)
        rec["pred_dpp_pair"] = round(dJ_gain / (T * N) / mse_std * 100.0, 4)
        print(f"[case {i}] deep+pair pp={pp_dp:.2f} "
              f"(d vs deep {pp_dp-pp_deep:+.3f}, vs ship {pp_dp-pp_ship:+.3f}) "
              f"pred={rec['pred_dpp_pair']:+.4f} time={rec['pair_s']}s", flush=True)

        out.append(rec)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)

    means = {k: round(sum(r[k] for r in out) / len(out), 4)
             for k in ("pp_ship", "pp_ship_pair1", "pp_deep", "pp_deep_pair")}
    print("MEAN", json.dumps(means), flush=True)


if __name__ == "__main__":
    main()
