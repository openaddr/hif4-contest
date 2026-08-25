"""msel step-1 analysis: J-gap distribution tables from results/step1.json."""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    path = os.path.join(HERE, "results", "step1.json")
    with open(path) as f:
        out = json.load(f)
    groups = [k for k in sorted(out) if k != "smoke_c1024"]
    VNAMES = ("smallT", "mag")
    print(f"{'group':<22} {'md':>2} {'g':>1} {'gr':>2} {'tmx':>4} "
          f"{'rms lr (smallT/mag)':>20} {'guard':>6}")
    for g in groups:
        e = out[g]
        sr = e["s_ratio"]
        s1 = sr.get("smallT", {}).get("rms_log_ratio", float("nan"))
        s2 = sr.get("mag", {}).get("rms_log_ratio", float("nan"))
        acc = e.get("smooth_dbg", {}).get("accepted")
        print(f"{g:<22} {e['mode']:>2} {e['g']:>1} {e['grams']:>2} {e['tmax']:>4} "
              f"{s1:>9.3f}/{s2:>8.3f} {str(acc):>6}")

    # ---- pooled J-gap distribution per variant ----
    print("\n=== rel J_true (relative to |J_def| ~ signal power), per variant ===")
    print(f"{'variant':<8} {'n':>3} {'wins<−1e−4':>10} {'median':>10} {'p10':>10} "
          f"{'worst(best)':>12}")
    for vn in VNAMES:
        rels = []
        wins = 0
        for g in groups:
            for row in out[g]["calls"]:
                r = row["variants"].get(vn, {}).get("rel_j_true")
                if r is not None:
                    rels.append(r)
                    if r < -1e-4:
                        wins += 1
        rels.sort()
        n = len(rels)
        med = rels[n // 2] if n else float("nan")
        p10 = rels[max(0, n // 10 - 1)] if n else float("nan")
        print(f"{vn:<8} {n:>3} {wins:>10} {med:>+10.3e} {p10:>+10.3e} "
              f"{rels[0]:>+12.3e}")

    # ---- relative to default MSE (pp-relevant scale) ----
    print("\n=== rel_mse (variant MSE / default MSE − 1), per variant ===")
    for vn in VNAMES:
        rels = []
        for g in groups:
            for row in out[g]["calls"]:
                r = row["variants"].get(vn, {}).get("rel_mse")
                if r is not None:
                    rels.append(r)
        rels.sort()
        n = len(rels)
        neg = sum(1 for r in rels if r < 0)
        print(f"{vn}: n={n} better-than-default={neg}  median={rels[n//2]:+.3f} "
              f"p10={rels[max(0,n//10-1)]:+.3f} best={rels[0]:+.3f} worst={rels[-1]:+.3f}")

    # ---- perfect-variant floor: does even zero-quant-error lose? ----
    print("\n=== j_perf (variant's EXACT transformed input vs ship target) ===")
    for vn in VNAMES:
        rels = []
        for g in groups:
            for row in out[g]["calls"]:
                r = row["variants"].get(vn, {}).get("rel_j_perf")
                if r is not None:
                    rels.append(r)
        n = len(rels)
        worse = sum(1 for r in rels if r > 1e-4)
        bestv = min(rels) if rels else float("nan")
        print(f"{vn}: n={n} floor-worse-than-default={worse} best floor rel={bestv:+.3e}")

    # ---- literal rule (wrong cross-term) divergence damage ----
    print("\n=== literal rule (J with x_v cross term -- the proposal as written) ===")
    div = 0
    tot = 0
    dmg = []
    for g in groups:
        for row in out[g]["calls"]:
            per = row["variants"]
            tot += 1
            lit_best = min(((v.get("rel_j_lit") if v.get("rel_j_lit") is not None else 0.0), k)
                           for k, v in per.items())
            if lit_best[0] < -1e-4:
                div += 1
                dmg.append((lit_best[1], per[lit_best[1]]["rel_j_true"]))
    print(f"calls={tot} literal-divergences={div} "
          f"({100.0*div/max(tot,1):.1f}%); when diverged, true rel J of chosen variant:")
    if dmg:
        vals = sorted(d for _, d in dmg)
        print(f"  median={vals[len(vals)//2]:+.3e} best={vals[0]:+.3e} worst={vals[-1]:+.3e} "
              f"(positive = selection HURTS the judge MSE)")

    # ---- T-dependence: do small-T calls prefer the small-T fit? ----
    print("\n=== per-T median rel J_true (smallT variant) ===")
    byT = {}
    for g in groups:
        for row in out[g]["calls"]:
            r = row["variants"].get("smallT", {}).get("rel_j_true")
            if r is not None:
                byT.setdefault(row["T"], []).append(r)
    for T in sorted(byT):
        v = sorted(byT[T])
        print(f"T={T:>5}: n={len(v):>2} median={v[len(v)//2]:+.3e} min={v[0]:+.3e} "
              f"max={v[-1]:+.3e} wins={sum(1 for x in v if x < -1e-4)}")

    # ---- decision gate ----
    tot_calls = 0
    wins = 0
    for g in groups:
        for row in out[g]["calls"]:
            tot_calls += 1
            best = None
            for vn, ev in row["variants"].items():
                r = ev.get("rel_j_true")
                if r is not None and (best is None or r < best):
                    best = r
            if best is not None and best < -1e-4:
                wins += 1
    frac_same = 100.0 * (tot_calls - wins) / max(tot_calls, 1)
    print(f"\nDECISION GATE: calls={tot_calls} variant-wins={wins} "
          f"best==default on {frac_same:.1f}% of calls "
          f"({'NO-SHIP' if frac_same > 70 else 'measure further'})")


if __name__ == "__main__":
    main()
