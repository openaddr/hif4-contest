"""Summarize dev/smattn/results.jsonl into the double-holdout B-side table."""
from __future__ import annotations

import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))

rows = [json.loads(l) for l in open(os.path.join(HERE, "results.jsonl"))]
errs = [r for r in rows if "error" in r]
rows = [r for r in rows if "error" not in r]
cfgs = []
for r in rows:
    if r["cfg"] not in cfgs:
        cfgs.append(r["cfg"])

print(f"{'config':24s} {'arm':9s} {'n':>2s} {'mean':>8s} {'med':>8s} {'p10':>8s} "
      f"{'min':>8s} {'neg%':>5s} {'acc%':>5s}")
summary = {}
for cfg in cfgs:
    arms = []
    for r in rows:
        if r["cfg"] == cfg and r["arm"] not in arms:
            arms.append(r["arm"])
    for arm in arms:
        sel = [r for r in rows if r["cfg"] == cfg and r["arm"] == arm]
        d = [r["delta_pp"] for r in sel]
        acc = [1.0 if r["dbg"].get("accepted") else 0.0 for r in sel]
        if not d:
            continue
        neg = 100.0 * sum(1 for x in d if x < -0.05) / len(d)
        p10 = sorted(d)[max(0, int(0.1 * len(d)) - (1 if len(d) > 1 else 0))]
        print(f"{cfg:24s} {arm:9s} {len(d):2d} {st.mean(d):+8.2f} "
              f"{st.median(d):+8.2f} {p10:+8.2f} {min(d):+8.2f} "
              f"{neg:5.0f} {100*st.mean(acc):5.0f}")
        summary.setdefault(cfg, {})[arm] = {
            "n": len(d), "mean": round(st.mean(d), 2),
            "med": round(st.median(d), 2), "p10": round(p10, 2),
            "min": round(min(d), 2), "neg_pct": round(neg, 1),
            "acc_pct": round(100 * st.mean(acc), 0)}
# SHIP-line rollup: structured regimes (exclude iid + cs_tf adversarial)
ship_cfgs = [c for c in cfgs if not c.endswith("_iid") and not c.endswith("_cs_tf")]
d_all, d_neg, n_acc, n_tot = [], 0, 0, 0
for cfg in ship_cfgs:
    for r in rows:
        if r["cfg"] == cfg and r["arm"] == "pre":
            d_all.append(r["delta_pp"])
            d_neg += r["delta_pp"] < -0.05
            n_acc += 1 if r["dbg"].get("accepted") else 0
            n_tot += 1
print(f"\nSHIP rollup (structured configs, arm=pre): n={len(d_all)} "
      f"mean={st.mean(d_all):+.2f} med={st.median(d_all):+.2f} "
      f"min={min(d_all):+.2f} neg={100*d_neg/len(d_all):.0f}% "
      f"acc={100*n_acc/n_tot:.0f}%")
if errs:
    print("errors:", len(errs))
with open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8") as fh:
    json.dump({"per_config": summary,
               "ship_rollup": {"n": len(d_all), "mean": round(st.mean(d_all), 2),
                               "neg_pct": round(100 * d_neg / len(d_all), 1),
                               "acc_pct": round(100 * n_acc / n_tot, 0)}}, fh, indent=1)
