"""Analyze battery results.jsonl -> B-side distribution tables per config/arm."""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results.jsonl")
rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
errs = [r for r in rows if "error" in r]
rows = [r for r in rows if "error" not in r]

# pair base mean with arm deltas per (cfg, seed)
runs = defaultdict(dict)
for r in rows:
    runs[(r["cfg"], r["seed"])][r["arm"]] = r

stats = defaultdict(lambda: defaultdict(list))
accs = defaultdict(lambda: defaultdict(list))
tcals = defaultdict(lambda: defaultdict(list))
for (cfg, seed), arms in runs.items():
    if "base" not in arms:
        continue
    for arm, r in arms.items():
        stats[cfg][arm].append(r["mean"] - arms["base"]["mean"])
        accs[cfg][arm].append(1 if r["dbg"].get("accepted") else 0)
        tcals[cfg][arm].append(r["t_cal"] - arms["base"]["t_cal"])

def q(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(p * len(s)))]

print(f"{'config':<22}{'arm':<10}{'n':>3}{'mean':>8}{'med':>8}{'p10':>8}{'neg%':>6}{'acc%':>6}{'dcal':>7}")
for cfg in stats:
    for arm in ("ff_icm", "ff_bal", "mag_scan", "ff_icm_ng"):
        if arm not in stats[cfg]:
            continue
        v = stats[cfg][arm]
        neg = sum(1 for x in v if x < -0.05) / len(v) * 100
        ac = sum(accs[cfg][arm]) / len(accs[cfg][arm]) * 100
        dc = sum(tcals[cfg][arm]) / len(tcals[cfg][arm])
        print(f"{cfg:<22}{arm:<10}{len(v):>3}{sum(v)/len(v):>8.2f}{q(v,0.5):>8.2f}"
              f"{q(v,0.1):>8.2f}{neg:>6.0f}{ac:>6.0f}{dc:>7.2f}")

# pooled per-arm across shared-structure configs (excl iid)
shared = [c for c in stats if "iid" not in c]
print("\nPooled over structure-sharing configs:")
for arm in ("ff_icm", "ff_bal", "mag_scan", "ff_icm_ng"):
    v = [x for c in shared if arm in stats[c] for x in stats[c][arm]]
    if not v:
        continue
    neg = sum(1 for x in v if x < -0.05) / len(v) * 100
    print(f"  {arm:<10} n={len(v)} mean={sum(v)/len(v):+.2f} med={q(v,0.5):+.2f} "
          f"p10={q(v,0.1):+.2f} min={min(v):+.2f} neg%={neg*1:.0f}")
if errs:
    print(f"\nERRORS: {len(errs)}")
    for e in errs[:5]:
        print(" ", e)
