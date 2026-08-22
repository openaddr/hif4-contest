"""roundopt/analyze: validate row-independence + measure active-set decay.

For every captured refine call:
  A. replica (core.refine_ship) must reproduce the ship module output
     bit-exactly (torch.equal on v4*d) -- guards the replica itself.
  B. row-freeze monotonicity: once a row has dr==0 in a round, all later
     rounds must also have dr==0 for that row (the premise of the active-set
     loop).  Asserted on the torch path.
  C. decay curve: |flipping rows| per round -> where the time actually goes.
  D. active twin (np_thresh=0) must match the ship flip trace exactly and
     v4*d bit-exactly.

Usage: python dev/roundopt/analyze.py [--groups N] [--limit k]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core  # noqa: E402

CAP = os.path.join(HERE, "capture")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    S = core.load_sol()
    files = sorted(glob.glob(os.path.join(CAP, "*.pt")))
    if args.groups:
        files = files[:args.groups]
    if args.limit:
        files = files[:args.limit]
    assert files, "no captures; run capture.py first"
    tot_calls = 0
    decay_sum = {}      # round -> total active fraction (torch path)
    n_torch_calls = 0
    tier = {}           # T -> [sum_active_over_rounds, rounds, T, n]
    for f in files:
        d = torch.load(f, weights_only=False)
        gw = d["gw"].float()
        gwf = d["gwf"].float()
        for i, rec in enumerate(d["recs"]):
            tr = []
            cv = []
            out = core.refine_ship(rec["x"], rec["v0"], rec["unit"], gw, gwf,
                                   S=S, trace=tr, curve=cv)
            assert torch.equal(out, rec["v1_ref"]), \
                f"replica mismatch {os.path.basename(f)}#{i}"
            tot_calls += 1
            # B: freeze monotonicity per row (dr==0 -> stays 0): the flip
            # rounds of each row must be a PREFIX of 0..K
            flips_per_row_per_round = {}
            for (rnd, r, c, dv) in tr:
                flips_per_row_per_round.setdefault(r, set()).add(rnd)
            T = rec["T"]
            if T > 32:  # torch path: full curve recorded every round
                for r, rnds in flips_per_row_per_round.items():
                    assert max(rnds) == len(rnds) - 1, \
                        f"row {r} not prefix-frozen: {sorted(rnds)}"
                for k, a in enumerate(cv):
                    decay_sum[k] = decay_sum.get(k, 0) + a
                n_torch_calls += 1
                e = tier.setdefault(T, [0, 0, 0, 0])
                e[0] += sum(cv)
                e[1] = len(cv)
                e[2] = T
                e[3] += 1
            # D: active twin identity (v1 = exact-op compaction; v2 =
            # pass-reduced round; v2n = v2 + numpy tail at A<=32)
            for tag, kw in (("v1", {}),
                            ("v2", {"v2": True}),
                            ("v2n", {"v2": True, "np_thresh": 32})):
                tr2 = []
                out2 = core.refine_active(rec["x"], rec["v0"], rec["unit"],
                                          gw, gwf, S=S, trace=tr2, **kw)
                assert tr == tr2, \
                    f"flip-sequence mismatch {tag} {os.path.basename(f)}#{i}" \
                    f" ({len(tr)} vs {len(tr2)} flips)"
                assert torch.equal(out, out2), \
                    f"active output mismatch {tag} {os.path.basename(f)}#{i}"
        print(f"{os.path.basename(f)}: {len(d['recs'])} calls OK")
    print(f"\nTOTAL {tot_calls} calls: replica==ship, freeze-prefix OK, "
          f"active-twin flip-sequence + output bit-identical")
    if tier:
        print("\nper-tier active-row-rounds (torch path):")
        print(f"{'T':>6} {'rounds':>7} {'calls':>6} {'full=n*T*R':>12} "
              f"{'active':>10} {'frac':>7}")
        for T in sorted(tier):
            s, R, _, n = tier[T]
            print(f"{T:>6} {R:>7} {n:>6} {n * T * R:>12} {s:>10} "
                  f"{s / (n * T * R):>7.1%}")


if __name__ == "__main__":
    main()
