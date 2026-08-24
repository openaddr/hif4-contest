"""Pre-measurement: is attention Q/K per-channel ENERGY persistent
(test vs calib)?  The free-form smoothing class transfers only if the
per-(head, channel) energy statistics persist across samples.

Metrics per (kvh, dh) channel map:
  corr_q   : Pearson corr of log q-channel-energy, calib[:-1] pool vs test pool
  corr_k   : same for k
  corr_r   : corr of log(B/A) ratio  (the quantity s_c ~ (B/A)^(1/4) eats)
  also cross-sample WITHIN calib (calib[0:2] vs calib[2:4]) as sanity.

Data: mini attn.pt + synthetic shared-gains attention groups (3+ shapes).
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "dev"))

E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _dq(pair):
    import synth  # lazy; has its own loader
    return synth.deq(pair)


def chan_energy(x, qh, kvh, dh, role):
    """Per (kv-head, channel) energy pooled over tokens (and q-heads for q)."""
    T = x.shape[0]
    if role == "q":
        xe = (x.view(T, qh, dh) ** 2)
        rep = qh // kvh
        xe = xe.view(T, kvh, rep, dh).sum(dim=2)      # sum over group q-heads
    else:
        xe = (x.view(T, kvh, dh) ** 2)
    return xe.sum(dim=0)                               # (kvh, dh)


def pooled_energy(samples, qh, kvh, dh):
    eq = torch.zeros(kvh, dh)
    ek = torch.zeros(kvh, dh)
    n = 0.0
    for smp in samples:
        q = _dq(smp["q"]).float()
        k = _dq(smp["k"]).float()
        T = q.shape[0]
        eq += chan_energy(q, qh, kvh, dh, "q")
        ek += chan_energy(k, qh, kvh, dh, "k")
        n += T
    return eq / max(n, 1.0), ek / max(n, 1.0)


def corr(a, b):
    a = a.flatten().log()
    b = b.flatten().log()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-30)
    return float((a * b).sum() / denom)


def report(tag, samples_a, samples_b, qh, kvh, dh):
    eq_a, ek_a = pooled_energy(samples_a, qh, kvh, dh)
    eq_b, ek_b = pooled_energy(samples_b, qh, kvh, dh)
    r_q = corr(eq_a, eq_b)
    r_k = corr(ek_a, ek_b)
    r_ratio = corr(ek_a / eq_a.clamp_min(1e-30), ek_b / eq_b.clamp_min(1e-30))
    # spread diagnostics
    ls_q = float(eq_a.flatten().log().std())
    ls_k = float(ek_a.flatten().log().std())
    ls_r = float((ek_a / eq_a.clamp_min(1e-30)).flatten().log().std())
    rec = {"tag": tag, "corr_q": round(r_q, 3), "corr_k": round(r_k, 3),
           "corr_ratio": round(r_ratio, 3),
           "logstd_q": round(ls_q, 3), "logstd_k": round(ls_k, 3),
           "logstd_ratio": round(ls_r, 3)}
    print(json.dumps(rec), flush=True)
    return rec


# ---------------- synthetic shared-gains attention generator ----------------

def _gains(C, spread, gen):
    return torch.exp((torch.rand(C, generator=gen) - 0.5) * 2 * math.log(10.0) * spread)


def _nvfp4_pair(x):
    T, C = x.shape
    xb = x.reshape(-1, 16)
    amax = xb.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    scale = ((amax / 6.0).to(torch.bfloat16).float()).clamp_min(1e-30)
    q = xb / scale
    idx = torch.bucketize(q.abs(), (E2M1_GRID[1:] + E2M1_GRID[:-1]) / 2.0)
    carrier = torch.sign(q) * E2M1_GRID[idx]
    return (carrier.reshape(T, -1).to(torch.bfloat16),
            scale.reshape(T, -1).to(torch.bfloat16))


def make_shared_attn(seed, qh, kvh, dh, calib_T=(128, 512, 512, 1024),
                     test_T=(128, 512, 1024), q_spread=0.5, k_spread=0.4,
                     share=1.0, outlier_p=0.0):
    """share=1: per-channel gains drawn ONCE, all calib+test share (mini-like).
    share=0: fresh gains per sample (stock iid)."""
    gen = torch.Generator().manual_seed(seed)
    qc, kc = qh * dh, kvh * dh
    gq = _gains(qc, q_spread, gen)
    gk = _gains(kc, k_spread, gen)
    gq2 = _gains(qc, q_spread, gen)
    gk2 = _gains(kc, k_spread, gen)

    def mix(g, g2):
        if share >= 1.0:
            return g
        if share <= 0.0:
            return None
        lg = share * (g.log() - g.log().mean()) + math.sqrt(1 - share ** 2) * (
            g2.log() - g2.log().mean())
        return (lg + g.log().mean()).exp()

    gq_t, gk_t = mix(gq, gq2), mix(gk, gk2)

    def sample(T, gains_q, gains_k):
        gq_ = gains_q if gains_q is not None else _gains(qc, q_spread, gen)
        gk_ = gains_k if gains_k is not None else _gains(kc, k_spread, gen)
        xq = torch.randn(T, 1, generator=gen) * gq_.unsqueeze(0) * torch.randn(T, qc, generator=gen)
        xk = torch.randn(T, 1, generator=gen) * gk_.unsqueeze(0) * torch.randn(T, kc, generator=gen)
        xv = torch.randn(T, 1, generator=gen) * torch.randn(T, kc, generator=gen)
        if outlier_p > 0:
            for x in (xq, xk):
                m = torch.rand(T, x.shape[1], generator=gen) < outlier_p
                x += m.float() * torch.randn(T, x.shape[1], generator=gen) * x.abs().amax() * 3
        return {"q": _nvfp4_pair(xq), "k": _nvfp4_pair(xk), "v": _nvfp4_pair(xv)}

    return {
        "q_num_heads": qh, "kv_num_heads": kvh, "head_dim": dh,
        # share<=0: TRUE iid -- every sample (calib included) draws fresh gains.
        # (The adversarial "calib structured / test fresh" regime is built by
        # share=1.0 calib + share=0.0 test: see battery config _cs_tf.)
        "calib": [sample(T, gq if share > 0.0 else None,
                         gk if share > 0.0 else None) for T in calib_T],
        "test": [sample(T, gq_t, gk_t) for T in test_T],
    }


SYN_SHAPES = [
    ("gqa32x8x64",  32, 8, 64),
    ("gqa16x2x256", 16, 2, 256),     # mini shape
    ("mha8x8x128",   8, 8, 128),
    ("gqa28x4x128", 28, 4, 128),
]

if __name__ == "__main__":
    out = []
    # ---- mini ----
    mini = torch.load(os.path.join(ROOT, "example", "mini_sample", "attn.pt"),
                      weights_only=True, map_location="cpu")[0]
    qh, kvh, dh = mini["q_num_heads"], mini["kv_num_heads"], mini["head_dim"]
    cal, tst = mini["calib"], mini["test"]
    out.append(report("mini_cal2_vs_cal24", cal[:2], cal[2:4], qh, kvh, dh))
    out.append(report("mini_cal_vs_test", cal[:-1], tst, qh, kvh, dh))
    out.append(report("mini_cal4_vs_calLast", cal[:4], cal[4:], qh, kvh, dh))

    # ---- synthetic shared ----
    for name, qh_, kvh_, dh_ in SYN_SHAPES:
        for share in (1.0, 0.7, 0.0):
            grp = make_shared_attn(4242 + sum(map(ord, name)), qh_, kvh_, dh_,
                                   share=share)
            out.append(report(f"syn_{name}_share{share}",
                              grp["calib"][:-1], grp["test"], qh_, kvh_, dh_))
    with open(os.path.join(HERE, "persist.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("DONE")
