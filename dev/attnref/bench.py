"""Measurement harness: attention-side lattice refinement (value + cost).

Modes (argv[1]): ident | mini | synth | vcheck | guard
Appends results to dev/attnref/results.json (checkpoint-friendly).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "dev"))
import hif4  # noqa: E402
import variants as V  # noqa: E402
import synth  # noqa: E402

SOLPATH = os.path.join(HERE, "solution.py")
RES = os.path.join(HERE, "results.json")


def load_sol(name="_attnref_sol"):
    spec = importlib.util.spec_from_file_location(name, SOLPATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clone_state(st):
    if isinstance(st, torch.Tensor):
        return st.clone()
    if isinstance(st, dict):
        return {k: clone_state(v) for k, v in st.items()}
    return st


def with_rf(state, rf):
    s2 = dict(state)
    s2["rf"] = rf
    return s2


def state_bytes(st) -> int:
    if isinstance(st, torch.Tensor):
        return st.numel() * st.element_size()
    if isinstance(st, dict):
        return sum(state_bytes(v) for v in st.values())
    return 0


def calibrate(SOL, group, gptq=True, force_h=True, guard=True, qs=0, ks=0):
    SOL.ATTN_GPTQ_ENABLE = gptq
    SOL.ATTN_REFINE_FORCE_H = force_h
    SOL.ATTN_REFINE_GUARD = guard
    SOL.ATTN_REFINE_Q_SWEEPS = qs
    SOL.ATTN_REFINE_K_SWEEPS = ks
    qh, kvh, dh = group["q_num_heads"], group["kv_num_heads"], group["head_dim"]
    torch.manual_seed(0)
    t0 = time.perf_counter()
    cal = SOL.hif4_calibration_attention(group["calib"], qh, kvh, dh)
    return cal, time.perf_counter() - t0


def eval_states(SOL, group, q_state, k_state, v_state, reps=3):
    """Per-case official-style score + per-call timing (median of reps)."""
    qh, kvh, dh = group["q_num_heads"], group["kv_num_heads"], group["head_dim"]
    cases = []
    vals_q = []
    for ti, smp in enumerate(group["test"]):
        q_ref = hif4.dequantize_nvfp4(*smp["q"])
        k_ref = hif4.dequantize_nvfp4(*smp["k"])
        v_ref = hif4.dequantize_nvfp4(*smp["v"])
        ref = hif4.attn_ref(q_ref, k_ref, v_ref, qh, kvh, dh)
        qs_ = V.deq(V.quant_alg1(q_ref.float()))
        ks_ = V.deq(V.quant_alg1(k_ref.float()))
        vs_ = V.deq(V.quant_alg1(v_ref.float()))
        mse_std = ((hif4.attn_ref(qs_, ks_, vs_, qh, kvh, dh) - ref) ** 2).mean().item()
        times = {}
        outs = {}
        for role, fn, st, nh in (
            ("q", SOL.hif4_dynamic_quantize_q, q_state, qh),
            ("k", SOL.hif4_dynamic_quantize_k, k_state, kvh),
            ("v", SOL.hif4_dynamic_quantize_v, v_state, kvh),
        ):
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                outs[role] = fn(smp[role][0], smp[role][1], nh, dh, clone_state(st))
                ts.append(time.perf_counter() - t0)
            times[role] = sorted(ts)[len(ts) // 2]
        out = hif4.attn_ref(hif4.hif4_dequantize(outs["q"]),
                            hif4.hif4_dequantize(outs["k"]),
                            hif4.hif4_dequantize(outs["v"]), qh, kvh, dh)
        mse_play = ((out - ref) ** 2).mean().item()
        cases.append({
            "ti": ti, "T": int(smp["q"][0].shape[0]),
            "mse_std": mse_std, "mse_play": mse_play,
            "score": (mse_std - mse_play) / mse_std,
            "ms": {r: times[r] * 1000.0 for r in times},
        })
        vals_q.append(hif4.hif4_dequantize(outs["q"]).clone())
    return {"cases": cases, "avg_score": sum(c["score"] for c in cases) / len(cases)}


def dscore_pp(res, base):
    return [round((c["score"] - b["score"]) * 100.0, 3)
            for c, b in zip(res["cases"], base["cases"])]


def changed_frac(SOL, group, base_q_state, rf):
    """Fraction of changed Q value elements vs rf=0 (uses case 0/3)."""
    qh = group["q_num_heads"]
    out = []
    for ti in (0, 3):
        smp = group["test"][ti]
        p0 = SOL.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh,
                                         group["head_dim"],
                                         clone_state(with_rf(base_q_state, 0)))
        p1 = SOL.hif4_dynamic_quantize_q(smp["q"][0], smp["q"][1], qh,
                                         group["head_dim"],
                                         clone_state(with_rf(base_q_state, rf)))
        v0 = hif4.hif4_dequantize(p0)
        v1 = hif4.hif4_dequantize(p1)
        out.append((ti, int(smp["q"][0].shape[0]),
                    round(float((v0 != v1).float().mean()) * 100, 3)))
    return out


def save(key, data):
    db = {}
    if os.path.exists(RES):
        with open(RES) as f:
            db = json.load(f)
    db[key] = data
    with open(RES, "w") as f:
        json.dump(db, f, indent=1)
    print(f"[saved] {key}")


def run_mini():
    SOL = load_sol()
    att = torch.load(os.path.join(ROOT, "example", "mini_sample", "attn.pt"),
                     weights_only=True, map_location="cpu")[0]
    qh, kvh, dh = att["q_num_heads"], att["kv_num_heads"], att["head_dim"]

    base_cal, t_base = calibrate(SOL, att, gptq=True, force_h=True, qs=0, ks=0)
    base = eval_states(SOL, att, base_cal["q_state"], base_cal["k_state"],
                       base_cal["v_state"])
    print(f"[base] cal {t_base:.1f}s avg_score {base['avg_score']:+.4f}")
    for c in base["cases"]:
        print(f"  t{c['ti']} T={c['T']:5d} score={c['score']:+.4f} "
              f"ms q/k/v={c['ms']['q']:.0f}/{c['ms']['k']:.0f}/{c['ms']['v']:.0f}")
    out = {"base": base, "t_cal_base": t_base,
           "state_bytes": {"q": state_bytes(base_cal["q_state"]),
                           "k": state_bytes(base_cal["k_state"])}}
    save("mini_base", out)

    # --- Q/K sweep curves (unguarded: rf injected directly) ---
    sweep = {}
    for sw in (4, 8, 16, 32):
        r_q = eval_states(SOL, att, with_rf(base_cal["q_state"], sw),
                          with_rf(base_cal["k_state"], 0), base_cal["v_state"])
        r_k = eval_states(SOL, att, with_rf(base_cal["q_state"], 0),
                          with_rf(base_cal["k_state"], sw), base_cal["v_state"])
        r_qk = eval_states(SOL, att, with_rf(base_cal["q_state"], sw),
                           with_rf(base_cal["k_state"], sw), base_cal["v_state"])
        sweep[sw] = {
            "q": {"dscore_pp": dscore_pp(r_q, base),
                  "avg_dpp": sum(dscore_pp(r_q, base)) / len(base["cases"]),
                  "ms_q": [c["ms"]["q"] for c in r_q["cases"]]},
            "k": {"dscore_pp": dscore_pp(r_k, base),
                  "avg_dpp": sum(dscore_pp(r_k, base)) / len(base["cases"]),
                  "ms_k": [c["ms"]["k"] for c in r_k["cases"]]},
            "qk": {"dscore_pp": dscore_pp(r_qk, base),
                   "avg_dpp": sum(dscore_pp(r_qk, base)) / len(base["cases"]),
                   "ms_q": [c["ms"]["q"] for c in r_qk["cases"]],
                   "ms_k": [c["ms"]["k"] for c in r_qk["cases"]]},
        }
        print(f"[sw {sw}] q {sweep[sw]['q']['avg_dpp']:+.3f}pp  "
              f"k {sweep[sw]['k']['avg_dpp']:+.3f}pp  "
              f"qk {sweep[sw]['qk']['avg_dpp']:+.3f}pp")
        sys.stdout.flush()
    save("mini_sweep", sweep)

    # --- interaction: no GPTQ (table + refine) ---
    nog_cal, t_nog = calibrate(SOL, att, gptq=False, force_h=True, qs=0, ks=0)
    nog_base = eval_states(SOL, att, nog_cal["q_state"], nog_cal["k_state"],
                           nog_cal["v_state"])
    inter = {"nogptq_base": nog_base, "t_cal": t_nog, "cfg": {}}
    print(f"[nogptq base] avg {nog_base['avg_score']:+.4f} "
          f"(vs gptq base {base['avg_score']:+.4f})")
    for sw in (4, 8, 16, 32):
        r = eval_states(SOL, att, with_rf(nog_cal["q_state"], sw),
                        with_rf(nog_cal["k_state"], sw), nog_cal["v_state"])
        inter["cfg"][sw] = {"dscore_pp_vs_nogptq_base": dscore_pp(r, nog_base),
                            "avg_dpp": sum(dscore_pp(r, nog_base)) / len(base["cases"]),
                            "ms_q": [c["ms"]["q"] for c in r["cases"]],
                            "ms_k": [c["ms"]["k"] for c in r["cases"]]}
        print(f"[nogptq qk sw {sw}] {inter['cfg'][sw]['avg_dpp']:+.3f}pp vs its base")
        sys.stdout.flush()
    save("mini_nogptq", inter)

    # --- guarded variants (deployment-realistic) ---
    guard = {}
    for sw in (4, 8, 16, 32):
        gcal, tg = calibrate(SOL, att, gptq=True, force_h=False, guard=True,
                             qs=sw, ks=sw)
        r = eval_states(SOL, att, gcal["q_state"], gcal["k_state"], gcal["v_state"])
        rfq = gcal["q_state"].get("rf")
        rfk = gcal["k_state"].get("rf")
        guard[sw] = {"rf_q": rfq, "rf_k": rfk, "t_cal": tg,
                     "dscore_pp": dscore_pp(r, base),
                     "avg_dpp": sum(dscore_pp(r, base)) / len(base["cases"])}
        print(f"[guard sw {sw}] rf_q={rfq} rf_k={rfk} t_cal={tg:.1f}s "
              f"{guard[sw]['avg_dpp']:+.3f}pp")
        sys.stdout.flush()
    save("mini_guard", guard)


SYNTH_CFGS = [
    # name, seed, qh, kvh, dh, seqlens, spread, outlier_p
    ("a16_4_64",    4101, 16, 4, 64,  (10, 128, 512, 1024), 0.4, 0.0),
    ("a32_8_128",   4102, 32, 8, 128, (10, 128, 512, 1024), 0.4, 0.0),
    ("a8_8_128",    4103, 8, 8, 128,  (10, 128, 512, 1024), 0.4, 0.0),
    ("a16_2_256",   4104, 16, 2, 256, (10, 128, 512, 1024), 0.4, 0.0),
    ("a4_2_64",     4105, 4, 2, 64,   (10, 256, 1024, 2048), 0.4, 0.0),
    ("a16_8_64",    4106, 16, 8, 64,  (10, 512, 1024, 2048), 0.4, 0.0),
    ("a32_8_128_sp", 4107, 32, 8, 128, (10, 128, 512, 1024), 0.7, 0.002),
]


def run_synth():
    SOL = load_sol()
    for name, seed, qh, kvh, dh, seqlens, spread, outp in SYNTH_CFGS:
        torch.manual_seed(0)
        g = synth.make_attn_group(seed, qh, kvh, dh, seqlens=seqlens,
                                  spread=spread, outlier_p=outp)
        base_cal, t_base = calibrate(SOL, g, gptq=True, force_h=True, qs=0, ks=0)
        base = eval_states(SOL, g, base_cal["q_state"], base_cal["k_state"],
                           base_cal["v_state"])
        rec = {"shape": [qh, kvh, dh], "seqlens": list(seqlens),
               "spread": spread, "outlier_p": outp,
               "t_cal": t_base, "base": base,
               "state_bytes_q": state_bytes(base_cal["q_state"]), "cfg": {}}
        print(f"[{name}] qh{qh} kvh{kvh} dh{dh} cal {t_base:.1f}s "
              f"base avg {base['avg_score']:+.4f}")
        for c in base["cases"]:
            print(f"  T={c['T']:5d} score={c['score']:+.4f} "
                  f"ms q/k/v={c['ms']['q']:.0f}/{c['ms']['k']:.0f}/{c['ms']['v']:.0f}")
        sys.stdout.flush()
        for sw in (4, 8):
            r = eval_states(SOL, g, with_rf(base_cal["q_state"], sw),
                            with_rf(base_cal["k_state"], sw), base_cal["v_state"])
            rec["cfg"][sw] = {"dscore_pp": dscore_pp(r, base),
                              "avg_dpp": sum(dscore_pp(r, base)) / len(base["cases"]),
                              "ms_q": [c["ms"]["q"] for c in r["cases"]],
                              "ms_k": [c["ms"]["k"] for c in r["cases"]]}
            print(f"  [qk sw {sw}] avg {rec['cfg'][sw]['avg_dpp']:+.3f}pp "
                  f"per-case {rec['cfg'][sw]['dscore_pp']}")
            sys.stdout.flush()
        save(f"synth_{name}", rec)


def run_rounds():
    """Depth = ROUNDS per sweep (the real cost knob; sweeps early-exit).
    Value + refine-only cost vs rounds on mini + two synth shapes."""
    SOL = load_sol()

    def micro(group, qs_only=False):
        qh, kvh, dh = group["q_num_heads"], group["kv_num_heads"], group["head_dim"]
        base_cal, _ = calibrate(SOL, group, gptq=True, force_h=True, qs=0, ks=0)
        rec = {}
        for ti, smp in enumerate(group["test"]):
            T = int(smp["q"][0].shape[0])
            # build deployment inputs: post-GPTQ values + bf16 H from state
            x = SOL.dequantize_nvfp4(smp["q"][0], smp["q"][1]).float()
            rot = base_cal["q_state"].get("rot") == 1
            if rot:
                Rm = SOL._make_R(dh)
                x = (x.view(T, qh, dh) @ Rm).reshape(T, -1).contiguous()
            p = SOL._dyn_table(x, None, has_scale=False)
            unit = SOL._params_unit_flat(p)
            if base_cal["q_state"].get("gq") == 1:
                u = base_cal["q_state"]["u"]
                xs = x.view(T, qh, dh).permute(1, 0, 2).contiguous()
                us = unit.view(T, qh, dh).permute(1, 0, 2).contiguous()
                rep = qh // kvh
                hv_of = torch.arange(qh) // rep
                u_full = u[hv_of.clamp_max(u.shape[0] - 1)].float()
                vals = SOL._gptq_quantize_batched(xs, us, u_full)
                vals = vals.permute(1, 0, 2).reshape(T, -1).contiguous()
            else:
                vals = SOL._deq_params(p)
            H = base_cal["q_state"]["H"]
            ent = {}
            for rnds in (2, 4, 8, 20):
                SOL._REF_ATTN_ROUNDS = rnds
                ts = []
                for _ in range(3):
                    t0 = time.perf_counter()
                    vr = SOL._refine_attn_heads(x, vals.clone(), unit.clone(),
                                                H, qh, 1)
                    ts.append(time.perf_counter() - t0)
                ent[rnds] = {"ms": sorted(ts)[1] * 1000.0,
                             "changed": int((vr != vals).sum()),
                             "Jrel": None}
                ent[rnds]["vals"] = vr
            # J on fp32 Hk for value curve
            Hk32 = H.float()
            def Jval(v):
                dq = (v - x).view(T, qh, dh)
                hv = torch.arange(qh) * kvh // qh
                j = 0.0
                for h in range(qh):
                    dqh = dq[:, h]
                    j += (dqh @ Hk32[hv[h]] * dqh).sum().item()
                return j
            J0 = Jval(vals)
            for rnds in ent:
                vv = ent[rnds].pop("vals")
                ent[rnds]["Jrem"] = round(1.0 - Jval(vv) / J0, 4)
            rec[f"t{ti}_T{T}"] = ent
            print(f"  T={T:5d} " + "  ".join(
                f"r{r}: {ent[r]['ms']:.0f}ms Jrem {ent[r]['Jrem']*100:.1f}%"
                for r in ent))
            sys.stdout.flush()
        SOL._REF_ATTN_ROUNDS = 20
        return rec

    att = torch.load(os.path.join(ROOT, "example", "mini_sample", "attn.pt"),
                     weights_only=True, map_location="cpu")[0]
    print("[mini]")
    out = {"mini": micro(att)}
    torch.manual_seed(0)
    g = synth.make_attn_group(4102, 32, 8, 128, seqlens=(10, 128, 512, 1024))
    print("[synth a32_8_128]")
    out["a32_8_128"] = micro(g)
    save("rounds_curve", out)


def run_vcheck():
    """Theorem confirmation: uniform-weight V refinement flips nothing."""
    SOL = load_sol()
    att = torch.load(os.path.join(ROOT, "example/mini_sample", "attn.pt"),
                     weights_only=True, map_location="cpu")[0]
    qh, kvh, dh = att["q_num_heads"], att["kv_num_heads"], att["head_dim"]
    rec = {}
    for ti, smp in enumerate(att["test"]):
        x = SOL.dequantize_nvfp4(smp["v"][0], smp["v"][1]).float()
        p = SOL._dyn_table(x, None, has_scale=False)
        base = SOL._deq_params(p)
        unit = SOL._params_unit_flat(p)
        eye = torch.eye(dh).unsqueeze(0).expand(kvh, dh, dh).contiguous()
        rec[ti] = {}
        for sw in (4, 8, 16):
            vr = SOL._refine_attn_heads(x, base.clone(), unit.clone(), eye,
                                        kvh, sw)
            rec[ti][sw] = int((vr != base).sum())
        print(f"  [v t{ti} T={x.shape[0]}] changed: "
              + " ".join(f"sw{sw}={rec[ti][sw]}" for sw in (4, 8, 16)))
    save("v_noop", rec)


def run_ident():
    """Bit-identity of the copy (flags off) vs the v33 mainline."""
    SOL = load_sol()
    S0 = load_mod(os.path.join(ROOT, "example", "solution", "solution.py"), "sol_v33")
    att = torch.load(os.path.join(ROOT, "example", "mini_sample", "attn.pt"),
                     weights_only=True, map_location="cpu")[0]
    qh, kvh, dh = att["q_num_heads"], att["kv_num_heads"], att["head_dim"]
    ok_all = True
    S0_res = S0.hif4_calibration_attention(att["calib"], qh, kvh, dh)
    SOL.ATTN_REFINE_FORCE_H = True
    SOL.ATTN_REFINE_Q_SWEEPS = 0
    SOL.ATTN_REFINE_K_SWEEPS = 0
    c1 = SOL.hif4_calibration_attention(att["calib"], qh, kvh, dh)
    for role in ("q", "k"):
        for k in ("rot", "kvh", "gq"):
            assert c1[role + "_state"][k] == S0_res[role + "_state"][k]
        assert torch.equal(c1[role + "_state"]["u"],
                           S0_res[role + "_state"]["u"]), role
    for ti, smp in enumerate(att["test"]):
        for role, f0, f1, nh in (
            ("q", S0.hif4_dynamic_quantize_q, SOL.hif4_dynamic_quantize_q, qh),
            ("k", S0.hif4_dynamic_quantize_k, SOL.hif4_dynamic_quantize_k, kvh),
            ("v", S0.hif4_dynamic_quantize_v, SOL.hif4_dynamic_quantize_v, kvh),
        ):
            p0 = f0(smp[role][0], smp[role][1], nh, dh, S0_res[role + "_state"])
            p1 = f1(smp[role][0], smp[role][1], nh, dh, c1[role + "_state"])
            for k in p0:
                if not torch.equal(p0[k], p1[k]):
                    ok_all = False
                    print(f"MISMATCH t{ti} {role} {k}")
    print("IDENT" if ok_all else "DIFFERS")
    save("ident", {"ok": ok_all})


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "mini"
    if mode == "ident":
        run_ident()
    elif mode == "mini":
        run_mini()
    elif mode == "synth":
        run_synth()
    elif mode == "rounds":
        run_rounds()
    elif mode == "vcheck":
        run_vcheck()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
