"""V column-mean-zeroing test on judge-like (random Gaussian) attention data.

If attention probs P are near-uniform (random q/k), the output error from V
quantization collapses to the COLUMN MEANS of dV = Vh - V. A post-pass that
moves a few mant values to zero each column sum should slash output MSE.
"""
import sys, os
import importlib.util

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "..", "example", "solution"))
sys.path.insert(0, ROOT)


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load_mod(os.path.join(ROOT, "..", "example", "solution", "solution.py"), "sol")
import hif4  # noqa: E402
import synth  # noqa: E402
import variants as V  # noqa: E402


def attn_score(group, fix_colmeans=False, passes=6):
    qh, kvh, dh = group["q_num_heads"], group["kv_num_heads"], group["head_dim"]
    calib, tests = group["calib"], group["test"]
    st = S.hif4_calibration_attention(calib, qh, kvh, dh)

    def out_of(q, k, v):
        return hif4.attn_ref(q, k, v, qh, kvh, dh)

    tot_std = tot_play = 0.0
    for smp in tests:
        q_ref = hif4.dequantize_nvfp4(*smp["q"])
        k_ref = hif4.dequantize_nvfp4(*smp["k"])
        v_ref = hif4.dequantize_nvfp4(*smp["v"])
        ref = out_of(q_ref.float(), k_ref.float(), v_ref.float())
        q_std = V.deq(V.quant_alg1(q_ref.float()))
        k_std = V.deq(V.quant_alg1(k_ref.float()))
        v_std = V.deq(V.quant_alg1(v_ref.float()))
        tot_std += ((out_of(q_std, k_std, v_std) - ref) ** 2).mean().item()

        pq = S.hif4_dynamic_quantize_q(*smp["q"], qh, dh, st["q_state"])
        pk = S.hif4_dynamic_quantize_k(*smp["k"], kvh, dh, st["k_state"])
        pv = S.hif4_dynamic_quantize_v(*smp["v"], kvh, dh, st["v_state"])
        vq = hif4.hif4_dequantize(pq)
        vk = hif4.hif4_dequantize(pk)
        vvv = hif4.hif4_dequantize(pv)

        if fix_colmeans:
            vf = v_ref.float()
            dv = vvv - vf
            unit = (pv["scale_factor"] * pv["scale_lv2"] * pv["scale_lv3"]) \
                .expand_as(pv["mant"]).flatten(-4, -1)
            mant = pv["mant"].flatten(-4, -1).clone()
            resid = dv.sum(dim=0)  # (C,) column sums
            step = 0.25 * unit
            for _ in range(passes):
                # per column: how many steps and in which direction
                nstep = torch.round(resid / (step.sum(dim=0)))
                nstep = torch.clamp(nstep, -3, 3)
                if nstep.abs().max() < 0.5:
                    break
                # spread the steps over rows with mant room, largest |dv| first:
                # simple greedy: apply to rows where moving reduces |dv| least...
                # here: apply uniformly to the rows with largest headroom in sign
                can_up = (mant + 0.25 <= 1.75).float()
                can_dn = (mant - 0.25 >= 0.0).float()
                up_mask = (nstep > 0).float()
                dn_mask = (nstep < 0).float()
                # pick one row per column (the one with biggest |dv| and room)
                for j in range(vf.shape[1]):
                    ns = int(nstep[j].item())
                    if ns == 0:
                        continue
                    direction = 1 if ns > 0 else -1
                    # rows movable in direction, largest |resid contribution|
                    if direction > 0:
                        room = can_up[:, j]
                    else:
                        room = can_dn[:, j]
                    cand = room.nonzero().flatten()
                    if cand.numel() == 0:
                        continue
                    # move on the row with the largest same-sign dv (excess)
                    dvc = dv[:, j]
                    if direction > 0:
                        pick = cand[dvc[cand].argmax()]
                    else:
                        pick = cand[dvc[cand].argmin()]
                    mant[pick, j] += 0.25 * direction
                    dv[pick, j] += step[pick, j] * direction
                    resid[j] -= step[pick, j] * direction * 0  # resid updated below
                resid = dv.sum(dim=0)
            pv["mant"] = mant.reshape(pv["mant"].shape)
            vvv = hif4.hif4_dequantize(pv)

        tot_play += ((out_of(vq, vk, vvv) - ref) ** 2).mean().item()
    n = len(tests)
    return tot_std / n, tot_play / n


torch.manual_seed(0)
for name, group in [
    ("gqa_256_r0.4", synth.make_attn_group(21, 16, 2, 256, spread=0.4)),
    ("mha_128_r0.3", synth.make_attn_group(22, 8, 8, 128, spread=0.3)),
    ("flat_256_r0.1", synth.make_attn_group(23, 16, 2, 256, spread=0.1)),
    ("gqa_128_r0.5", synth.make_attn_group(24, 32, 4, 128, spread=0.5)),
]:
    ms, mp = attn_score(group, fix_colmeans=False)
    ms2, mp2 = attn_score(group, fix_colmeans=True)
    print(f"{name:16s} base score {100*(1-mp/ms):+6.2f}%   colfix {100*(1-mp2/ms2):+6.2f}%   "
          f"(play MSE {mp:.3e} -> {mp2:.3e})")
