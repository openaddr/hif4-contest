"""Local reproduction harness for the v15 online "wrong answer" incident.

v15 added lattice refinement: the linear activation_state now carries two
C x C float32 Gram matrices (gw = q_used^T q_used, gwf = w_final^T q_used)
and online 6 of ~50 linear groups came back WA while attn passed and the
official mini self_check passed 22/22.

This harness builds synthetic linear groups across extreme shapes
(C x N x T x spread x w_spike), writes each as its own dataset directory
(linear.pt + hardlinked attn.pt), and provides:

  build   materialize dev/stress_data/<name>/
  probe   run calibration directly; report state bytes per entry, finiteness,
          calibration time, peak RSS, and one timed dynamic call
  check   run the OFFICIAL example/self_check.py per dataset dir

Usage:
  python dev/stress.py build
  python dev/stress.py probe [names...]
  python dev/stress.py check [names...]
"""
from __future__ import annotations

import ctypes
import importlib.util
import os
import subprocess
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dev"))
import synth  # noqa: E402  (make_linear_group / make_attn_group)

OUT_DIR = os.path.join(ROOT, "dev", "stress_data")
MINI_ATTN = os.path.join(ROOT, "example", "mini_sample", "attn.pt")
SELF_CHECK = os.path.join(ROOT, "example", "self_check.py")
SOLUTION_DIR = os.path.join(ROOT, "example", "solution")

# name, C(K), N(M), calib tokens, test tokens, spread, outlier_p, w_spread
CONFIGS = [
    ("c1024_n8192",       1024, 8192, (10, 128, 512, 1024), (128, 1024), 0.5, 0.0, 0.3),
    ("c2048_n8192",       2048, 8192, (10, 128, 512, 1024), (10, 128, 512, 1024), 0.5, 0.0, 0.3),
    ("c2048_n1024",       2048, 1024, (10, 128, 512), (128, 512), 0.5, 0.0, 0.3),
    ("c2048_wspike",      2048, 8192, (10, 512, 1024), (512, 1024), 0.6, 0.002, 0.95),
    ("c2048_bigT",        2048, 8192, (128, 512, 1024, 2048), (512, 2048), 0.5, 0.0, 0.3),
    ("c2048_tiny",        2048, 4096, (10,), (10, 128), 0.5, 0.0, 0.3),
    ("c4096_n4096",       4096, 4096, (10, 128, 512, 1024), (128, 512, 1024), 0.5, 0.0, 0.3),
    ("c4096_n8192",       4096, 8192, (128, 512, 1024), (128, 1024), 0.5, 0.0, 0.3),
    ("c4096_spiky",       4096, 4096, (128, 512, 1024), (512, 1024), 0.8, 0.002, 0.9),
    ("c4096_flat",        4096, 4096, (128, 1024), (128, 1024), 0.15, 0.0, 0.1),
    ("c6144_n6144",       6144, 6144, (128, 512, 1024), (128, 1024), 0.5, 0.0, 0.3),
    ("c8192_n1024",       8192, 1024, (128, 512), (128, 512), 0.5, 0.0, 0.3),
    ("c8192_n8192",       8192, 8192, (10, 128, 512, 1024), (128, 1024), 0.5, 0.0, 0.3),
    ("c8192_n8192_spiky", 8192, 8192, (128, 512, 1024), (128, 1024), 0.9, 0.001, 0.9),
    ("c8192_n8192_flat",  8192, 8192, (128, 1024), (128, 1024), 0.15, 0.0, 0.1),
]

# extra configs behind a flag (very heavy: >2 GiB datasets, minutes each)
BIG_CONFIGS = [
    ("c16384_n4096", 16384, 4096, (128, 512), (128, 512), 0.5, 0.0, 0.3),
]


# ---- windows peak-RSS ------------------------------------------------------

class _PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_rss_gib() -> float:
    try:
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 2 ** 30
    except Exception:
        return -1.0


def state_bytes(value) -> int:
    if type(value) is torch.Tensor:
        return value.numel() * value.element_size()
    if value is None or type(value) in (bool, int, float, str):
        return 0
    if type(value) in (list, tuple):
        return sum(state_bytes(v) for v in value)
    if type(value) is dict:
        return sum(state_bytes(v) for v in value.values())
    return 0


def build(names=None, extra=False) -> None:
    cfgs = list(CONFIGS) + (list(BIG_CONFIGS) if extra else [])
    if names:
        cfgs = [c for c in cfgs if c[0] in names]
    for i, (name, C, N, calib_T, test_T, spread, outp, wspread) in enumerate(cfgs):
        d = os.path.join(OUT_DIR, name)
        os.makedirs(d, exist_ok=True)
        lin = os.path.join(d, "linear.pt")
        if not os.path.exists(lin):
            t0 = time.perf_counter()
            g = synth.make_linear_group(1000 + 7 * i, N, C, tokens=calib_T,
                                        spread=spread, outlier_p=outp,
                                        w_spread=wspread)
            g2 = synth.make_linear_group(9000 + 7 * i, N, C, tokens=test_T,
                                         spread=spread, outlier_p=outp,
                                         w_spread=wspread)
            g["test_activation_list"] = g2["test_activation_list"]
            torch.save([g], lin)
            print(f"[build] {name}: C={C} N={N} calib={calib_T} test={test_T} "
                  f"({time.perf_counter() - t0:.1f}s, "
                  f"{os.path.getsize(lin) / 2 ** 20:.0f} MiB)")
        attn = os.path.join(d, "attn.pt")
        if not os.path.exists(attn):
            try:
                os.link(MINI_ATTN, attn)  # hardlink: no extra disk
            except OSError:
                import shutil
                shutil.copyfile(MINI_ATTN, attn)
    print("[build] done ->", OUT_DIR)


def load_solution():
    spec = importlib.util.spec_from_file_location(
        "_stress_sol", os.path.join(SOLUTION_DIR, "solution.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_selfcheck():
    spec = importlib.util.spec_from_file_location(
        "_selfcheck", SELF_CHECK)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def probe(names=None) -> None:
    SC = load_selfcheck()
    S = load_solution()
    for name, C, N, calib_T, test_T, *_ in _select(names):
        d = os.path.join(OUT_DIR, name)
        group = torch.load(os.path.join(d, "linear.pt"), weights_only=True,
                           map_location="cpu")[0]
        torch.manual_seed(0)
        t0 = time.perf_counter()
        out = S.hif4_calibration_and_quantize_weight(
            group["weight"][0], group["weight"][1], group["calib_activation_list"])
        t_cal = time.perf_counter() - t0
        st = out["activation_state"]
        parts = []
        for k, v in st.items():
            if type(v) is torch.Tensor:
                mb = v.numel() * v.element_size() / 2 ** 20
                fin = ("finite" if not v.is_floating_point() or bool(torch.isfinite(v).all())
                       else "NON-FINITE")
                parts.append(f"{k}:{tuple(v.shape)}/{str(v.dtype).replace('torch.', '')}"
                             f"/{mb:.0f}MiB/{fin}")
            else:
                parts.append(f"{k}={v if not isinstance(v, torch.Tensor) else 'tensor'}")
        tb = state_bytes(st) / 2 ** 20
        werr = SC.validate_hif4_params(out["weight_params"],
                                       tuple(group["weight"][0].shape), "w")
        serr = SC.validate_frozen_state(st, "state")
        # dynamic calls on every test sample
        dyn = []
        for pair in group["test_activation_list"]:
            t0 = time.perf_counter()
            p = S.hif4_dynamic_quantize_activation(pair[0], pair[1], st)
            dt = time.perf_counter() - t0
            err = SC.validate_hif4_params(p, tuple(pair[0].shape), "a")
            dyn.append((dt, err))
        print(f"[probe] {name} C={C} N={N}: cal {t_cal:.1f}s peakRSS "
              f"{peak_rss_gib():.2f}GiB state {tb:.0f}MiB")
        print(f"        state[{'; '.join(parts)}]")
        print(f"        weight_params errors={werr or 'NONE'} state errors={serr or 'NONE'}")
        for i, (dt, err) in enumerate(dyn):
            print(f"        dyn[{i}] T={group['test_activation_list'][i][0].shape[0]} "
                  f"{dt:.2f}s errors={err or 'NONE'}")
        sys.stdout.flush()


def check(names=None) -> None:
    for name, *_ in _select(names):
        d = os.path.join(OUT_DIR, name)
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, SELF_CHECK, "--solution_dir", SOLUTION_DIR,
             "--datasets_dir", d],
            capture_output=True, text=True)
        out = (proc.stdout or "") + (proc.stderr or "")
        tail = "\n".join("        " + ln for ln in out.strip().splitlines()[-5:])
        verdict = "PASS" if proc.returncode == 0 else "FAIL"
        print(f"[check] {name}: {verdict} ({time.perf_counter() - t0:.0f}s)\n{tail}")
        bad = [ln for ln in out.splitlines() if "FAILED" in ln]
        for ln in bad:
            print(f"        >> {ln}")
        sys.stdout.flush()


def _select(names, include_big=False):
    if names:
        cfgs = list(CONFIGS) + list(BIG_CONFIGS)
        cfgs = [c for c in cfgs if c[0] in names]
        if not cfgs:
            raise SystemExit(f"no config matches {names}")
        return cfgs
    return list(CONFIGS) + (list(BIG_CONFIGS) if include_big else [])


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    names = [a for a in sys.argv[2:] if not a.startswith("-")]
    if mode == "build":
        build(names, extra="--big" in sys.argv)
    elif mode == "probe":
        probe(names)
    elif mode == "check":
        check(names)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
