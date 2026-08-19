"""Build an uploadable solution.zip and archive it under dist/ with a version.

Usage:
    python build_zip.py            # package current example/solution/solution.py
    python build_zip.py "note"     # same, plus a changelog note

Reads the version from the VERSION file (single integer). Bump it manually
after finalizing each iteration. Refuses to package unless the official
self_check passes on mini_sample.
"""
from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.join(ROOT, "example", "solution")
SOLUTION_PY = os.path.join(SOLUTION_DIR, "solution.py")
SELF_CHECK = os.path.join(ROOT, "example", "self_check.py")
DATASETS = os.path.join(ROOT, "example", "mini_sample")
VERSION_FILE = os.path.join(ROOT, "VERSION")
DIST = os.path.join(ROOT, "dist")
CHANGELOG = os.path.join(ROOT, "dev", "CHANGELOG.md")


def read_version() -> int:
    with open(VERSION_FILE, encoding="utf-8") as f:
        return int(f.read().strip())


def run_self_check() -> None:
    print("[build] running official self_check ...")
    proc = subprocess.run(
        [sys.executable, SELF_CHECK, "--solution_dir", SOLUTION_DIR, "--datasets_dir", DATASETS],
        capture_output=True,
        text=True,
    )
    tail = (proc.stdout or "").strip().splitlines()
    print("\n".join("    " + line for line in tail[-4:]))
    if proc.returncode != 0 or "ALL OUTPUT-FORMAT CHECKS PASSED" not in (proc.stdout or ""):
        raise SystemExit("[build] ABORT: self_check failed, nothing packaged.")


def main() -> None:
    note = sys.argv[1] if len(sys.argv) > 1 else ""
    version = read_version()
    run_self_check()

    os.makedirs(DIST, exist_ok=True)
    zip_path = os.path.join(DIST, f"solution_v{version}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(SOLUTION_PY, arcname="solution.py")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    if names != ["solution.py"]:
        raise SystemExit(f"[build] ABORT: unexpected zip content {names}")

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"## v{version} - {stamp}\n- artifact: dist/solution_v{version}.zip\n"
    if note:
        entry += f"- note: {note}\n"
    entry += "\n"
    os.makedirs(os.path.dirname(CHANGELOG), exist_ok=True)
    with open(CHANGELOG, "a", encoding="utf-8") as f:
        f.write(entry)

    print(f"[build] OK -> {zip_path} (contents: {names})")


if __name__ == "__main__":
    main()
