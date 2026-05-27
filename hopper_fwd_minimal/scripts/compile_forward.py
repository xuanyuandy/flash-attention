#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="FakeTensor compile pass for the minimal Hopper forward tests.")
    parser.add_argument("-k", default="forward_smoke", help="pytest -k expression")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env["FLASH_ATTENTION_FAKE_TENSOR"] = "1"
    env["FLASH_ATTENTION_ARCH"] = "sm_90a"
    env["CUTE_DSL_ARCH"] = "sm_90a"
    env.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "hopper_fwd_minimal/tests/test_forward.py",
        "-k",
        args.k,
    ]
    if args.quiet:
        cmd.insert(3, "-q")
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
