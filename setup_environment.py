#!/usr/bin/env python3
"""Create a copied, project-local Python 3.11 venv and install pinned wheels."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import venv

from src.runtime import ROOT, RESULTS, inside, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create-only", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Invoke this setup script with installed Python 3.11; no system software will be installed")
    venv_dir = inside(ROOT / ".venv")
    if not venv_dir.exists():
        venv.EnvBuilder(with_pip=True, symlinks=False).create(venv_dir)
    python = inside(venv_dir / "bin/python")
    probe = subprocess.run([str(python), "-B", "-c", "import sys; print(sys.version); assert sys.version_info[:2] == (3,11); assert sys.prefix != sys.base_prefix"], cwd=ROOT, text=True, capture_output=True)
    print(probe.stdout, end="")
    probe.check_returncode()
    if args.create_only:
        return
    cmd = [str(python), "-B", "-m", "pip", "--disable-pip-version-check", "install", "--only-binary=:all:", "-r", str(ROOT / "requirements.txt")]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    (RESULTS / "dependency_install.txt").write_text(f"Command: {shlex.join(cmd)}\nCWD: {ROOT}\nExit status: {result.returncode}\n\nSTDOUT\n{result.stdout}\nSTDERR\n{result.stderr}")
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    result.check_returncode()


if __name__ == "__main__":
    main()
