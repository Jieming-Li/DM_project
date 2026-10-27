"""Project boundary and task-local runtime configuration (stdlib only)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "external/hallthruster-surrogate-benchmark"
DATA = ROOT / "data"
RESULTS = ROOT / "results/stage1"


def inside(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.resolve().is_relative_to(ROOT):
        raise RuntimeError(f"Path leaves DM_PROJECT_ROOT: {path}")
    return path


def configure() -> None:
    if ROOT.name != "DM_project" or not Path.cwd().resolve().is_relative_to(ROOT):
        raise RuntimeError(f"Run from the existing DM_project: {ROOT}")
    # Check symlinks before creating directories, including venv and Git metadata.
    for base, dirs, files in os.walk(ROOT, followlinks=False):
        for name in dirs + files:
            p = Path(base) / name
            if p.is_symlink():
                inside(p)
    for rel in (".git", "external/hallthruster-surrogate-benchmark/.git"):
        p = ROOT / rel
        if p.is_file():
            raise RuntimeError(f"External/worktree Git metadata is unsupported: {p}")
    paths = {
        "TMPDIR": ".tmp", "TMP": ".tmp", "TEMP": ".tmp",
        "PIP_CACHE_DIR": ".cache/pip", "MPLCONFIGDIR": ".cache/matplotlib",
        "XDG_CACHE_HOME": ".cache", "UV_CACHE_DIR": ".cache/uv",
    }
    for name, rel in paths.items():
        p = inside(ROOT / rel)
        p.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(p)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["HTSB_DATA_DIR"] = str(DATA)
    os.environ["MPLBACKEND"] = "Agg"
    # Small pilot: bounded, repeatable BLAS use; set before numerical imports.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    sys.dont_write_bytecode = True
    RESULTS.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, obj: object) -> None:
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


configure()
