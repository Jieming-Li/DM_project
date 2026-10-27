"""Our read-only HDF5 loader and composite-key manifest handling."""
from __future__ import annotations

from dataclasses import dataclass
import csv
from functools import lru_cache
import importlib.util
from pathlib import Path, PurePosixPath

from .runtime import BENCHMARK, DATA, inside

import h5py
import numpy as np

FIELDS = ("ne", "Tev", "potential", "ui", "nn")
INPUTS = ("V_a", "mdot", "P_b", "B_hat")
SIMULATION_SOURCES = (
    "core/spt100_gbohm_12k_nc3_s42_20260617.h5",
    "extrapolation_pools/extrap_pool_D3_2400.h5",
    "extrapolation_pools/extrap_pool_D4_2400.h5",
)
UNITS = {"V_a": "V", "mdot": "kg/s", "P_b": "Torr", "B_hat": "dimensionless",
         "z": "m", "ne": "m^-3", "Tev": "eV", "potential": "V", "ui": "m/s", "nn": "m^-3", "E": "V/m"}


@dataclass(frozen=True, order=True)
class RunKey:
    source_dataset: str
    run_id: int

    @property
    def group(self) -> str:
        return f"run_{self.run_id:05d}"

    def as_dict(self) -> dict:
        return {"source_dataset": self.source_dataset, "run_id": self.run_id}


def source_path(source: str) -> Path:
    p = PurePosixPath(source)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Unsafe dataset source: {source}")
    target = inside(DATA / source)
    if not target.resolve().is_relative_to(DATA.resolve()):
        raise ValueError(f"Source outside data directory: {source}")
    return target


def manifest(partition: str) -> dict[str, list[RunKey]]:
    if partition not in ("D1", "D5"):
        raise ValueError("Stage one uses only D1 and D5 manifests")
    out = {"train": [], "test": []}
    with (BENCHMARK / "benchmark/manifests" / f"{partition}.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            source_path(row["source_dataset"])
            if row["split"] not in out:
                raise ValueError(f"Unexpected split: {row}")
            key = RunKey(row["source_dataset"], int(row["run_id"]))
            if key.run_id < 0:
                raise ValueError(f"Negative run id: {key}")
            out[row["split"]].append(key)
    for split, keys in out.items():
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate composite keys in {partition}/{split}")
    if set(out["train"]) & set(out["test"]):
        raise ValueError(f"Train/test overlap in {partition}")
    return out


def development_pool() -> list[RunKey]:
    d1, d5 = manifest("D1"), manifest("D5")
    pool = sorted(set(d1["train"]) & set(d5["train"]))
    assert not set(pool) & (set(d1["test"]) | set(d5["test"]))
    return pool


@dataclass
class LoadedRuns:
    keys: list[RunKey]
    inputs: np.ndarray
    outputs: np.ndarray
    z: np.ndarray


def read_group(group: h5py.Group, fields=FIELDS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([group.attrs[name] for name in INPUTS], dtype=np.float64)
    z = np.asarray(group["z"][:], dtype=np.float64)
    if x.shape != (4,) or z.shape != (202,) or not np.all(np.diff(z) > 0):
        raise ValueError(f"Invalid input/grid shape or grid order in {group.name}")
    values = []
    for name in fields:
        raw = np.asarray(group[name][:], dtype=np.float64)
        expected = (3, 202) if name == "ui" else (202,)
        if raw.shape != expected or not np.isfinite(raw).all():
            raise ValueError(f"Invalid shape/nonfinite {name} in {group.name}: {raw.shape}")
        values.append(raw[0] if name == "ui" else raw)
    y = np.stack(values)
    if not np.isfinite(x).all() or not np.isfinite(z).all():
        raise ValueError(f"Nonfinite input/grid in {group.name}")
    return x, y, z


def load_runs(keys: list[RunKey], fields=FIELDS) -> LoadedRuns:
    if not keys:
        raise ValueError("Empty run selection")
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate composite keys in load selection")
    x = np.empty((len(keys), 4), dtype=np.float64)
    y = np.empty((len(keys), len(fields), 202), dtype=np.float64)
    grid = None
    for source in sorted({k.source_dataset for k in keys}):
        with h5py.File(source_path(source), "r") as f:
            for i, key in enumerate(keys):
                if key.source_dataset != source:
                    continue
                if key.group not in f:
                    raise KeyError(f"Missing referenced run: {key}")
                x[i], y[i], z = read_group(f[key.group], fields)
                if grid is None:
                    grid = z
                elif not np.array_equal(grid, z):
                    raise ValueError(f"Inconsistent spatial grid at {key}")
    return LoadedRuns(list(keys), x, y, grid)


@lru_cache(maxsize=None)
def official_module(name: str):
    if name not in ("metrics", "score"):
        raise ValueError(name)
    path = BENCHMARK / "benchmark" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"htsb_official_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
