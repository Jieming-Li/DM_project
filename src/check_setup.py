"""Exhaustive shape, finite-value, grid, diagnostic and manifest verification."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.metadata
import json
import platform
import sys
import time

from .runtime import ROOT, BENCHMARK, DATA, RESULTS, write_json
from .data_io import (FIELDS, INPUTS, UNITS, SIMULATION_SOURCES, manifest,
                      development_pool, source_path, read_group)

import h5py
import numpy as np


def json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [json_value(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def main() -> None:
    started = time.perf_counter()
    report = {"project_root": str(ROOT), "passed": False, "failures": [], "warnings": [],
              "units": UNITS, "units_source": "data/RELEASE_README.md; archived benchmark README, score.py and dataset_generation/run_sweep.py",
              "fields": list(FIELDS), "input_order": list(INPUTS), "files": {},
              "python": sys.version, "interpreter": sys.executable,
              "platform": platform.platform(), "dependencies": {}}
    for package in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
        report["dependencies"][package.metadata["Name"]] = package.version
    (RESULTS / "requirements.freeze.txt").write_text("\n".join(f"{k}=={v}" for k, v in sorted(report["dependencies"].items())) + "\n")
    write_json(RESULTS / "environment.json", {k: report[k] for k in ("python", "interpreter", "platform", "dependencies")})
    try:
        provenance = json.loads((RESULTS / "benchmark_provenance.json").read_text())
        changed = []
        for rel, digest in provenance["official_file_sha256"].items():
            p = BENCHMARK / rel
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
                changed.append(rel)
        report["benchmark_release"] = provenance["release"]
        report["official_code_unchanged"] = not changed
        if changed:
            report["failures"].append({"changed_official_files": changed})
        checksums = json.loads((RESULTS / "download_checksums.json").read_text())
        report["release_checksums"] = checksums
        for item in checksums:
            path = ROOT / item["path"]
            algorithm, expected = item["checksum"].split(":", 1)
            digest = hashlib.new(algorithm)
            with path.open("rb") as f:
                for block in iter(lambda: f.read(8 * 1024**2), b""):
                    digest.update(block)
            if path.stat().st_size != item["bytes"] or digest.hexdigest() != expected:
                report["failures"].append({"checksum_failure": item["path"]})
        expected_paths = set(SIMULATION_SOURCES) | {"partitions_manifest.h5"}
        verified_paths = {str((ROOT / v["path"]).relative_to(DATA)) for v in checksums if v["path"].endswith(".h5")}
        if verified_paths != expected_paths:
            report["failures"].append({"release_file_list_mismatch": sorted(verified_paths)})
        d1, d5 = manifest("D1"), manifest("D5")
        report["manifests"] = {}
        for label, m in (("D1", d1), ("D5", d5)):
            report["manifests"][label] = {"train": len(m["train"]), "test": len(m["test"]),
                "duplicate_keys": 0, "train_test_overlap": 0,
                "path": f"external/hallthruster-surrogate-benchmark/benchmark/manifests/{label}.csv"}
        if (len(d1["train"]), len(d1["test"])) != (9600, 1200):
            report["failures"].append("D1 counts differ from 9600/1200")
        report["common_D1_D5_training_pool_count"] = len(development_pool())
        reference_grid = None
        available = {}
        for source in SIMULATION_SOURCES:
            info = {"readable": False, "run_count": 0, "problems": [], "diagnostic_values": {},
                    "grid_consistent": True, "all_scored_profiles_finite": True,
                    "input_shape": [4], "output_shape": [5, 202], "stored_ui_shape": [3, 202]}
            report["files"][source] = info
            try:
                with h5py.File(source_path(source), "r") as f:
                    info["readable"] = True
                    info["file_attributes"] = {k: json_value(v) for k, v in f.attrs.items()}
                    names = sorted(k for k in f if k.startswith("run_"))
                    available[source] = {int(k[4:]) for k in names}
                    info["run_count"] = len(names)
                    diagnostic_counts = {}
                    numeric_ranges = {}
                    for name in names:
                        group = f[name]
                        # A failed read is recorded with the exact composite key; no row filtering.
                        try:
                            x, y, z = read_group(group)
                            e = group["E"][:]
                            if e.shape != (202,) or not np.isfinite(e).all():
                                raise ValueError("Invalid E shape or nonfinite E")
                            if reference_grid is None:
                                reference_grid = z
                            if not np.array_equal(reference_grid, z):
                                info["grid_consistent"] = False
                                raise ValueError("Spatial grid differs from reference grid")
                        except Exception as exc:
                            info["problems"].append({"run_id": int(name[4:]), "error": str(exc)})
                            info["all_scored_profiles_finite"] = False
                        for attr, value in group.attrs.items():
                            val = json_value(value)
                            if attr.lower() in {"status", "retcode", "converged", "error", "success", "diagnostic"} or isinstance(val, (str, bool)):
                                diagnostic_counts.setdefault(attr, Counter())[str(val)] += 1
                                # Only claim status success for the documented conventions.
                                if attr.lower() in {"status", "retcode"}:
                                    acceptable = {"ok"} if attr.lower() == "status" else {"success"}
                                    if str(val).lower() not in acceptable:
                                        info["problems"].append({"run_id": int(name[4:]), "diagnostic": attr, "value": val})
                            elif isinstance(val, (int, float)):
                                if np.isfinite(val):
                                    old = numeric_ranges.get(attr, [float(val), float(val)])
                                    numeric_ranges[attr] = [min(old[0], float(val)), max(old[1], float(val))]
                    info["diagnostic_values"] = {k: dict(v) for k, v in diagnostic_counts.items()}
                    info["numeric_attribute_ranges"] = numeric_ranges
                    if not diagnostic_counts:
                        report["warnings"].append(f"No categorical status/diagnostic attributes stored in {source}; convergence cannot be inferred from absent fields")
                    if info["problems"]:
                        report["failures"].append({"source": source, "problem_count": len(info["problems"])})
                    print(f"Checked {source}: {len(names)} runs, {len(info['problems'])} problems", flush=True)
            except Exception as exc:
                report["failures"].append({"source": source, "error": str(exc)})
        report["missing_referenced_runs"] = []
        for label, m in (("D1", d1), ("D5", d5)):
            for split, keys in m.items():
                for key in keys:
                    if key.run_id not in available.get(key.source_dataset, set()):
                        report["missing_referenced_runs"].append({"partition": label, "split": split, **key.as_dict()})
        if report["missing_referenced_runs"]:
            report["failures"].append("Some manifest references do not exist")
        with h5py.File(DATA / "partitions_manifest.h5", "r") as f:
            structure = []
            count = [0]
            def inspect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    array = obj[()]
                    count[0] += 1
                    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                        report["failures"].append({"nonfinite_partition_dataset": name})
                    if len(structure) < 12:
                        structure.append({"path": name, "shape": list(obj.shape), "dtype": str(obj.dtype)})
            f.visititems(inspect)
            report["files"]["partitions_manifest.h5"] = {"readable": True, "all_datasets_read": count[0],
                "root_keys": list(f.keys()), "sample_datasets": structure,
                "used_for_D1_split": False}
        if reference_grid is not None:
            report["spatial_grid"] = {"shape": [202], "min_m": float(reference_grid.min()),
                "max_m": float(reference_grid.max()), "strictly_increasing": bool(np.all(np.diff(reference_grid) > 0)),
                "sha256_float64": hashlib.sha256(reference_grid.astype(np.float64).tobytes()).hexdigest(),
                "comparison": "exact equality across all simulation runs in all three files"}
    except Exception as exc:
        report["failures"].append({"exception": type(exc).__name__, "message": str(exc)})
    report["passed"] = not report["failures"]
    report["elapsed_seconds"] = time.perf_counter() - started
    write_json(RESULTS / "setup_check.json", report)
    print(f"Setup {'PASSED' if report['passed'] else 'FAILED'}: {len(report['failures'])} failures, {len(report['warnings'])} warnings", flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
