"""Verify saved artifact interfaces, split reproducibility and reported errors."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .runtime import ROOT, RESULTS, BENCHMARK, write_json
from .data_io import FIELDS, RunKey, development_pool, manifest, official_module

import numpy as np


def main() -> None:
    evidence = {"passed": False, "checks": [], "artifacts": {}}
    baseline = json.loads((RESULTS / "d1_mean_checks.json").read_text())
    setup = json.loads((RESULTS / "setup_check.json").read_text())
    assert setup["passed"] and baseline["scorer_exit_status"] == 0
    with np.load(RESULTS / "d1_mean_predictions.npz", allow_pickle=False) as archive:
        test_keys = manifest("D1")["test"]
        np.testing.assert_array_equal(archive["run_ids"], [k.run_id for k in test_keys])
        np.testing.assert_array_equal(archive["fields"], FIELDS)
        assert archive["predictions"].shape == (1200, 5, 202)
        assert archive["predictions"].dtype == np.float64
        assert np.isfinite(archive["predictions"]).all()
        with np.load(RESULTS / "d1_training_mean.npz", allow_pickle=False) as means:
            expected = np.broadcast_to(means["mean_profiles"], (1200, 5, 202))
            np.testing.assert_array_equal(archive["predictions"], expected)
    evidence["checks"].append("Saved D1 IDs, field order, float64 shape/finiteness and identical training-mean predictions")
    required = ["setup_check.json", "example_run.png", "example_run_metadata.json",
                "d1_mean_predictions.npz", "d1_mean_score.txt", "d1_mean_score.json", "d1_mean_summary.csv"]
    config_path = RESULTS / "pilot_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        pool = development_pool()
        order = np.random.default_rng(42).permutation(len(pool))[:2200]
        expected = {"train": sorted(pool[i] for i in order[:2000]),
                    "validation": sorted(pool[i] for i in order[2000:2200])}
        with (RESULTS / "pilot_split.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        saved = {part: [RunKey(r["source_dataset"], int(r["run_id"])) for r in rows if r["split"] == part]
                 for part in expected}
        assert saved == expected
        assert not set(saved["train"]) & set(saved["validation"])
        official_tests = set(manifest("D1")["test"]) | set(manifest("D5")["test"])
        assert not (set(saved["train"]) | set(saved["validation"])) & official_tests
        with (RESULTS / "pilot_per_run_errors.csv").open(newline="") as f:
            errors_csv = list(csv.DictReader(f))
        with (RESULTS / "pilot_metrics.csv").open(newline="") as f:
            metric_rows = {r["method"]: r for r in csv.DictReader(f)}
        metrics = official_module("metrics")
        with np.load(RESULTS / "pilot_predictions.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["run_ids"], [k.run_id for k in saved["validation"]])
            np.testing.assert_array_equal(archive["source_datasets"], [k.source_dataset for k in saved["validation"]])
            np.testing.assert_array_equal(archive["fields"], ["potential"])
            assert archive["truth"].shape == (200, 202)
            for method in ("knn", "local_linear"):
                assert archive[method].shape == (200, 202)
                assert archive[method].dtype == np.float64 and np.isfinite(archive[method]).all()
                errors = metrics.relative_l2(archive[method][:, None, :], archive["truth"][:, None, :])[:, 0]
                np.testing.assert_array_equal(errors, [float(r[f"{method}_relative_l2"]) for r in errors_csv])
                np.testing.assert_allclose(100 * errors.mean(), float(metric_rows[method]["mean_relative_l2_percent"]), rtol=1e-14)
                np.testing.assert_allclose(100 * np.median(errors), float(metric_rows[method]["median_relative_l2_percent"]), rtol=1e-14)
        assert config["example_key"] == saved["validation"][0].as_dict()
        evidence["checks"].append("Saved pilot split reproduces seed 42 exactly with 2000/200 rows and no official-test overlap")
        evidence["checks"].append("Saved physical-unit pilot predictions reproduce per-run and aggregate errors via official metric")
        required += ["pilot_split.csv", "pilot_metrics.csv", "pilot_per_run_errors.csv",
                     "pilot_predictions.npz", "pilot_example.png", "local_solver_code_check.json"]
    provenance = json.loads((RESULTS / "benchmark_provenance.json").read_text())
    for relative, expected_hash in provenance["official_file_sha256"].items():
        assert hashlib.sha256((BENCHMARK / relative).read_bytes()).hexdigest() == expected_hash
    evidence["checks"].append("All archived official source files remain unchanged after experiments")
    for name in required:
        path = RESULTS / name
        assert path.is_file() and path.stat().st_size > 0
        evidence["artifacts"][name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    evidence["checks"].append("All required artifacts exist and have recorded sizes/SHA256")
    evidence["passed"] = True
    write_json(RESULTS / "artifact_verification.json", evidence)
    print("Saved artifact verification PASSED; pilot split/error artifacts and official source integrity checked")


if __name__ == "__main__":
    main()
