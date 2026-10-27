"""Our D1 physical-unit arithmetic mean; evaluation by unchanged official CLI."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys

from .runtime import ROOT, BENCHMARK, DATA, RESULTS, write_json
from .data_io import FIELDS, manifest, load_runs, official_module

import numpy as np

REFERENCES_PERCENT = {"ne": 76.695, "Tev": 30.673, "potential": 25.977,
                      "ui": 17.810, "nn": 28.381, "headline": 35.907}


def main() -> None:
    setup = json.loads((RESULTS / "setup_check.json").read_text())
    if not setup["passed"]:
        raise RuntimeError("Setup verification must pass before the baseline")
    split = manifest("D1")
    train_keys, test_keys = split["train"], split["test"]
    if (len(train_keys), len(test_keys)) != (9600, 1200):
        raise ValueError("Prescribed D1 counts not met")
    official = official_module("score")
    official_manifest = official.load_manifest("D1")
    for part in ("train", "test"):
        keys = split[part]
        if len({k.source_dataset for k in keys}) != 1:
            raise ValueError("Official scorer uses one source per split; cannot represent this manifest unambiguously")
        if [k.run_id for k in keys] != official_manifest[part] or keys[0].source_dataset != official_manifest[f"{part}_source"]:
            raise ValueError("Our composite-key manifest disagrees with official scorer")
    training = load_runs(train_keys)
    assert training.outputs.shape == (9600, 5, 202)
    # Check the separate loader against official code at fixed training positions.
    positions = [0, len(train_keys) // 2, len(train_keys) - 1]
    official_training = official.load_true_fields(DATA, train_keys[0].source_dataset, [train_keys[i].run_id for i in positions])
    np.testing.assert_array_equal(training.outputs[positions], official_training["fields"])
    np.testing.assert_array_equal(training.z, official_training["z"])
    means = np.mean(training.outputs, axis=0, dtype=np.float64)
    assert means.shape == (5, 202)
    # Independent arithmetic check: fsum at fixed positions for each field.
    for field in range(5):
        for spatial in (0, 100, 201):
            expected = math.fsum(training.outputs[:, field, spatial]) / 9600
            if not np.isclose(means[field, spatial], expected, rtol=1e-12, atol=1e-12):
                raise AssertionError("Float64 training mean arithmetic check failed")
    test_ids = np.array([k.run_id for k in test_keys], dtype=np.int64)
    if len(set(test_ids.tolist())) != len(test_ids):
        raise ValueError("Integer-only scorer IDs ambiguous for D1")
    predictions = np.broadcast_to(means, (1200, 5, 202)).copy()
    path = RESULTS / "d1_mean_predictions.npz"
    np.savez(path, run_ids=test_ids, predictions=predictions, fields=np.array(FIELDS))
    np.savez(RESULTS / "d1_training_mean.npz", mean_profiles=means, fields=np.array(FIELDS), z=training.z)
    np.testing.assert_array_equal(official.load_predictions(path, official_manifest["test"]), predictions)
    audit = {"training_shape": list(training.outputs.shape), "mean_shape": list(means.shape),
        "prediction_shape": list(predictions.shape), "dtype": str(means.dtype), "mean_axis": 0,
        "training_rows_used": len(train_keys), "test_rows_predicted": len(test_keys),
        "units": "original physical units", "transformations": [],
        "fields": list(FIELDS), "ids_dtype": str(test_ids.dtype),
        "loader_matches_official_at_training_positions": positions, "independent_fsum_check": True,
        "prediction_roundtrip_matches_official": True,
        "D1_manifest_sha256": hashlib.sha256((BENCHMARK / "benchmark/manifests/D1.csv").read_bytes()).hexdigest()}
    write_json(RESULTS / "d1_mean_audit.json", audit)
    cmd = [str(ROOT / ".venv/bin/python"), "-B", "benchmark/score.py", "--predictions", str(path),
           "--partition", "D1", "--json-out", str(RESULTS / "d1_mean_score.json")]
    completed = subprocess.run(cmd, cwd=BENCHMARK, capture_output=True, text=True)
    command_record = {"argv": cmd, "command": shlex.join(cmd), "cwd": str(BENCHMARK),
        "HTSB_DATA_DIR": str(DATA), "exit_status": completed.returncode,
        "stdout": completed.stdout, "stderr": completed.stderr}
    write_json(RESULTS / "d1_scorer_command.json", command_record)
    (RESULTS / "d1_mean_score.txt").write_text(
        f"Command: {shlex.join(cmd)}\nCWD: {BENCHMARK}\nHTSB_DATA_DIR: {DATA}\nExit status: {completed.returncode}\n\n"
        f"STDOUT\n{completed.stdout}\nSTDERR\n{completed.stderr}")
    print(completed.stdout, end="")
    print(completed.stderr, end="", file=sys.stderr)
    completed.check_returncode()
    score = json.loads((RESULTS / "d1_mean_score.json").read_text())["whole_profile"]
    measured = {**score["per_field"], "headline": score["field_avg"]}
    rows = []
    for field, reference in REFERENCES_PERCENT.items():
        difference = measured[field] - reference
        rows.append({"field": field, "measured_error_percent": measured[field],
            "assignment_reference_percent": reference, "difference_percentage_points": difference,
            "agrees_at_documented_3_decimals": abs(difference) <= 0.000500000001})
    with (RESULTS / "d1_mean_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    checks = {"headline_error_percent": score["field_avg"],
        "meeting_error_above_35_percent": score["field_avg"] > 35,
        "reference_headline_percent": REFERENCES_PERCENT["headline"],
        "headline_difference_percentage_points": score["field_avg"] - REFERENCES_PERCENT["headline"],
        "reference_agreement_at_3_decimals": all(r["agrees_at_documented_3_decimals"] for r in rows),
        "reference_tolerance_percentage_points": 0.0005,
        "scorer_exit_status": completed.returncode,
        "pilot_gate_passed": all(r["agrees_at_documented_3_decimals"] for r in rows),
        "unresolved_discrepancy": None}
    if not checks["pilot_gate_passed"]:
        checks["unresolved_discrepancy"] = "Reference mismatch after checks of composite/integer IDs, official loader agreement, grid, field order, units, shape, float64 averaging and archived version. Pilot deferred; predictions unaltered."
    write_json(RESULTS / "d1_mean_checks.json", checks)
    print(f"Above 35%: {checks['meeting_error_above_35_percent']}; agrees with documented reference: {checks['reference_agreement_at_3_decimals']}")


if __name__ == "__main__":
    main()
