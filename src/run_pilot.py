"""One fixed potential-profile pilot on protected common training keys."""
from __future__ import annotations

import csv
import json
import time

from .runtime import RESULTS, write_json
from .data_io import development_pool, load_runs, manifest, official_module
from .local_models import LocalProfileModels, numerical_checks

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    setup = json.loads((RESULTS / "setup_check.json").read_text())
    baseline = json.loads((RESULTS / "d1_mean_checks.json").read_text())
    if not setup["passed"] or not baseline["pilot_gate_passed"]:
        raise RuntimeError("Pilot deferred: setup/scoring gate did not pass")
    code_check = numerical_checks()
    write_json(RESULTS / "local_solver_code_check.json", code_check)
    pool = development_pool()
    if len(pool) < 2200:
        raise RuntimeError(f"Need 2200 suitable common training runs; only {len(pool)} available")
    order = np.random.default_rng(42).permutation(len(pool))[:2200]
    train_keys = sorted(pool[i] for i in order[:2000])
    validation_keys = sorted(pool[i] for i in order[2000:2200])
    assert len(train_keys) == 2000 and len(validation_keys) == 200
    assert not set(train_keys) & set(validation_keys)
    final_test = set(manifest("D1")["test"]) | set(manifest("D5")["test"])
    assert not (set(train_keys) | set(validation_keys)) & final_test
    split_rows = [{"split": split, "order_within_split": i, **key.as_dict()}
                  for split, keys in (("train", train_keys), ("validation", validation_keys))
                  for i, key in enumerate(keys)]
    with (RESULTS / "pilot_split.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(split_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(split_rows)
    # Read only the selected potential targets, not all five fields or test targets.
    train = load_runs(train_keys, fields=("potential",))
    validation = load_runs(validation_keys, fields=("potential",))
    started = time.perf_counter()
    model = LocalProfileModels(train.inputs, train.outputs[:, 0], k=32, ridge=1e-3)
    fit_seconds = time.perf_counter() - started
    metrics = official_module("metrics")
    predictions, errors, rows = {}, {}, []
    for method in ("knn", "local_linear"):
        started = time.perf_counter()
        pred = model.predict(validation.inputs, method)
        elapsed = time.perf_counter() - started
        predictions[method] = pred
        official_score = metrics.score(pred[:, None, :], validation.outputs, fields=["potential"], percent=True)
        errors[method] = metrics.relative_l2(pred[:, None, :], validation.outputs)[:, 0]
        rows.append({"method": method, "evidence": "preliminary validation / implementation check",
            "field": "potential", "n_train": 2000, "n_validation": 200, "seed": 42, "k": 32,
            "ridge_lambda": 1e-3 if method == "local_linear" else 0.,
            "mean_relative_l2_percent": official_score["per_field"]["potential"],
            "median_relative_l2_percent": official_score["per_field_median"]["potential"],
            "prediction_seconds_total": elapsed, "prediction_ms_per_run": elapsed / 200 * 1000,
            "shared_fit_seconds": fit_seconds})
    with (RESULTS / "pilot_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (RESULTS / "pilot_per_run_errors.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_dataset", "run_id", "knn_relative_l2", "local_linear_relative_l2"], lineterminator="\n")
        writer.writeheader()
        writer.writerows({**key.as_dict(), "knn_relative_l2": errors["knn"][i],
                         "local_linear_relative_l2": errors["local_linear"][i]} for i, key in enumerate(validation_keys))
    np.savez(RESULTS / "pilot_predictions.npz", run_ids=np.array([k.run_id for k in validation_keys], dtype=np.int64),
        source_datasets=np.array([k.source_dataset for k in validation_keys]), fields=np.array(["potential"]),
        truth=validation.outputs[:, 0], knn=predictions["knn"], local_linear=predictions["local_linear"], z=validation.z)
    key = validation_keys[0]
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="white")
    ax.plot(validation.z * 1e3, validation.outputs[0, 0], color="black", linewidth=2.2, label="True simulation")
    ax.plot(validation.z * 1e3, predictions["knn"][0], color="#1d71a3", linewidth=1.8, linestyle="--", label="Distance-weighted kNN (k=32)")
    ax.plot(validation.z * 1e3, predictions["local_linear"][0], color="#b95f14", linewidth=1.8, linestyle="-.", label="Local linear (k=32, λ=0.001)")
    ax.axvline(25, color="grey", linestyle=":", label="Channel exit (25 mm)")
    ax.set(xlabel="Axial position [mm]", ylabel="Potential [V]", xlim=(0, 80))
    ax.set_title(f"Preliminary validation / implementation check · run {key.run_id}\n{key.source_dataset}", fontsize=11)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "pilot_example.png", dpi=180, facecolor="white")
    plt.close(fig)
    write_json(RESULTS / "pilot_config.json", {"evidence": "preliminary validation / implementation check",
        "pool_definition": "official D1 train intersection official D5 train, joined by (source_dataset, run_id)",
        "pool_count": len(pool), "pool_order": "source_dataset lexicographic then integer run_id ascending",
        "sample_rule": "numpy default_rng(42).permutation(pool size)[:2200]; first 2000 train, last 200 validation; sort each selected split by composite key",
        "seed": 42, "n_train": 2000, "n_validation": 200, "official_test_overlap": 0,
        "input_standardization": "training-only per-feature mean and population std (ddof=0)",
        "input_means": model.mean.tolist(), "input_scales": model.scale.tolist(),
        "k": 32, "distance": "Euclidean in standardized input space; scipy cKDTree p=2, workers=1",
        "neighbor_ties": "returned neighbors sorted by (distance, training row index); cKDTree deterministic for fixed ordered data",
        "weights": "normalized inverse distance, sum to one",
        "exact_zero_rule": "both methods average exact-match neighbors among the same k and ignore other neighbors; no ridge fit in this case",
        "local_linear_objective": "sum_i w_i ||a+B(x_i-x_query)-y_i||^2 + 1e-3 ||B||_F^2; intercept unpenalized",
        "solver": "numpy.linalg.lstsq on augmented sqrt-weighted design, no explicit matrix inverse",
        "output_units": "V", "output_transformations": [],
        "metric_api": "official metrics.score(..., fields=['potential'], percent=True) and metrics.relative_l2",
        "denominator_floor": float(metrics.NORM_FLOOR),
        "timing_scope": "single end-to-end call per method, input transformation + neighbor search + weighting/local fits; excludes shared training/scoring/I/O; no timing repetitions or tuning",
        "thread_count": 1, "example_key": key.as_dict(), "example_rule": "first validation key in sorted composite-key order",
        "limitation": "Random validation holdout does not establish performance inside missing regions; reserve D5 test for stage two"})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
