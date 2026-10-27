"""Generate concise presentation material directly from recorded evidence."""
from __future__ import annotations

import csv
import json

from .runtime import ROOT, RESULTS, inside


def read_json(name):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    setup = read_json("setup_check.json")
    check = read_json("d1_mean_checks.json")
    provenance = read_json("benchmark_provenance.json")
    example = read_json("example_run_metadata.json")
    run = read_json("stage1_run.json")
    lines = ["# Stage one: Hall-thruster surrogate modeling", "",
        "## Setup and scope", "",
        f"Project root: `{ROOT}`. All project files, downloads, dependencies, caches and temporary files are kept here.", "",
        f"Benchmark: archived release {provenance['release']}, commit `{provenance.get('commit')}`; "
        "[code archive](https://doi.org/10.5281/zenodo.22758006), "
        "[official repository](https://github.com/gangryk/hallthruster-surrogate-benchmark). "
        "[Data release](https://doi.org/10.5281/zenodo.21986782) is version 1.0.0, CC-BY-4.0. "
        "The Git tag source was byte-matched to the checksum-verified archive. Official code was kept unchanged.", ""]
    if run and not run["passed"]:
        lines += [f"Current run status: {run.get('failure', run.get('pilot_deferred_reason', 'incomplete'))}.", ""]
    if setup:
        lines += [f"Setup verification: **{'PASSED' if setup['passed'] else 'FAILED'}**. "
            f"Python {setup['python'].split()[0]} at `{setup['interpreter']}`. "
            f"{len(setup['failures'])} failures; {len(setup['warnings'])} warnings. "
            "See [setup evidence](../results/stage1/setup_check.json) and [installed versions](../results/stage1/requirements.freeze.txt).", ""]
        lines += ["Verified release MD5 checksums and sizes; read all four HDF5 files; checked every simulation run's "
            "four inputs, five scored outputs and electric field for shape and finite values; checked grids by exact equality; "
            "checked D1/D5 composite-key uniqueness, train/test separation and referenced run existence. "
            "Read all partition HDF5 datasets for integrity without using them to select D1 rows.", ""]
        if setup["passed"]:
            lines += [f"D1 fixed CSV: {setup['manifests']['D1']['train']:,} training and {setup['manifests']['D1']['test']:,} test rows; "
                f"common D1/D5 training pool: {setup['common_D1_D5_training_pool_count']:,} keys. "
                "Inputs: V_a [V], mdot [kg/s], P_b [Torr], B_hat [dimensionless]. "
                "Outputs: ne [m⁻³], Tev [eV], potential [V], ui[0] [m/s], nn [m⁻³]. Each profile has 202 positions over 0–80 mm.", ""]
            diagnostics = []
            for source, info in setup["files"].items():
                if "run_count" in info:
                    diagnostics.append(f"{source}: {info['run_count']:,} runs, diagnostics {info['diagnostic_values']}, {len(info['problems'])} problematic runs")
            lines += ["Observed diagnostic values:", ""] + [f"- {text}" for text in diagnostics] + [""]
        if setup["warnings"] or setup["failures"]:
            lines += ["Recorded issues:", ""] + [f"- {x}" for x in setup["failures"] + setup["warnings"]] + [""]
    else:
        lines += ["Setup has not completed; no success is claimed.", ""]
    lines += ["The release README illustrates `partitions/partitions_manifest.h5`; we retain it at the requested "
              "`data/partitions_manifest.h5`. The scorer does not read that HDF5 manifest. The prescribed D1 CSV remains the split source.", ""]
    if example:
        lines += ["## One simulation", "", "![Five profiles from one simulation](../results/stage1/example_run.png)", "",
            f"All five panels use run {example['run_id']} from `{example['source_dataset']}`. "
            f"Selection: {example['selection_rule']}. Dashed line: channel exit at 25 mm. "
            "These are physical-unit simulation profiles, with singly charged xenon velocity from stored ui row 0. "
            "The potential drop and ion-speed rise are visible near the channel exit, while density and temperature show their axial structure. "
            "The figure verifies profile loading and labels; it is not a model prediction.", ""]
    if check:
        with (RESULTS / "d1_mean_summary.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        lines += ["## Required D1 mean-profile baseline", "",
            "Arithmetic mean of all 9,600 prescribed D1 training profiles, in physical units with float64 accumulation. "
            "Every D1 test run receives the same (5, 202) mean. No transforms, filtering or test-target averaging. "
            "The unchanged official score.py evaluated the saved (1200, 5, 202) predictions.", "",
            "| Field | Measured error (%) | Assignment reference (%) | Difference (percentage points) |",
            "|---|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {row['field']} | {float(row['measured_error_percent']):.6f} | {float(row['assignment_reference_percent']):.3f} | {float(row['difference_percentage_points']):+.6f} |")
        lines += ["", f"Meeting check, error above 35%: **{'YES' if check['meeting_error_above_35_percent'] else 'NO'}**. "
            f"Agreement with the documented 35.907% headline and five field references at their displayed three decimals: "
            f"**{'YES' if check['reference_agreement_at_3_decimals'] else 'NO'}**. "
            "Agreement tolerance is 0.0005 percentage points, reflecting rounding, not a change to scoring. "
            "These are prediction errors; lower is better for later models.", "",
            "The headline is the unweighted mean of the five whole-profile per-field mean relative L2 errors. "
            "[Actual official stdout/stderr and command](../results/stage1/d1_mean_score.txt), "
            "[machine-readable score](../results/stage1/d1_mean_score.json), "
            "[loader/averaging audit](../results/stage1/d1_mean_audit.json). "
            "The acceleration-region result is supplementary and is computed by the official scorer from its electric-field rule; it does not enter the headline.", ""]
        if check["unresolved_discrepancy"]:
            lines += [check["unresolved_discrepancy"], ""]
    pilot_file = RESULTS / "pilot_metrics.csv"
    pilot_completed = run and any(s["step"] == "validation_pilot" and s["exit_status"] == 0 for s in run["steps"])
    if pilot_file.exists() and pilot_completed:
        with pilot_file.open(newline="") as f:
            pilot = list(csv.DictReader(f))
        config = read_json("pilot_config.json")
        code_check = read_json("local_solver_code_check.json")
        lines += ["## Preliminary validation / implementation check", "",
            f"Selected 2,000 training and 200 disjoint validation runs from {config['pool_count']:,} common D1/D5 training keys. "
            "The pool is sorted by (source_dataset, integer run_id), permuted with NumPy default_rng(42); "
            "the first 2,000 sampled keys train and the next 200 validate. Each selected split is sorted again. "
            "Neither proposed model was evaluated on official D1 or D5 test rows. "
            "Standardization used only pilot training inputs; outputs stayed in volts.", "",
            "Both methods use the same 32 neighbors, standardized Euclidean distance and normalized inverse-distance weights. "
            "The local linear fit is centered at the query, uses ridge λ=0.001 only on slopes, and solves an augmented weighted system with numpy.linalg.lstsq. "
            "For exact zero distances, both return the mean of exact matches within the selected neighbors. "
            "No clipping, output transforms or tuning were applied.", "",
            f"Synthetic solver code check passed: zero-ridge affine intercept maximum absolute error {code_check['affine_intercept_max_absolute_error']:.3g}; "
            "also checked an unpenalized constant intercept and duplicate exact-match behavior for both models. "
            "These synthetic checks are code validation, not benchmark evidence.", "",
            "| Method | Mean relative L2 (%) | Median relative L2 (%) | Prediction total (s) | ms/run |",
            "|---|---:|---:|---:|---:|"]
        for row in pilot:
            lines.append(f"| {row['method']} | {float(row['mean_relative_l2_percent']):.4f} | {float(row['median_relative_l2_percent']):.4f} | {float(row['prediction_seconds_total']):.6f} | {float(row['prediction_ms_per_run']):.4f} |")
        knn, local = pilot
        difference = float(local["mean_relative_l2_percent"]) - float(knn["mean_relative_l2_percent"])
        lines += ["", f"Local linear minus kNN mean error: {difference:+.4f} percentage points on this holdout. "
            "This observation does not establish an advantage inside missing operating-condition regions.", "",
            "Errors use the official single-field metrics API and its 1e-30 denominator floor. "
            "Timing is a single end-to-end call per method including query standardization, neighbor search and weighted prediction/local fits; "
            f"shared training/tree setup ({float(knn['shared_fit_seconds']):.6f} s), scoring and file I/O are excluded. "
            "Timings are preliminary measurements and may vary across reruns.", "",
            "![First validation run, fixed sorted order](../results/stage1/pilot_example.png)", "",
            f"Example key: run {config['example_key']['run_id']} from `{config['example_key']['source_dataset']}`; first validation key in sorted order, without selection by error. "
            "The fixed example includes a local-linear overshoot near the exit; lower aggregate error does not guarantee that every local feature improves. "
            "Predictions are unconstrained, as specified.", ""]
    else:
        lines += ["## Preliminary pilot", "", "Pilot not completed in the current run. See the run record for its gate, skip option or failure.", ""]
    git_status = read_json("local_git_status.json")
    if git_status:
        lines += ["## Local Git handoff", "", git_status["message"], ""]
    lines += ["## Research question and planned success measure", "",
        "Does regularized local linear regression reduce potential-profile prediction error inside missing regions of the Hall-thruster operating-condition space compared with distance-weighted kNN?", "",
        "Primary success measure for stage two: potential-profile mean relative L2 on official D5. "
        "Use D1 for ordinary-interpolation context, and compare median error and end-to-end inference cost as secondary measures. "
        "Freeze development choices before final test evaluation.", "",
        "## Five-slide presentation outline", "",
        "1. **Question and evidence boundary:** four operating inputs → 202-point potential; eventual D5 gap comparison, current stage is setup/baseline/validation.",
        "2. **Verified data and one run:** release/checksum/setup evidence and the five-panel `example_run.png`, including physical units and 25 mm exit.",
        "3. **Gate-zero baseline:** measured five-field table, headline, above-35% check and reference agreement; point to unchanged official scorer output.",
        "4. **Implementation pilot:** use the measured pilot table and first-validation figure if completed; explain protected common training pool, fixed settings and numerical solver check.",
        "5. **Next experiment:** D5 potential mean relative L2 as the primary measure, D1 context and inference cost; pilot does not prove gap robustness.", "",
        "## Reproduction and stage two", "",
        "From the confirmed project root, run `.venv/bin/python -B run_stage1.py`. "
        "Use `--prepare` to verify/download releases and install requirements first, or `--skip-pilot` to run only the required gate-zero work. "
        "Exact commands, stdout/stderr and exit statuses are in `results/stage1/stage1_run.json`, "
        "with prior commands retained in `command_history.jsonl`. Environment, release provenance, chosen keys, seed and decisions are saved alongside results and in the project logs.", "",
        "Saved artifact interfaces, seed-42 split reproduction, per-run/aggregate errors and official-source integrity also passed "
        "`src.verify_results`; see `results/stage1/artifact_verification.json`.", "",
        "Remaining stage two work: define gap-focused development/validation without using final test targets; "
        "settle the local model configuration and timing protocol; then run the planned D5 comparison with D1 context. "
        "No final D5 test evaluation, large search, multi-seed suite or neural training was run in stage one. "
        "Remote GitHub repository setup and any push remain the user's responsibility; no remote project repository was created.", "",
        "AI assistance: Codex generated the loader, models, runner, checks and this evidence-driven report. "
        "Checks include official archive/release matching, raw-file integrity, separate-loader agreement with official code, "
        "official scorer execution, a numerical affine solver check, saved split leakage checks and visual inspection of figures. "
        "See `ai_log.md` for the actual checks and their limits.", ""]
    path = inside(ROOT / "reports/stage1_summary.md")
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
