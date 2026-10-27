# Hall-thruster surrogate modeling: stage one

Course project. Research question: **Does regularized local linear regression reduce potential-profile prediction error inside missing regions of the Hall-thruster operating-condition space compared with distance-weighted kNN?**

The intended primary measure is potential-profile mean relative L2 on D5; D1 supplies ordinary-interpolation context. Stage one performs data verification, a five-panel simulation plot, the prescribed five-output D1 mean baseline, and one small protected validation pilot. It does not evaluate the proposed models on D1/D5 test targets or perform final D5 evaluation.

## Project boundary and environment

The verified project root for this execution is `/home/jimmy/Desktop/DM_project`. Start every command here. All scripts validate the working directory and reject symlinks escaping the project. They set task-local temporary/cache paths, disable bytecode writes, use the noninteractive Matplotlib backend, and set numerical-library threads to one. They do not edit global settings or another Python environment.

Installed Python 3.11 was found at `/home/jimmy/miniconda3/envs/dl_project/bin/python` (3.11.15). It was read and used to create a copied local virtual environment; its installation was not modified. The experiment interpreter is **`/home/jimmy/Desktop/DM_project/.venv/bin/python`**. No uv download or system installation was needed.

For the existing prepared environment, reproduce all experiments with:

```bash
.venv/bin/python -B run_stage1.py
```

For initial preparation on this machine, the commands are:

```bash
# From DM_project, confirm the canonical existing directory before writes.
DM_PROJECT_ROOT="$(pwd -P)"
test "$DM_PROJECT_ROOT" = /home/jimmy/Desktop/DM_project || exit 1
export PYTHONDONTWRITEBYTECODE=1
# The Python setup scripts validate symlinks and create/set local runtime paths.
/home/jimmy/miniconda3/envs/dl_project/bin/python -B setup_environment.py --create-only
.venv/bin/python -B prepare_stage1.py --code-only
.venv/bin/python -B setup_environment.py
.venv/bin/python -B prepare_stage1.py
.venv/bin/python -B run_stage1.py
```

On another machine, use an available Python 3.11 path for the first command; do not change a system interpreter. The existing venv is inspected and reused, never blindly replaced. `setup_environment.py` invokes only its explicit `.venv/bin/python -m pip`, installs binary wheels and records the command and exit status in `results/stage1/dependency_install.txt`. The runtime sets `TMPDIR`, `TMP`, `TEMP` to `.tmp`; `PIP_CACHE_DIR` to `.cache/pip`; `MPLCONFIGDIR` to `.cache/matplotlib`; `XDG_CACHE_HOME` to `.cache`; and `UV_CACHE_DIR` to `.cache/uv`. No command repurposes HOME or CODEX_HOME.

Installed direct versions are the archived benchmark pins: NumPy 2.4.3, SciPy 1.17.1, h5py 3.16.0, pandas 3.0.1 and Matplotlib 3.10.8. The pilot uses SciPy's cKDTree and NumPy; no additional ML package is needed. All successfully installed versions, including transitive dependencies, are in `results/stage1/requirements.freeze.txt` and `environment.json`.

To verify/download releases and install requirements before rerunning experiments:

```bash
.venv/bin/python -B run_stage1.py --prepare
```

## Official code and released data

- Code: [official repository](https://github.com/gangryk/hallthruster-surrogate-benchmark), [Zenodo archived release](https://doi.org/10.5281/zenodo.22758006), release **v1.0.0**, commit **30d45b23b4fc444b95490a7d161d0f102da212e7**.
- Data: [Zenodo release 1.0.0](https://doi.org/10.5281/zenodo.21986782), simulation generator HallThruster.jl v0.18.7, dataset license CC-BY-4.0.

`prepare_stage1.py` uses the actual versioned Zenodo API metadata, checks file sizes and published MD5 checksums, and preserves complete verified downloads. Partial downloads stay beside their targets as `.part` files. An existing final file that fails verification is preserved and reported rather than overwritten. The code archive is extracted inside `.tmp` only after rejecting absolute/traversal paths, links and special files. When Git is available, its v1.0.0 checkout is byte-matched against the verified archive before attaching the checkout metadata. Default-branch contents are not assumed to match the assignment.

```text
external/hallthruster-surrogate-benchmark/  # separate official checkout, unchanged
data/RELEASE_README.md
data/core/spt100_gbohm_12k_nc3_s42_20260617.h5
data/extrapolation_pools/extrap_pool_D3_2400.h5
data/extrapolation_pools/extrap_pool_D4_2400.h5
data/partitions_manifest.h5
```

The data README illustrates the HDF5 manifest under `partitions/`; the benchmark README says the scorer does not use it and allows it elsewhere. We retain it at the requested `data/partitions_manifest.h5`. The **fixed official `benchmark/manifests/D1.csv`** selects all 9,600 training rows and 1,200 test rows. It is not reconstructed from the HDF5 manifest or a new random split.

Download and archive provenance, release API snapshots and verified checksums are saved in `results/stage1/benchmark_provenance.json`, `benchmark_git_acquisition.json`, `zenodo_*.json` and `download_checksums.json`. To obtain the same code independently with Git (after the same boundary/runtime checks), clone the official repository at tag v1.0.0 and verify `git rev-parse HEAD` equals the recorded commit. The provided preparation script also verifies agreement with the archived assignment source.

## Individual experiment commands

Run these from the project root with the local interpreter. Importing our runtime configures project-local paths before numerical libraries or subprocesses start.

```bash
.venv/bin/python -B -m src.check_setup
.venv/bin/python -B -m src.plot_one_run
.venv/bin/python -B -m src.mean_profile_baseline
.venv/bin/python -B -m src.run_pilot
.venv/bin/python -B -m src.verify_results
.venv/bin/python -B -m src.write_summary
```

`check_setup` reads raw HDF5 in read-only mode, verifies the four files, release checksums, composite-key uniqueness/separation/existence, finite values, input/output shapes, stored ui shape, exact grid equality, diagnostics and units. It writes `setup_check.json`, returning nonzero if a required check fails. No rows are silently filtered. Output order is ne, Tev, potential, ui[0], nn. Inputs use V, kg/s, Torr and a dimensionless magnetic scale; outputs use m⁻³, eV, V, m/s and m⁻³, respectively.

The example is the smallest composite key in the common D1/D5 training pool. Every panel uses the same run; axial position is in millimeters, with a 25 mm channel-exit marker. The rule, exact inputs and key are in `example_run_metadata.json`.

The baseline computes `training_outputs.mean(axis=0, dtype=float64)` over all prescribed D1 training rows in physical units, with no transforms. It saves the official `.npz` interface, checks compatibility with the official loader, then invokes the unchanged scorer using the local interpreter. The actual invocation uses the following arguments with the benchmark checkout as its working directory:

```text
/home/jimmy/Desktop/DM_project/.venv/bin/python -B benchmark/score.py \
  --predictions /home/jimmy/Desktop/DM_project/results/stage1/d1_mean_predictions.npz \
  --partition D1 \
  --json-out /home/jimmy/Desktop/DM_project/results/stage1/d1_mean_score.json
```

The runner sets `HTSB_DATA_DIR=/home/jimmy/Desktop/DM_project/data`. The baseline captures exact argv, cwd, stdout, stderr and exit status in `d1_scorer_command.json` and `d1_mean_score.txt`; a scorer failure fails the step. `d1_mean_summary.csv` keeps measured errors, assignment reference errors and percentage-point differences in separate columns. Reference agreement uses the rounding tolerance for three displayed decimals (0.0005 percentage points). The above-35% meeting check is reported independently. Lower prediction error is better for subsequent models.

## Fixed preliminary validation pilot

The runner performs the pilot only after loading and official scoring agree with the reference. Use `run_stage1.py --skip-pilot` for the required baseline alone. An unresolved reference discrepancy defers the pilot while preserving evidence.

The development pool is the composite-key intersection of official D1 and D5 training sets, ordered by source then integer run ID. With NumPy `default_rng(42)`, the first 2,200 permuted pool entries give 2,000 training and 200 validation runs; each selected split is then sorted. Keys are saved in `pilot_split.csv`, and checked against both official test sets. Only potential targets for these selected runs are loaded by the pilot.

Input means and population standard deviations are fitted on the 2,000 training runs only. Both methods use exactly the same k=32 neighbors in standardized Euclidean space and normalized inverse-distance weights. With any exact zero-distance neighbors among the selected k, both return their equal-weight mean. Otherwise kNN averages the profiles, and local linear fits `a + B(x_i-x_query)` by minimizing the normalized weighted squared residual plus `1e-3 * ||B||²`. The intercept is not penalized. `numpy.linalg.lstsq` solves the augmented weighted system; no inverse, clipping or output transformation is used.

A tiny asymmetric affine-recovery check with zero regularization distinguishes the regression solver from averaging. Constant-intercept and duplicate exact-match checks also run before interpreting results. These are synthetic **code checks**, not synthetic benchmark data. Metrics use official `metrics.score(..., fields=['potential'])` and `relative_l2`, including its 1e-30 denominator floor.

Each method's prediction timer includes input standardization, neighbor search, weights and local predictions/fits for all 200 queries. Shared model/tree construction is recorded separately; scoring and I/O are excluded. There is one configuration and one timed call per method, without tuning or repeated timing. The example plot is the first validation key in fixed sorted order. A random holdout is a **preliminary validation / implementation check** and does not demonstrate gap robustness.

## Outputs and provenance

- `reports/stage1_summary.md`: measured results, limitations, research question, success measure, five-slide outline and stage-two work.
- `results/stage1/example_run.png`, `pilot_example.png`: presentation-ready figures.
- `results/stage1/d1_mean_predictions.npz`, `d1_training_mean.npz`, `d1_mean_score.txt`, `d1_mean_score.json`, `d1_mean_summary.csv`, `d1_mean_audit.json`, `d1_mean_checks.json`: required baseline and evidence.
- `results/stage1/pilot_split.csv`, `pilot_metrics.csv`, `pilot_per_run_errors.csv`, `pilot_predictions.npz`, `pilot_config.json`, `local_solver_code_check.json`: fixed preliminary pilot and check evidence.
- `results/stage1/stage1_run.json`, `command_history.jsonl`: exact commands, outputs, exit statuses and run history. Errors are preserved; no successful execution is claimed without a successful command.
- `results/stage1/artifact_verification.json`: saved-array shapes/IDs/field order, exact split reproducibility, per-run/aggregate metric agreement, official-source integrity and result hashes.
- `decision_log.md`, `ai_log.md`: choices, actual AI assistance and checks.

No PowerPoint is created. Raw HDF5, `.tools`, `.runtime`, `.venv`, caches, temporary files and the official checkout are excluded from the project Git repository. Required prediction arrays are about 10 MB total and intentionally retained; the mean prediction archive compresses strongly in Git. A local commit is created only if existing Git identity and local permissions permit. No project remote is created and nothing is pushed; the user handles remote GitHub setup separately.

For this execution, the local commit is pending because Git `user.name` is missing. The existing email setting was left unchanged. After reviewing the staged files, supply your own name and commit locally:

```bash
git config --local user.name 'YOUR NAME'
git commit -m 'Implement and execute Hall-thruster stage-one experiments'
```
