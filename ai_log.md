# AI assistance record

Date: 2026-10-26 (America/New_York). Assistant: Codex in this workspace. No more specific backend model identifier is asserted.

The user supplied the stage-one specification and a setup update, asked for implementation and actual execution, and restricted all writes to the existing DM_project. Codex read both requests, inspected the empty workspace and enclosing Git state, located installed Python versions, queried the official repository and actual Zenodo release APIs, and implemented separate project code and documentation. No subagents were used.

AI-generated components: read-only loader and manifest checks; checksum-verified source/data acquisition; copied-venv setup; one-run plotting; float64 mean baseline and official scorer invocation; fixed kNN/local-linear pilot; solver checks; ordered runner; evidence-driven report and reproducibility documentation.

Verification evidence is saved, rather than substituting AI estimates:

- Official release file sizes and MD5 checksums, archived source/Git tag byte agreement, release/commit provenance, and unchanged source SHA-256 hashes.
- Exhaustive required data/grid/shape/finiteness/diagnostic checks and exact D1 counts with composite-key leakage/existence checks.
- Comparison of our loader with the official loader on fixed D1 training positions; independent fsum checks of mean arithmetic; saved-prediction compatibility with the actual official scorer.
- Actual scorer subprocess stdout, stderr and exit status. Measured scores remain separate from the assignment's reference numbers.
- Saved pilot split keys and final-test exclusions; numerical affine recovery at zero ridge on asymmetric offsets; constant-intercept and duplicate exact-match checks; official metric calls with its actual denominator treatment.
- The generated figures are inspected after saving for profile count, units, run metadata, channel-exit location, shape and legibility. Final inspection findings are appended after execution.

Scientific limitation: these checks validate implementation and a small random validation experiment. They do not independently validate simulator physics or establish missing-region robustness. No claim is made that AI assistance replaces scientific review, and no scores or tool/model versions are invented.

Execution outcomes: all four data files passed release checksums and HDF5 reads; all 16,800 runs had finite required arrays, identical 202-point grids and retcode=success; the required D1 scorer exited 0. Headline error was 35.907216109854346%, agreeing with 35.907% at documented precision. The pilot's mean potential errors were 3.9790031401059403% for kNN and 1.37585299836826% for local linear, with zero official-test overlap. The affine solver check's maximum intercept error was 5.329070518200751e-15.

Visual inspection: opened both saved PNGs. Confirmed the five output panels use run 0, show the correct physical units and 0–80 mm axial range, display all four inputs and source ID, and mark the channel exit at 25 mm. Tightened the example figure's unused title spacing after inspection. The pilot plot shows true potential and both predictions for the first sorted validation key, run 32, with readable legends and units. It visibly includes a local-linear overshoot near the exit; that limitation is retained and discussed, with no clipping or favorable-example substitution.

Final artifact review reproduced the saved pilot selection exactly and recomputed the saved per-run and aggregate errors with the official metric. The staged Git whitespace check initially flagged the CSV writer's default CRLF endings. Changed the writer to LF and normalized only those four saved CSV files, preserving all parsed values. Rechecked saved artifacts afterward; no model configuration, prediction or measurement was changed.
