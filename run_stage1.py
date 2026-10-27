#!/usr/bin/env python3
"""Run stage one in order using the project venv, preserving failed commands."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import shlex
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from src.runtime import ROOT, RESULTS, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Verify/download releases and install requirements first")
    parser.add_argument("--skip-pilot", action="store_true", help="Run required checks, figure and D1 mean only")
    args = parser.parse_args()
    python = ROOT / ".venv/bin/python"
    if sys.version_info[:2] != (3, 11) or sys.prefix != str(ROOT / ".venv"):
        raise RuntimeError(f"Use the project interpreter: {python} -B run_stage1.py")
    steps = []
    if args.prepare:
        steps += [("obtain_releases", [str(python), "-B", "prepare_stage1.py"]),
                  ("install_dependencies", [str(python), "-B", "setup_environment.py"])]
    steps += [("setup_check", [str(python), "-B", "-m", "src.check_setup"]),
              ("example_plot", [str(python), "-B", "-m", "src.plot_one_run"]),
              ("mean_baseline_and_official_score", [str(python), "-B", "-m", "src.mean_profile_baseline"])]
    if not args.skip_pilot:
        steps += [("validation_pilot", [str(python), "-B", "-m", "src.run_pilot"])]
    steps += [("verify_saved_results", [str(python), "-B", "-m", "src.verify_results"])]
    run = {"started_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
           "project_root": str(ROOT), "argv": sys.argv, "steps": [], "passed": False,
           "pilot_requested": not args.skip_pilot}
    status = 0
    for name, cmd in steps:
        if name == "validation_pilot":
            gate = json.loads((RESULTS / "d1_mean_checks.json").read_text())
            if not gate["pilot_gate_passed"]:
                run["pilot_deferred_reason"] = gate["unresolved_discrepancy"]
                print(f"Pilot deferred: {run['pilot_deferred_reason']}", flush=True)
                break
        print(f"Running {name}: {shlex.join(cmd)}", flush=True)
        started = time.perf_counter()
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        entry = {"step": name, "argv": cmd, "command": shlex.join(cmd), "cwd": str(ROOT),
                 "exit_status": result.returncode, "seconds": time.perf_counter() - started,
                 "stdout": result.stdout, "stderr": result.stderr}
        run["steps"].append(entry)
        write_json(RESULTS / "stage1_run.json", run)
        with (RESULTS / "command_history.jsonl").open("a") as f:
            f.write(json.dumps({"run_started_at": run["started_at"], **entry}) + "\n")
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", file=sys.stderr, flush=True)
        if result.returncode:
            status = result.returncode
            run["failure"] = f"{name} failed with exit status {result.returncode}"
            break
    run["passed"] = status == 0 and "pilot_deferred_reason" not in run
    run["exit_status"] = status
    run["finished_at"] = datetime.now(ZoneInfo("America/New_York")).isoformat()
    write_json(RESULTS / "stage1_run.json", run)
    from src.write_summary import main as write_summary
    write_summary()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
