#!/usr/bin/env python3
"""Download checksum-verified official releases inside DM_project only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import urllib.request

from src.runtime import ROOT, RESULTS, BENCHMARK, DATA, inside, write_json


def download(file: dict, target: Path) -> dict:
    inside(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    algorithm, expected = file["checksum"].split(":", 1)

    def checksum(p: Path) -> str:
        h = hashlib.new(algorithm)
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024**2), b""):
                h.update(chunk)
        return h.hexdigest()

    if target.exists():
        if target.stat().st_size == file["size"] and checksum(target) == expected:
            print(f"Verified existing {target.relative_to(ROOT)}", flush=True)
            return {"path": str(target.relative_to(ROOT)), "release_filename": file["key"], "url": file["links"]["self"],
                    "bytes": file["size"], "checksum": file["checksum"], "verified": True}
        raise RuntimeError(f"Existing file is incomplete or has wrong checksum; preserve and inspect: {target}")
    partial = inside(target.with_name(target.name + ".part"))
    # A partial transfer may be restarted, but never replace a complete final file.
    for attempt in range(1, 4):
        try:
            h = hashlib.new(algorithm)
            total = 0
            last = time.monotonic()
            req = urllib.request.Request(file["links"]["self"], headers={"User-Agent": "DM-project-stage1/1.0"})
            with urllib.request.urlopen(req, timeout=90) as r, partial.open("wb") as out:
                while chunk := r.read(8 * 1024**2):
                    out.write(chunk)
                    h.update(chunk)
                    total += len(chunk)
                    if time.monotonic() - last > 15:
                        print(f"Downloading {target.name}: {total / 1024**2:.1f}/{file['size'] / 1024**2:.1f} MiB", flush=True)
                        last = time.monotonic()
            if total != file["size"] or h.hexdigest() != expected:
                raise RuntimeError(f"Release checksum/size mismatch for {target.name}")
            partial.rename(target)
            print(f"Downloaded and verified {target.relative_to(ROOT)} ({total} bytes)", flush=True)
            return {"path": str(target.relative_to(ROOT)), "release_filename": file["key"], "url": file["links"]["self"],
                    "bytes": total, "checksum": file["checksum"], "verified": True}
        except Exception as exc:
            print(f"Download attempt {attempt} failed for {target.name}: {exc}", flush=True)
            if attempt == 3:
                raise
            time.sleep(2)


def record(number: str) -> dict:
    path = RESULTS / f"zenodo_{number}.json"
    if path.exists():
        return json.loads(path.read_text())
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{number}", timeout=60) as r:
        obj = json.load(r)
    write_json(path, obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-only", action="store_true")
    args = parser.parse_args()
    code = record("22758006")
    archive_file, = code["files"]
    archive = ROOT / "external" / archive_file["key"]
    code_download = download(archive_file, archive)
    if not BENCHMARK.exists():
        # Reject traversal, absolute paths, symlinks, hard links and special files.
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            prefixes = {Path(m.name).parts[0] for m in members}
            if len(prefixes) != 1:
                raise RuntimeError(f"Unexpected archive roots: {prefixes}")
            prefix, = prefixes
            staging = inside(ROOT / ".tmp/code_extract")
            staging.mkdir(exist_ok=True)
            for m in members:
                if Path(m.name).is_absolute() or ".." in Path(m.name).parts:
                    raise RuntimeError(f"Unsafe archive member: {m.name}")
                inside(staging / m.name)
                if not (m.isdir() or m.isfile()):
                    raise RuntimeError(f"Nonregular archive member rejected: {m.name}")
            tar.extractall(staging, members=members, filter="data")
            (staging / prefix).rename(BENCHMARK)
    # Prefer Git when available, then prove its tagged source matches the
    # checksum-verified assignment archive before attaching the Git metadata.
    repository_url = "https://github.com/gangryk/hallthruster-surrogate-benchmark.git"
    commit = None
    git_log = []
    if shutil.which("git"):
        if not (BENCHMARK / ".git").exists():
            checkout = inside(ROOT / ".tmp/benchmark_git_v1.0.0")
            if checkout.exists():
                raise RuntimeError(f"Inspect preserved incomplete Git checkout before retrying: {checkout}")
            cmd = ["git", "clone", "--depth", "1", "--branch", "v1.0.0", repository_url, str(checkout)]
            result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            git_log.append({"argv": cmd, "cwd": str(ROOT), "exit_status": result.returncode,
                            "stdout": result.stdout, "stderr": result.stderr})
            write_json(RESULTS / "benchmark_git_acquisition.json", git_log)
            result.check_returncode()
            for p in BENCHMARK.rglob("*"):
                if p.is_file():
                    other = checkout / p.relative_to(BENCHMARK)
                    if not other.is_file() or p.read_bytes() != other.read_bytes():
                        raise RuntimeError(f"Archive differs from Git tag: {p.relative_to(BENCHMARK)}")
            archived_names = {str(p.relative_to(BENCHMARK)) for p in BENCHMARK.rglob("*") if p.is_file()}
            tagged_names = {str(p.relative_to(checkout)) for p in checkout.rglob("*") if p.is_file() and ".git" not in p.relative_to(checkout).parts}
            if archived_names != tagged_names:
                raise RuntimeError("Archive and Git tag contain different file sets")
            (checkout / ".git").rename(BENCHMARK / ".git")
        verified_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=BENCHMARK, text=True).strip()
        git_dir = subprocess.check_output(["git", "rev-parse", "--absolute-git-dir"], cwd=BENCHMARK, text=True).strip()
        if Path(verified_root).resolve() != BENCHMARK.resolve() or not Path(git_dir).resolve().is_relative_to(ROOT):
            raise RuntimeError("Benchmark Git metadata is outside the project")
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BENCHMARK, text=True).strip()
        tag_commit = subprocess.check_output(["git", "rev-parse", "v1.0.0^{commit}"], cwd=BENCHMARK, text=True).strip()
        if commit != tag_commit:
            raise RuntimeError("Benchmark HEAD differs from verified v1.0.0 tag")
    official_hashes = {}
    for p in sorted(BENCHMARK.rglob("*")):
        if p.is_file() and ".git" not in p.relative_to(BENCHMARK).parts:
            official_hashes[str(p.relative_to(BENCHMARK))] = hashlib.sha256(p.read_bytes()).hexdigest()
    provenance = {
        "source_repository": "https://github.com/gangryk/hallthruster-surrogate-benchmark",
        "code_doi": "10.5281/zenodo.22758006", "release": code["metadata"].get("version"),
        "release_tag_url": "https://github.com/gangryk/hallthruster-surrogate-benchmark/releases/tag/v1.0.0",
        "obtained_from": "Zenodo versioned archive, byte-matched to Git tag v1.0.0" if commit else "Zenodo versioned archive",
        "commit": commit,
        "archive": code_download, "official_file_sha256": official_hashes,
    }
    provenance_path = RESULTS / "benchmark_provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if previous["official_file_sha256"] != official_hashes:
            raise RuntimeError("Official benchmark files changed since acquisition")
    write_json(provenance_path, provenance)
    if args.code_only:
        return
    data_record = record("21986782")
    downloads = []
    for file in data_record["files"]:
        name = file["key"]
        if name.startswith("spt100_"):
            target = DATA / "core" / name
        elif name.startswith("extrap_pool_"):
            target = DATA / "extrapolation_pools" / name
        elif name == "README.md":
            target = DATA / "RELEASE_README.md"
        elif name == "partitions_manifest.h5":
            target = DATA / name
        else:
            raise RuntimeError(f"Unrecognized released file: {name}")
        downloads.append(download(file, target))
        write_json(RESULTS / "download_checksums.json", downloads)


if __name__ == "__main__":
    main()
