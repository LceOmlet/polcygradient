#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MAX_BYTES = 3 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mirror external artifacts into the repository while excluding large "
            "files and recording a traceable manifest."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="External artifact root to snapshot.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        required=True,
        help="Git repository root where the tracked snapshot should be written.",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=Path("tracked_artifacts/external_artifacts"),
        help="Destination directory relative to --repo-root.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("tracked_artifacts/external_artifacts_manifest.json"),
        help="Manifest path relative to --repo-root.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="Copy only files whose size is <= this threshold.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def ensure_under_repo(repo_root: Path, relpath: Path) -> Path:
    full = (repo_root / relpath).resolve()
    repo = repo_root.resolve()
    if os.path.commonpath([str(repo), str(full)]) != str(repo):
        raise ValueError(f"path escapes repo root: {relpath}")
    return full


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    repo_root = args.repo_root.resolve()
    dest_root = ensure_under_repo(repo_root, args.dest_root)
    manifest_path = ensure_under_repo(repo_root, args.manifest_path)
    max_bytes = int(args.max_bytes)

    if not source_root.is_dir():
        raise SystemExit(f"missing source root: {source_root}")
    if not (repo_root / ".git").exists():
        raise SystemExit(f"missing git repo root: {repo_root}")

    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    copied_entries: list[dict[str, object]] = []
    excluded_entries: list[dict[str, object]] = []

    all_files = sorted(
        [path for path in source_root.rglob("*") if path.is_file()],
        key=lambda path: path.relative_to(source_root).as_posix(),
    )

    for source_path in all_files:
        relpath = source_path.relative_to(source_root)
        size_bytes = source_path.stat().st_size
        sha256 = sha256_file(source_path)
        entry = {
            "source_relpath": relpath.as_posix(),
            "source_abspath": str(source_path),
            "size_bytes": size_bytes,
            "sha256": sha256,
            "mtime_epoch": source_path.stat().st_mtime,
        }
        if size_bytes <= max_bytes:
            dest_path = dest_root / relpath
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)
            entry["repo_relpath"] = dest_path.relative_to(repo_root).as_posix()
            copied_entries.append(entry)
        else:
            entry["excluded_reason"] = f"size_gt_{max_bytes}"
            excluded_entries.append(entry)

    manifest = {
        "snapshot_created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "repo_root": str(repo_root),
        "repo_head": git_head(repo_root),
        "dest_root": dest_root.relative_to(repo_root).as_posix(),
        "manifest_relpath": manifest_path.relative_to(repo_root).as_posix(),
        "max_bytes": max_bytes,
        "copied_count": len(copied_entries),
        "copied_total_bytes": sum(int(item["size_bytes"]) for item in copied_entries),
        "excluded_count": len(excluded_entries),
        "excluded_total_bytes": sum(int(item["size_bytes"]) for item in excluded_entries),
        "copied": copied_entries,
        "excluded": excluded_entries,
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
