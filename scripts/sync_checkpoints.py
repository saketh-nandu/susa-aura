#!/usr/bin/env python3
"""
SUSA-1B CodeMind — Checkpoint Synchronization & Export Manifest Utility.

Handles syncing active checkpoints between fast local runtime disk and persistent
Google Drive storage, generating an experiment manifest with Git commit provenance.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_git_commit_hash() -> str:
    """Retrieves current Git commit SHA if in a git repository."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return commit
    except Exception:
        return "uncommitted_or_non_git"


def sync_checkpoints(src_dir: str, dest_dir: str, manifest_name: str = "run_manifest.json") -> dict:
    """Copies checkpoint files from src_dir to dest_dir and creates an execution manifest."""
    src = Path(src_dir)
    dest = Path(dest_dir)

    if not src.exists():
        print(f"[ERROR] Source directory does not exist: {src}", file=sys.stderr)
        return {"status": "failed", "error": f"Source directory {src} does not exist"}

    dest.mkdir(parents=True, exist_ok=True)

    copied_files = []
    # Copy .pt checkpoint files and logs
    for item in src.glob("*.pt"):
        dest_file = dest / item.name
        # Copy only if file does not exist or has newer mtime / different size
        if not dest_file.exists() or item.stat().st_size != dest_file.stat().st_size:
            shutil.copy2(item, dest_file)
            copied_files.append(item.name)
            print(f"  [SYNC] Copied: {item.name} -> {dest_file}")

    # Build provenance manifest
    manifest_data = {
        "project": "susa-1b-codemind",
        "sync_time_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(),
        "source_dir": str(src.resolve()),
        "destination_dir": str(dest.resolve()),
        "copied_checkpoints": copied_files,
        "all_checkpoints_at_dest": [p.name for p in dest.glob("*.pt")],
    }

    manifest_file = dest / manifest_name
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"  [SYNC] Exported manifest to: {manifest_file}")
    return manifest_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync model checkpoints and export run manifest.")
    parser.add_argument("--src-dir", type=str, default="checkpoints", help="Source checkpoint directory.")
    parser.add_argument("--dest-dir", type=str, required=True, help="Destination directory (e.g. Google Drive path).")
    parser.add_argument("--manifest-name", type=str, default="run_manifest.json", help="Manifest filename.")

    args = parser.parse_args()

    print("=" * 70)
    print("  SUSA-1B CODEMIND — CHECKPOINT SYNC & EXPORT MANIFEST")
    print("=" * 70)

    res = sync_checkpoints(args.src_dir, args.dest_dir, args.manifest_name)
    if res.get("status") == "failed":
        return 1

    print("=" * 70)
    print(f"  [SUCCESS] Synced {len(res['copied_checkpoints'])} checkpoint(s) successfully.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
