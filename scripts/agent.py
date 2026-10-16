#!/usr/bin/env python3
"""
SUSA-1B CodeMind — Coding Agent Runtime CLI.

Usage:
    python scripts/agent.py --workspace . --inspect
    python scripts/agent.py --workspace . --propose-file example.py --content "print('hello world')"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from susacodemind.agent import CodeMindAgent


def main() -> int:
    parser = argparse.ArgumentParser(description="SUSA-1B CodeMind Safety-Constrained Coding Agent.")
    parser.add_argument("--workspace", type=str, default=".", help="Path to project directory.")
    parser.add_argument("--inspect", action="store_true", help="Print visual workspace tree.")
    parser.add_argument("--propose-file", type=str, default=None, help="Relative file path for proposed edit.")
    parser.add_argument("--content", type=str, default=None, help="New content to propose.")
    parser.add_argument("--apply", action="store_true", help="Authorize applying edit to disk.")

    args = parser.parse_args()

    agent = CodeMindAgent(workspace_root=args.workspace)

    print("=" * 70)
    print("  SUSA-1B CODEMIND — CODING AGENT RUNTIME")
    print("=" * 70)

    if args.inspect or (not args.propose_file and not args.content):
        tree = agent.inspect_workspace()
        print(tree)

    if args.propose_file and args.content is not None:
        print(f"\n  [PROPOSED DIFF FOR {args.propose_file}]:")
        diff = agent.propose_edit(args.propose_file, args.content)
        print(diff)

        if args.apply:
            agent.apply_edit(args.propose_file, args.content, user_approved=True)
            print(f"\n  [SUCCESS] Edit applied to {args.propose_file} (Backup created).")
        else:
            print("\n  [INFO] To apply this change, re-run with '--apply'.")

    print("\n" + agent.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
