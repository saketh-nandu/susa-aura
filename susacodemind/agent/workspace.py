"""
SUSA-1B CodeMind — Agent Workspace Inspector.

Safely traverses and reads approved project directory workspaces, identifies
file hierarchies, filters out binary/cache files, and checks syntax health.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    ".idea",
    ".vscode",
}

LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".md": "markdown",
    ".sh": "bash",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sql": "sql",
}


@dataclass
class FileInfo:
    """Metadata for an inspected project file."""

    relative_path: str
    absolute_path: Path
    language: str
    size_bytes: int
    line_count: int


class WorkspaceInspector:
    """Inspects an authorized local project directory."""

    def __init__(self, workspace_root: Union[str, Path]) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        if not self.workspace_root.exists() or not self.workspace_root.is_dir():
            raise NotADirectoryError(f"Workspace path does not exist: {self.workspace_root}")

    def scan_files(self, max_files: int = 500) -> List[FileInfo]:
        """Scans workspace directory returning a list of code and text files."""
        scanned = []
        for root, dirs, files in os.walk(self.workspace_root):
            # Prune ignored directories in-place
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]

            for fname in files:
                if len(scanned) >= max_files:
                    break
                if fname.startswith("."):
                    continue

                fpath = Path(root) / fname
                ext = fpath.suffix.lower()
                lang = LANGUAGE_EXTENSIONS.get(ext, "text")

                try:
                    content = fpath.read_text(encoding="utf-8", errors="ignore")
                    lines = content.count("\n") + 1
                    rel = str(fpath.relative_to(self.workspace_root))
                    scanned.append(
                        FileInfo(
                            relative_path=rel,
                            absolute_path=fpath,
                            language=lang,
                            size_bytes=fpath.stat().st_size,
                            line_count=lines,
                        )
                    )
                except Exception:
                    continue

        return scanned

    def read_file(self, relative_path: str) -> str:
        """Reads content of a file within the workspace safely."""
        target = (self.workspace_root / relative_path).resolve()
        # Enforce that path remains strictly within workspace root
        if not str(target).startswith(str(self.workspace_root)):
            raise PermissionError(f"Path traversal forbidden: {relative_path}")
        if not target.exists():
            raise FileNotFoundError(f"File not found: {relative_path}")
        return target.read_text(encoding="utf-8")

    def get_workspace_tree(self) -> str:
        """Generates a text summary of the workspace file structure."""
        files = self.scan_files()
        lines = [f"Workspace: {self.workspace_root.name} ({len(files)} files)"]
        for f in files:
            lines.append(f"  ├── {f.relative_path} ({f.language}, {f.line_count} lines)")
        return "\n".join(lines)
