#!/usr/bin/env python3
"""
SUSA-1B CodeMind — Real Dataset Acquisition & Ingestion Tool.

Acquires clean, zero-cost, permitted code datasets for pretraining and instruction tuning:
1. Python Standard Library (`stdlib`): Hundreds of PEP8-compliant Python modules from active environment.
2. GitHub Popular Repositories (`github`): TheAlgorithms/Python, psf/requests, pallets/flask, pallets/click.
3. Hugging Face Stream (`huggingface`): Streams thousands of real Python files from bigcode/the-stack-smol.
4. Curated & Online Coding SFT Instructions (`instructions`): Real coding tasks and solutions.

Usage:
    python scripts/download_dataset.py --source stdlib --output-dir data/raw_code
    python scripts/download_dataset.py --source github --output-dir data/raw_code
    python scripts/download_dataset.py --source huggingface --hf-samples 2000 --output-dir data/raw_code
    python scripts/download_dataset.py --source all --output-dir data/raw_code
    python scripts/download_dataset.py --generate-instructions --instructions-file data/coding_instructions.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


CURATED_INSTRUCTIONS = [
    {
        "instruction": "Write a Python function `quicksort(arr: list[int]) -> list[int]` that sorts an array using the quicksort algorithm.",
        "response": 'def quicksort(arr: list[int]) -> list[int]:\n    """Sorts an integer list using recursive quicksort."""\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n',
        "system_prompt": "You are SUSA CodeMind, an expert software engineer and AI coding assistant."
    },
    {
        "instruction": "Implement a Least Recently Used (LRU) Cache in Python using `collections.OrderedDict`.",
        "response": 'from collections import OrderedDict\nfrom typing import Any\n\nclass LRUCache:\n    """Capacity-bounded Least Recently Used (LRU) Cache."""\n    def __init__(self, capacity: int):\n        if capacity <= 0:\n            raise ValueError("Capacity must be positive.")\n        self.capacity = capacity\n        self.cache: OrderedDict[str, Any] = OrderedDict()\n\n    def get(self, key: str) -> Any:\n        if key not in self.cache:\n            return None\n        self.cache.move_to_end(key)\n        return self.cache[key]\n\n    def put(self, key: str, value: Any) -> None:\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.capacity:\n            self.cache.popitem(last=False)\n',
        "system_prompt": "You are SUSA CodeMind, an expert software engineer and AI coding assistant."
    },
    {
        "instruction": "Write a Python decorator `timed` that measures and prints function execution duration in milliseconds.",
        "response": 'import time\nfrom functools import wraps\nfrom typing import Callable, Any\n\ndef timed(func: Callable) -> Callable:\n    """Decorator measuring runtime execution duration in milliseconds."""\n    @wraps(func)\n    def wrapper(*args: Any, **kwargs: Any) -> Any:\n        start_time = time.perf_counter()\n        result = func(*args, **kwargs)\n        elapsed_ms = (time.perf_counter() - start_time) * 1000.0\n        print(f"[{func.__name__}] executed in {elapsed_ms:.2f} ms")\n        return result\n    return wrapper\n',
        "system_prompt": "You are SUSA CodeMind, an expert software engineer and AI coding assistant."
    },
    {
        "instruction": "Write a function `binary_search(arr: list[int], target: int) -> int` that returns the index of target or -1 if not found.",
        "response": 'def binary_search(arr: list[int], target: int) -> int:\n    """Finds target in a sorted list using binary search. Returns index or -1."""\n    low = 0\n    high = len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1\n',
        "system_prompt": "You are SUSA CodeMind, an expert software engineer and AI coding assistant."
    },
    {
        "instruction": "Implement a thread-safe Singleton pattern in Python.",
        "response": 'import threading\nfrom typing import Any\n\nclass SingletonMeta(type):\n    """Thread-safe implementation of the Singleton design pattern."""\n    _instances: dict[type, Any] = {}\n    _lock: threading.Lock = threading.Lock()\n\n    def __call__(cls, *args: Any, **kwargs: Any) -> Any:\n        with cls._lock:\n            if cls not in cls._instances:\n                instance = super().__call__(*args, **kwargs)\n                cls._instances[cls] = instance\n        return cls._instances[cls]\n',
        "system_prompt": "You are SUSA CodeMind, an expert software engineer and AI coding assistant."
    },
    {
        "instruction": "Write a Python function `is_valid_email(email: str) -> bool` using regex validation.",
        "response": 'import re\n\n_EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$")\n\ndef is_valid_email(email: str) -> bool:\n    """Validates whether email string conforms to standard format."""\n    if not email or len(email) > 254:\n        return False\n    return bool(_EMAIL_REGEX.match(email))\n',
        "system_prompt": "You are SUSA CodeMind, an expert software engineer and AI coding assistant."
    },
    {
        "instruction": "Write a function `fibonacci(n: int) -> int` using dynamic programming with O(1) space.",
        "response": 'def fibonacci(n: int) -> int:\n    """Computes the n-th Fibonacci number in O(n) time and O(1) space."""\n    if n < 0:\n        raise ValueError("n must be non-negative.")\n    if n in (0, 1):\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n',
        "system_prompt": "You are SUSA CodeMind, an expert software engineer and AI coding assistant."
    },
    {
        "instruction": "Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges overlapping intervals.",
        "response": 'def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n    """Merges all overlapping intervals in an input list."""\n    if not intervals:\n        return []\n    sorted_intervals = sorted(intervals, key=lambda x: x[0])\n    merged: list[list[int]] = [sorted_intervals[0]]\n    for current in sorted_intervals[1:]:\n        prev = merged[-1]\n        if current[0] <= prev[1]:\n            prev[1] = max(prev[1], current[1])\n        else:\n            merged.append(current)\n    return merged\n',
        "system_prompt": "You are SUSA CodeMind, an expert software engineer and AI coding assistant."
    },
    {
        "instruction": "Write a function `two_sum(nums: list[int], target: int) -> tuple[int, int] | None` using a hash map for O(n) time.",
        "response": 'def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:\n    """Finds two indices such that nums[i] + nums[j] == target."""\n    seen: dict[int, int] = {}\n    for idx, num in enumerate(nums):\n        diff = target - num\n        if diff in seen:\n            return (seen[diff], idx)\n        seen[num] = idx\n    return None\n',
        "system_prompt": "You are SUSA CodeMind, an expert software engineer and AI coding assistant."
    },
    {
        "instruction": "Implement a Trie (Prefix Tree) in Python with `insert`, `search`, and `starts_with` methods.",
        "response": 'class TrieNode:\n    def __init__(self):\n        self.children: dict[str, TrieNode] = {}\n        self.is_end_of_word: bool = False\n\nclass Trie:\n    """Prefix tree for efficient string dictionary search."""\n    def __init__(self):\n        self.root = TrieNode()\n\n    def insert(self, word: str) -> None:\n        node = self.root\n        for char in word:\n            if char not in node.children:\n                node.children[char] = TrieNode()\n            node = node.children[char]\n        node.is_end_of_word = True\n\n    def search(self, word: str) -> bool:\n        node = self.root\n        for char in word:\n            if char not in node.children:\n                return False\n            node = node.children[char]\n        return node.is_end_of_word\n\n    def starts_with(self, prefix: str) -> bool:\n        node = self.root\n        for char in prefix:\n            if char not in node.children:\n                return False\n            node = node.children[char]\n        return True\n',
        "system_prompt": "You are SUSA CodeMind, an expert software engineer and AI coding assistant."
    },
    {
        "instruction": "Write a Python function `flatten_nested_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict`.",
        "response": 'def flatten_nested_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:\n    """Flattens a deeply nested dictionary into dot-separated keys."""\n    items: list[tuple[str, Any]] = []\n    for k, v in d.items():\n        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)\n        if isinstance(v, dict):\n            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())\n        else:\n            items.append((new_key, v))\n    return dict(items)\n',
        "system_prompt": "You are SUSA CodeMind, an expert software engineer and AI coding assistant."
    },
    {
        "instruction": "Write a Python function to safely read a JSON file and return an empty dict on error.",
        "response": 'import json\nfrom pathlib import Path\nfrom typing import Any\n\ndef read_json_safely(file_path: str | Path) -> dict[str, Any]:\n    """Reads a JSON file safely, returning empty dictionary on failure."""\n    path = Path(file_path)\n    if not path.is_file():\n        return {}\n    try:\n        with path.open("r", encoding="utf-8") as f:\n            data = json.load(f)\n            return data if isinstance(data, dict) else {}\n    except (json.JSONDecodeError, OSError) as err:\n        print(f"Failed to read {file_path}: {err}")\n        return {}\n',
        "system_prompt": "You are SUSA CodeMind, an expert software engineer and AI coding assistant."
    }
]

POPULAR_REPOS = [
    ("TheAlgorithms", "https://github.com/TheAlgorithms/Python.git"),
    ("requests", "https://github.com/psf/requests.git"),
    ("flask", "https://github.com/pallets/flask.git"),
    ("click", "https://github.com/pallets/click.git"),
    ("fastapi", "https://github.com/tiangolo/fastapi.git"),
    ("jinja", "https://github.com/pallets/jinja.git"),
    ("rich", "https://github.com/Textualize/rich.git"),
    ("python-patterns", "https://github.com/faif/python-patterns.git"),
]


def extract_stdlib_code(target_dir: Path) -> int:
    """Copies clean, non-test standard library Python modules to target directory."""
    stdlib_path = Path(sysconfig.get_path("stdlib"))
    print(f"  [stdlib] Scanning Python standard library from: {stdlib_path}...")
    dest = target_dir / "stdlib"
    dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    for p in stdlib_path.glob("**/*.py"):
        parts_lower = [part.lower() for part in p.parts]
        if any(skip in parts_lower for skip in ("test", "tests", "idle_test", "__pycache__", "site-packages", "dist-packages")):
            continue
        try:
            rel = p.relative_to(stdlib_path)
            out_file = dest / rel
            out_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out_file)
            copied += 1
        except Exception:
            continue

    print(f"  [stdlib] Copied {copied} clean standard library Python files to {dest}")
    return copied


def clone_popular_repos(target_dir: Path) -> int:
    """Clones popular open-source Python repositories for high-quality pretraining code."""
    repos_dir = target_dir / "github_repos"
    repos_dir.mkdir(parents=True, exist_ok=True)
    total_files = 0

    for name, url in POPULAR_REPOS:
        repo_dest = repos_dir / name
        if not repo_dest.exists():
            print(f"  [GitHub] Cloning {name} from {url}...")
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", url, str(repo_dest)],
                    check=True,
                    capture_output=True,
                )
                git_dir = repo_dest / ".git"
                if git_dir.exists():
                    shutil.rmtree(git_dir, ignore_errors=True)
            except Exception as e:
                print(f"  [GitHub] Warning: Cloning {name} failed: {e}")
                continue
        else:
            print(f"  [GitHub] {name} already exists at {repo_dest}")

        py_count = len(list(repo_dest.glob("**/*.py")))
        print(f"        Found {py_count} Python files in {name}")
        total_files += py_count

    return total_files


def fetch_huggingface_code(target_dir: Path, max_samples: int = 1500) -> int:
    """Streams permissive Python code samples from Hugging Face bigcode/the-stack-smol."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [HuggingFace] `datasets` library not installed. Run `pip install datasets` to enable streaming.")
        return 0

    dest = target_dir / "huggingface_stack"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  [HuggingFace] Streaming up to {max_samples:,} Python code files from `bigcode/the-stack-smol`...")

    count = 0
    try:
        ds = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train", streaming=True)
        for idx, item in enumerate(ds):
            if count >= max_samples:
                break
            content = item.get("content", "")
            if len(content.strip()) > 50:
                file_path = dest / f"hf_sample_{count}.py"
                file_path.write_text(content, encoding="utf-8", errors="ignore")
                count += 1
                if count % 250 == 0:
                    print(f"        Streamed {count}/{max_samples} files from Hugging Face...")
    except Exception as e:
        print(f"  [HuggingFace] Note: Stream completed or ended ({e}). Kept {count} files.")

    print(f"  [HuggingFace] Ingested {count} real Python files to {dest}")
    return count


def fetch_huggingface_instructions(output_file: Path, max_samples: int = 500) -> int:
    """Streams real coding instruction pairs from Hugging Face."""
    try:
        from datasets import load_dataset
    except ImportError:
        return 0

    print(f"  [HuggingFace SFT] Streaming up to {max_samples} coding instruction pairs...")
    count = 0
    instructions = []
    try:
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train", streaming=True)
        for item in ds:
            if count >= max_samples:
                break
            inst = item.get("instruction", "")
            inp = item.get("input", "")
            out = item.get("output", "")
            if inst and out:
                full_inst = f"{inst}\n{inp}".strip() if inp else inst
                instructions.append({
                    "instruction": full_inst,
                    "response": out,
                    "system_prompt": "You are SUSA CodeMind, an expert software engineer and AI coding assistant."
                })
                count += 1
    except Exception:
        pass

    if instructions:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("a", encoding="utf-8") as f:
            for item in instructions:
                f.write(json.dumps(item) + "\n")
        print(f"  [HuggingFace SFT] Added {count} real coding instructions to {output_file}")
    return count


def write_coding_instructions(output_file: Path) -> int:
    """Writes curated coding instructions to a JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for item in CURATED_INSTRUCTIONS:
            f.write(json.dumps(item) + "\n")
    print(f"  [instructions] Exported {len(CURATED_INSTRUCTIONS)} curated base coding tasks to {output_file}")
    return len(CURATED_INSTRUCTIONS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Acquire and ingest clean code datasets for SUSA-1B CodeMind.")
    parser.add_argument(
        "--source",
        choices=["stdlib", "github", "huggingface", "all"],
        default="all",
        help="Dataset source to acquire (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw_code",
        help="Target directory to store code files.",
    )
    parser.add_argument(
        "--hf-samples",
        type=int,
        default=5000,
        help="Number of Python samples to stream from Hugging Face (default: 5000).",
    )
    parser.add_argument(
        "--generate-instructions",
        action="store_true",
        help="Generate curated coding SFT instruction dataset.",
    )
    parser.add_argument(
        "--instructions-file",
        type=str,
        default="data/coding_instructions.jsonl",
        help="Path for generated instruction dataset.",
    )

    args = parser.parse_args()
    print("=" * 70)
    print("  SUSA-1B CODEMIND — DATASET ACQUISITION ENGINE")
    print("=" * 70)

    out_path = Path(args.output_dir)
    total_files = 0

    if args.source in ("stdlib", "all"):
        total_files += extract_stdlib_code(out_path)

    if args.source in ("github", "all"):
        total_files += clone_popular_repos(out_path)

    if args.source in ("huggingface", "all"):
        total_files += fetch_huggingface_code(out_path, max_samples=args.hf_samples)

    if args.generate_instructions or not Path(args.instructions_file).exists():
        write_coding_instructions(Path(args.instructions_file))
        fetch_huggingface_instructions(Path(args.instructions_file), max_samples=1000)

    print("\n" + "=" * 70)
    print(f"  STATUS: DATASET ACQUISITION COMPLETE ({total_files:,} code files ready)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
