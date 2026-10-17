"""
SUSA AURA — Reliable Checkpoint Persistence, Synchronization, and Discovery Engine.

Provides atomic checkpoint persistence, resilient Google Drive synchronization,
checkpoint manifest tracking, corruption detection, numerical step sorting,
and RAM safety to prevent OOM crashes during checkpoint serialization.
"""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


@dataclass
class CheckpointMetadata:
    """Metadata record for a saved checkpoint."""

    step: int
    filename: str
    size_bytes: int
    timestamp_utc: str
    validation_status: str  # "valid", "corrupted", "incomplete", "unverified"
    val_loss: Optional[float] = None
    train_loss: Optional[float] = None
    model_profile: Optional[str] = None
    sha256: Optional[str] = None


@dataclass
class CheckpointCandidate:
    """Discovered checkpoint candidate with validation status."""

    step: int
    path: Path
    size_bytes: int
    is_valid: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CheckpointManifest:
    """Manages manifest.json for persistent checkpoint discovery and integrity tracking."""

    def __init__(self, entries: Optional[List[CheckpointMetadata]] = None) -> None:
        self.entries: List[CheckpointMetadata] = entries or []

    @classmethod
    def load(cls, manifest_path: Union[str, Path]) -> "CheckpointManifest":
        p = Path(manifest_path)
        if not p.exists():
            return cls([])
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            raw_entries = data.get("checkpoints", [])
            entries = []
            for item in raw_entries:
                entries.append(
                    CheckpointMetadata(
                        step=item["step"],
                        filename=item["filename"],
                        size_bytes=item.get("size_bytes", 0),
                        timestamp_utc=item.get("timestamp_utc", ""),
                        validation_status=item.get("validation_status", "valid"),
                        val_loss=item.get("val_loss"),
                        train_loss=item.get("train_loss"),
                        model_profile=item.get("model_profile"),
                        sha256=item.get("sha256"),
                    )
                )
            return cls(entries)
        except Exception:
            return cls([])

    def save(self, manifest_path: Union[str, Path]) -> None:
        p = Path(manifest_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Sort entries by step ascending
        self.entries.sort(key=lambda x: x.step)
        data = {
            "project": "SUSA AURA",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "latest_step": self.entries[-1].step if self.entries else 0,
            "total_checkpoints": len(self.entries),
            "checkpoints": [asdict(e) for e in self.entries],
        }
        tmp_p = p.with_name(f".{p.name}.{os.getpid()}.{time.time_ns()}.tmp")
        with open(tmp_p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_p, p)

    def add_or_update(self, entry: CheckpointMetadata) -> None:
        self.entries = [e for e in self.entries if e.step != entry.step]
        self.entries.append(entry)
        self.entries.sort(key=lambda x: x.step)

    def get_latest_valid(self) -> Optional[CheckpointMetadata]:
        valid_entries = [e for e in self.entries if e.validation_status == "valid"]
        if not valid_entries:
            return None
        return max(valid_entries, key=lambda x: x.step)


def extract_step_from_filename(filename: str) -> Optional[int]:
    """Extracts integer global step from checkpoint filename.
    
    Handles patterns like:
      - checkpoint_step_21.pt -> 21
      - checkpoint_step_1.pt -> 1
      - aura_1b_step_100.pt -> 100
      - step_50.pt -> 50
    """
    m = re.search(r"(?:step_|checkpoint_|^)(\d+)(?:\.pt|\.pth|$)", filename)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def validate_checkpoint_file(
    path: Union[str, Path],
    expected_vocab_size: Optional[int] = None,
    expected_d_model: Optional[int] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Inspects a checkpoint file for completeness, tensor loadability, and architecture match."""
    p = Path(path)
    if not p.exists():
        return False, f"File does not exist: {p}", None
    if p.name.startswith(".") or p.suffix == ".tmp":
        return False, f"Ignored temporary file: {p.name}", None

    size = p.stat().st_size
    if size == 0:
        return False, f"File is empty (0 bytes): {p}", None

    try:
        # Load header and state dict to CPU
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        return False, f"Failed to load checkpoint tensors ({type(e).__name__}): {e}", None

    if not isinstance(ckpt, dict):
        return False, f"Checkpoint root is not a dictionary (got {type(ckpt).__name__})", None

    # Check required core keys
    for req_key in ["model_state_dict", "step"]:
        if req_key not in ckpt:
            return False, f"Missing required key '{req_key}' in checkpoint", None

    model_state = ckpt["model_state_dict"]
    if not isinstance(model_state, dict) or len(model_state) == 0:
        return False, "Checkpoint 'model_state_dict' is empty or invalid", None

    # Validate model config if present
    model_cfg = ckpt.get("model_config", {})
    if isinstance(model_cfg, dict):
        if expected_vocab_size is not None:
            actual_vocab = model_cfg.get("vocab_size")
            if actual_vocab is not None and actual_vocab != expected_vocab_size:
                return (
                    False,
                    f"Vocabulary size mismatch: expected {expected_vocab_size}, found {actual_vocab}",
                    ckpt,
                )
        if expected_d_model is not None:
            actual_d_model = model_cfg.get("d_model")
            if actual_d_model is not None and actual_d_model != expected_d_model:
                return (
                    False,
                    f"Model dimension mismatch: expected {expected_d_model}, found {actual_d_model}",
                    ckpt,
                )

    step = ckpt.get("step", 0)
    return True, f"Valid checkpoint (Step: {step}, {len(model_state)} layers, {size / (1024**2):.1f} MB)", ckpt


def sync_checkpoint_file(
    source_path: Union[str, Path],
    dest_dir: Union[str, Path],
    min_free_bytes: int = 1_073_741_824,  # 1 GiB default
) -> Tuple[bool, str, Optional[Path]]:
    """Atomically synchronizes a checkpoint file to a destination directory (e.g. Google Drive).
    
    Uses temporary file staging and atomic rename to guarantee partial writes
    cannot be mistaken for valid checkpoints during network or session interruptions.
    """
    src = Path(source_path)
    dst_dir = Path(dest_dir)

    if not src.exists():
        return False, f"Source file does not exist: {src}", None

    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        return False, f"Cannot create destination directory '{dst_dir}': {e}", None

    src_size = src.stat().st_size

    # Check available storage
    try:
        usage = shutil.disk_usage(dst_dir)
        if usage.free < src_size + min_free_bytes:
            return (
                False,
                f"Insufficient storage in destination {dst_dir}: {usage.free / (1024**3):.2f} GiB free, "
                f"required {((src_size + min_free_bytes) / (1024**3)):.2f} GiB.",
                None,
            )
    except Exception as e:
        # Some virtual filesystems do not support disk_usage; warn and continue
        pass

    dest_file = dst_dir / src.name
    # If destination already exists and matches size, avoid re-copying
    if dest_file.exists() and dest_file.stat().st_size == src_size:
        return True, f"Destination file already up to date: {dest_file}", dest_file

    tmp_file = dst_dir / f".{src.name}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        # Copy to temporary staged file
        shutil.copy2(src, tmp_file)
        # Flush to physical media
        with open(tmp_file, "rb") as f:
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename into authoritative path
        os.replace(tmp_file, dest_file)

        # Verification
        if not dest_file.exists() or dest_file.stat().st_size != src_size:
            return (
                False,
                f"Sync verification failed: size mismatch (src={src_size}, dst={dest_file.stat().st_size if dest_file.exists() else 0})",
                None,
            )
        return True, f"Successfully synchronized {src.name} to {dest_file}", dest_file
    except Exception as e:
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass
        return False, f"Synchronization failed: {e}", None


def clean_orphaned_tmp_files(directory: Union[str, Path], max_age_seconds: float = 300.0) -> int:
    """Cleans up leftover temporary files from interrupted writes or crashes."""
    d = Path(directory)
    if not d.exists():
        return 0
    now = time.time()
    cleaned = 0
    try:
        for p in d.iterdir():
            if p.is_file() and (p.name.startswith(".") and ".tmp" in p.name or p.name.endswith(".tmp")):
                try:
                    if now - p.stat().st_mtime > max_age_seconds:
                        p.unlink()
                        cleaned += 1
                except OSError:
                    pass
    except Exception:
        pass
    return cleaned


def find_best_checkpoint(
    search_dirs: List[Union[str, Path]],
    target_step: Optional[int] = None,
    explicit_path: Optional[Union[str, Path]] = None,
    expected_vocab_size: Optional[int] = None,
    expected_d_model: Optional[int] = None,
) -> Optional[CheckpointCandidate]:
    """Discovers and validates the highest numerical step checkpoint across specified directories.
    
    Guarantees:
      - Resumes from numerical max step, never alphabetical order.
      - Explicit user path is validated first; halts if invalid.
      - Corrupted candidates are skipped with informative errors rather than silently loading bad weights.
      - Drive and local directories are searched without duplicate preference.
    """
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            return CheckpointCandidate(
                step=-1,
                path=p,
                size_bytes=0,
                is_valid=False,
                error_message=f"Explicit checkpoint path not found: {p}",
            )
        is_valid, msg, ckpt = validate_checkpoint_file(
            p, expected_vocab_size=expected_vocab_size, expected_d_model=expected_d_model
        )
        step = ckpt.get("step", extract_step_from_filename(p.name) or 0) if ckpt else 0
        return CheckpointCandidate(
            step=step,
            path=p,
            size_bytes=p.stat().st_size,
            is_valid=is_valid,
            error_message=None if is_valid else msg,
            metadata=ckpt,
        )

    # Collect candidate files across search directories
    candidates: List[Tuple[int, Path]] = []
    seen_paths = set()

    for s_dir in search_dirs:
        d = Path(s_dir)
        if not d.exists() or not d.is_dir():
            continue
        clean_orphaned_tmp_files(d)
        for f in d.glob("*.pt"):
            if f.is_file() and not f.name.startswith(".") and not f.name.endswith(".tmp"):
                resolved = f.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                step = extract_step_from_filename(f.name)
                if step is not None:
                    if target_step is not None and step != target_step:
                        continue
                    candidates.append((step, f))

    if not candidates:
        return None

    # Sort descending by step number (HIGHEST FIRST)
    candidates.sort(key=lambda x: x[0], reverse=True)

    # Validate candidates from highest step down to find the highest valid one
    for step, path in candidates:
        is_valid, msg, ckpt = validate_checkpoint_file(
            path, expected_vocab_size=expected_vocab_size, expected_d_model=expected_d_model
        )
        if is_valid:
            return CheckpointCandidate(
                step=step,
                path=path,
                size_bytes=path.stat().st_size,
                is_valid=True,
                error_message=None,
                metadata=ckpt,
            )
        else:
            print(f"  [WARN] Checkpoint at step {step} ({path.name}) failed validation: {msg}. Checking older candidate...")

    # If all were invalid, return the highest candidate marked as invalid
    highest_step, highest_path = candidates[0]
    return CheckpointCandidate(
        step=highest_step,
        path=highest_path,
        size_bytes=highest_path.stat().st_size if highest_path.exists() else 0,
        is_valid=False,
        error_message="All discovered checkpoints failed integrity validation.",
    )


class CheckpointManager:
    """Authoritative checkpoint manager supporting atomic saves, immediate Drive sync,
    manifest tracking, pruning, and RAM safety."""

    def __init__(
        self,
        local_dir: Union[str, Path] = "checkpoints",
        sync_dir: Optional[Union[str, Path]] = None,
        keep_last: int = 3,
        min_free_bytes: int = 1_073_741_824,
    ) -> None:
        self.local_dir = Path(local_dir)
        self.sync_dir = Path(sync_dir) if sync_dir else None
        self.keep_last = max(2, keep_last)  # Never keep fewer than 2 valid checkpoints
        self.min_free_bytes = min_free_bytes

        self.local_dir.mkdir(parents=True, exist_ok=True)
        if self.sync_dir:
            try:
                self.sync_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        self.local_manifest = CheckpointManifest.load(self.local_dir / "manifest.json")
        self.sync_manifest = (
            CheckpointManifest.load(self.sync_dir / "manifest.json")
            if self.sync_dir and self.sync_dir.exists()
            else None
        )

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Optional[Any] = None,
        lr_scheduler: Optional[Any] = None,
        total_tokens_processed: int = 0,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        val_loss: Optional[float] = None,
        train_loss: Optional[float] = None,
        rng_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Path, Optional[Path]]:
        """Saves checkpoint atomically locally and immediately syncs to Drive."""
        # 1. RAM Protection: Garbage collect and empty CUDA cache before assembling state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        filename = f"checkpoint_step_{step}.pt"
        local_file = self.local_dir / filename
        tmp_local = self.local_dir / f".{filename}.{os.getpid()}.{time.time_ns()}.tmp"

        # Check local disk capacity
        try:
            usage = shutil.disk_usage(self.local_dir)
            if usage.free < self.min_free_bytes:
                raise RuntimeError(
                    f"Insufficient local disk space: {usage.free / (1024**3):.2f} GiB free, "
                    f"minimum required {self.min_free_bytes / (1024**3):.2f} GiB."
                )
        except Exception:
            pass

        state: Dict[str, Any] = {
            "step": step,
            "total_tokens_processed": total_tokens_processed,
            "model_state_dict": model.state_dict(),
            "model_config": model_config or (getattr(model, "config", None).to_dict() if hasattr(model, "config") else {}),
            "training_config": training_config or {},
            "val_loss": val_loss,
            "train_loss": train_loss,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        if optimizer is not None:
            state["optimizer_state_dict"] = optimizer.state_dict()
        if scaler is not None and hasattr(scaler, "state_dict"):
            state["scaler_state_dict"] = scaler.state_dict()
        if lr_scheduler is not None and hasattr(lr_scheduler, "state_dict"):
            state["scheduler_state_dict"] = lr_scheduler.state_dict()
        if rng_state is not None:
            state["rng_state"] = rng_state

        # 2. Atomic write to local file
        torch.save(state, tmp_local)
        with open(tmp_local, "rb") as f:
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_local, local_file)

        size_bytes = local_file.stat().st_size

        # Update local manifest
        meta = CheckpointMetadata(
            step=step,
            filename=filename,
            size_bytes=size_bytes,
            timestamp_utc=state["timestamp_utc"],
            validation_status="valid",
            val_loss=val_loss,
            train_loss=train_loss,
            model_profile=state["model_config"].get("profile_name") if isinstance(state["model_config"], dict) else None,
        )
        self.local_manifest.add_or_update(meta)
        self.local_manifest.save(self.local_dir / "manifest.json")

        # 3. Free state dict memory immediately
        del state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 4. Immediate Drive Synchronization
        synced_path: Optional[Path] = None
        if self.sync_dir and self.sync_dir.exists():
            success, msg, s_path = sync_checkpoint_file(
                local_file, self.sync_dir, min_free_bytes=self.min_free_bytes
            )
            if success and s_path:
                synced_path = s_path
                print(f"  [DRIVE SYNC ✓] Step {step} successfully synchronized to: {synced_path}")
                if self.sync_manifest is None:
                    self.sync_manifest = CheckpointManifest.load(self.sync_dir / "manifest.json")
                self.sync_manifest.add_or_update(meta)
                self.sync_manifest.save(self.sync_dir / "manifest.json")
            else:
                print(f"\n  [DRIVE SYNC WARNING] Failed to sync checkpoint step {step} to Drive:\n  {msg}\n")

        # 5. Prune old checkpoints beyond keep_last
        self._prune_checkpoints()

        return local_file, synced_path

    def _prune_checkpoints(self) -> None:
        """Prunes older checkpoints beyond keep_last while protecting at least the 2 newest valid checkpoints."""
        valid_local = [e for e in self.local_manifest.entries if e.validation_status == "valid"]
        if len(valid_local) > self.keep_last:
            valid_local.sort(key=lambda x: x.step)
            to_remove = valid_local[: -self.keep_last]
            for r in to_remove:
                # Remove locally
                loc_f = self.local_dir / r.filename
                if loc_f.exists():
                    try:
                        loc_f.unlink()
                    except OSError:
                        pass
                # Update manifest
                self.local_manifest.entries = [e for e in self.local_manifest.entries if e.step != r.step]
            self.local_manifest.save(self.local_dir / "manifest.json")
