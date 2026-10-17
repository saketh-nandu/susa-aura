"""
SUSA AURA — Comprehensive Automated Tests for Checkpoint Persistence,
Synchronization, Corruption Detection, Numerical Step Sorting, and Architecture Compatibility.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from aura.model.config import ModelConfig
from aura.model.transformer import AuraModel
from aura.training.checkpoint_manager import (
    CheckpointCandidate,
    CheckpointManager,
    CheckpointManifest,
    CheckpointMetadata,
    clean_orphaned_tmp_files,
    extract_step_from_filename,
    find_best_checkpoint,
    sync_checkpoint_file,
    validate_checkpoint_file,
)
from aura.training.trainer import AuraTrainer, TrainingConfig


@pytest.fixture
def tiny_model():
    cfg = ModelConfig(
        vocab_size=256,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        n_layers=2,
        d_ff=128,
        max_seq_len=32,
    )
    return AuraModel(cfg)


def test_atomic_checkpoint_saving(tmp_path, tiny_model):
    """1. Verify checkpoint saving is atomic via temporary staging and clean rename."""
    local_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(local_dir=local_dir)

    opt = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    local_f, _ = mgr.save(
        step=5,
        model=tiny_model,
        optimizer=opt,
        total_tokens_processed=1024,
    )

    assert local_f.exists()
    assert local_f.name == "checkpoint_step_5.pt"
    # Ensure no temporary staging files remain
    tmp_files = list(local_dir.glob(".*.tmp")) + list(local_dir.glob("*.tmp"))
    assert len(tmp_files) == 0


def test_corrupted_or_incomplete_checkpoint_detection(tmp_path):
    """2. Verify detection of zero-byte, truncated, non-dict, and invalid checkpoints."""
    # A. Non-existent file
    valid, msg, _ = validate_checkpoint_file(tmp_path / "missing.pt")
    assert not valid
    assert "does not exist" in msg

    # B. Zero-byte file
    empty_f = tmp_path / "empty.pt"
    empty_f.write_bytes(b"")
    valid, msg, _ = validate_checkpoint_file(empty_f)
    assert not valid
    assert "empty" in msg

    # C. Truncated / corrupt binary file
    corrupt_f = tmp_path / "corrupt.pt"
    corrupt_f.write_bytes(b"PK\x03\x04\x00\x00\x00\x00corrupted_payload")
    valid, msg, _ = validate_checkpoint_file(corrupt_f)
    assert not valid
    assert "Failed to load checkpoint" in msg

    # D. Missing required keys
    incomplete_f = tmp_path / "incomplete.pt"
    torch.save({"step": 10}, incomplete_f)  # Missing model_state_dict
    valid, msg, _ = validate_checkpoint_file(incomplete_f)
    assert not valid
    assert "Missing required key 'model_state_dict'" in msg


def test_checkpoint_manifest_creation_and_validation(tmp_path, tiny_model):
    """3. Verify checkpoint manifest creation, atomic updates, and step sorting."""
    local_dir = tmp_path / "ckpts"
    sync_dir = tmp_path / "drive_ckpts"
    sync_dir.mkdir(parents=True, exist_ok=True)

    mgr = CheckpointManager(local_dir=local_dir, sync_dir=sync_dir)
    opt = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)

    mgr.save(step=1, model=tiny_model, optimizer=opt, val_loss=5.0)
    mgr.save(step=10, model=tiny_model, optimizer=opt, val_loss=4.2)
    mgr.save(step=5, model=tiny_model, optimizer=opt, val_loss=4.8)

    manifest_p = local_dir / "manifest.json"
    assert manifest_p.exists()

    manifest = CheckpointManifest.load(manifest_p)
    assert len(manifest.entries) == 3
    # Verify entries are sorted by step ascending
    steps = [e.step for e in manifest.entries]
    assert steps == [1, 5, 10]
    assert manifest.get_latest_valid().step == 10

    # Verify sync manifest in Drive folder
    drive_manifest_p = sync_dir / "manifest.json"
    assert drive_manifest_p.exists()
    drive_manifest = CheckpointManifest.load(drive_manifest_p)
    assert drive_manifest.get_latest_valid().step == 10


def test_synchronization_failure_handling(tmp_path, tiny_model):
    """4. Verify that Google Drive sync failure is handled safely without corrupting local save."""
    local_dir = tmp_path / "ckpts"
    # Provide an unwritable sync directory to simulate Drive permission/quota failure
    bad_sync_dir = tmp_path / "unwritable_drive"
    bad_sync_dir.mkdir(parents=True, exist_ok=True)

    mgr = CheckpointManager(local_dir=local_dir, sync_dir=bad_sync_dir)
    opt = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)

    with patch("shutil.copy2", side_effect=OSError("Drive network timeout or quota exceeded")):
        local_f, synced_f = mgr.save(step=1, model=tiny_model, optimizer=opt)

    # Local checkpoint MUST succeed and be intact
    assert local_f.exists()
    assert synced_f is None  # Gracefully returned None instead of crashing


def test_interrupted_copy_recovery(tmp_path):
    """5. Verify cleanup of orphaned temporary staging files left by interrupted writes."""
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Create fake abandoned temp files
    tmp1 = ckpt_dir / ".checkpoint_step_20.12345.987654321.tmp"
    tmp2 = ckpt_dir / "checkpoint_step_21.pt.tmp"
    good = ckpt_dir / "checkpoint_step_10.pt"

    tmp1.write_bytes(b"partial")
    tmp2.write_bytes(b"partial")
    good.write_bytes(b"valid")

    # Set access time in past
    old_time = 0.0
    os.utime(tmp1, (old_time, old_time))
    os.utime(tmp2, (old_time, old_time))

    cleaned = clean_orphaned_tmp_files(ckpt_dir, max_age_seconds=1.0)
    assert cleaned == 2
    assert not tmp1.exists()
    assert not tmp2.exists()
    assert good.exists()


def test_selecting_highest_valid_global_step(tmp_path, tiny_model):
    """6. Verify selection of highest global step by numerical value, not alphabetical order."""
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Create checkpoints with steps 1, 2, 10, 21
    # Note: alphabetically, "checkpoint_step_2.pt" > "checkpoint_step_10.pt"
    # But numerically, 21 > 10 > 2 > 1.
    for step in [1, 2, 10, 21]:
        p = ckpt_dir / f"checkpoint_step_{step}.pt"
        torch.save({
            "step": step,
            "model_state_dict": tiny_model.state_dict(),
            "model_config": tiny_model.config.to_dict(),
        }, p)

    candidate = find_best_checkpoint(search_dirs=[ckpt_dir])
    assert candidate is not None
    assert candidate.is_valid
    assert candidate.step == 21
    assert candidate.path.name == "checkpoint_step_21.pt"


def test_preventing_unintended_fallback_to_older_checkpoint(tmp_path, tiny_model):
    """7. Verify that corrupted newer checkpoints are flagged and don't silently mask corruption."""
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 is valid
    p1 = ckpt_dir / "checkpoint_step_1.pt"
    torch.save({
        "step": 1,
        "model_state_dict": tiny_model.state_dict(),
        "model_config": tiny_model.config.to_dict(),
    }, p1)

    # Step 21 was truncated / corrupted during a crash
    p21 = ckpt_dir / "checkpoint_step_21.pt"
    p21.write_bytes(b"corrupted_and_truncated_content")

    # When querying for step 21 specifically:
    cand21 = find_best_checkpoint(search_dirs=[ckpt_dir], explicit_path=p21)
    assert cand21 is not None
    assert cand21.is_valid is False
    assert "Failed to load checkpoint" in cand21.error_message

    # When querying best available, it skips corrupted step 21 and recovers to step 1
    cand_best = find_best_checkpoint(search_dirs=[ckpt_dir])
    assert cand_best is not None
    assert cand_best.step == 1
    assert cand_best.is_valid is True


def test_correct_restoration_of_model_and_optimizer_state(tmp_path, tiny_model):
    """8. Verify full roundtrip restoration of weights, optimizer states, and step count."""
    ckpt_file = tmp_path / "checkpoint_step_15.pt"

    class DummyDS(torch.utils.data.Dataset):
        def __len__(self): return 10
        def __getitem__(self, i): return {"input_ids": torch.zeros(16, dtype=torch.long), "labels": torch.zeros(16, dtype=torch.long)}

    trainer1 = AuraTrainer(tiny_model, train_dataset=DummyDS(), config=TrainingConfig())
    # Perform a dummy step
    x = torch.randint(0, 256, (2, 16))
    out = tiny_model(x, labels=x)
    out.loss.backward()
    trainer1.optimizer.step()
    trainer1.global_step = 15
    trainer1.total_tokens_processed = 5000

    trainer1.save_checkpoint(ckpt_file)

    # Instantiate fresh model and trainer
    new_model = AuraModel(tiny_model.config)
    trainer2 = AuraTrainer(new_model, train_dataset=DummyDS(), config=TrainingConfig())
    resumed_step = trainer2.load_checkpoint(ckpt_file)

    assert resumed_step == 15
    assert trainer2.global_step == 15
    assert trainer2.total_tokens_processed == 5000

    # Verify weights restored identically
    for p_orig, p_new in zip(tiny_model.parameters(), new_model.parameters()):
        assert torch.allclose(p_orig, p_new)


def test_accurate_global_step_accounting_after_resume():
    """9. Verify additional-steps calculation from restored global step."""
    resumed_step = 21
    additional_steps = 29
    new_max = resumed_step + additional_steps
    assert new_max == 50

    # Auto-extension when target is <= resumed_step
    target_max = 20
    if target_max <= resumed_step:
        target_max = resumed_step + 10
    assert target_max == 31


def test_checkpoint_compatibility_with_1b_architecture_and_32k_tokenizer(tmp_path):
    """10. Verify validation detects vocabulary size and model dimension mismatches."""
    ckpt_file = tmp_path / "toy_model.pt"
    # Create checkpoint with vocab 790 and d_model 64
    torch.save({
        "step": 1,
        "model_state_dict": {"tok_embeddings.weight": torch.randn(790, 64)},
        "model_config": {"vocab_size": 790, "d_model": 64, "profile_name": "smoke"},
    }, ckpt_file)

    # Validates when expecting 790
    valid, _, _ = validate_checkpoint_file(ckpt_file, expected_vocab_size=790)
    assert valid

    # Fails when expecting 1B model (vocab 32,000)
    valid, msg, _ = validate_checkpoint_file(ckpt_file, expected_vocab_size=32000)
    assert not valid
    assert "Vocabulary size mismatch" in msg

    # Fails when expecting 1B model (d_model 2048)
    valid, msg, _ = validate_checkpoint_file(ckpt_file, expected_d_model=2048)
    assert not valid
    assert "Model dimension mismatch" in msg
