# SUSA AURA — Pre-Training & Fine-Tuning Guide

This guide documents the full training pipeline for SUSA AURA:
1. Dataset ingestion & sanitization
2. Tokenizer training
3. Pretraining with causal language modeling loss
4. Supervised Fine-Tuning (SFT) with prompt loss masking
5. Checkpoint resumption and export

---

## 1. Zero-Cost Training Philosophy

- **Local Development:** Tiny smoke models (`smoke.yaml`, ~158k params) are trained on local laptop CPU to verify pipeline logic and gradient backpropagation in seconds.
- **Scaling on Free Accelerators:** Larger profiles (`small`, `medium`, `target_1b`) are trained using Google Colab free tier GPU sessions (NVIDIA T4 / V100).
- **Crash Recovery:** Fast local NVMe writes are synced periodically to persistent Google Drive storage.

---

## 2. Step-by-Step Training Workflow

### Step 1: Ingest & Clean Raw Datasets
```bash
# Acquire Python standard library and generate curated instructions
python scripts/download_dataset.py --source stdlib --output-dir data/raw_code --generate-instructions
```

### Step 2: Train Byte-Level BPE Tokenizer
```bash
# Train tokenizer for smoke profile (1024 vocab) or target profile (32000 vocab)
python scripts/train_tokenizer.py --input-dir data/raw_code --vocab-size 1024 --output-dir tokenizer_cache/smoke
```

### Step 3: Shard Memory-Mapped Binary Dataset
```bash
python scripts/prepare_dataset.py --tokenizer-dir tokenizer_cache/smoke --input-dir data/raw_code --output-dir data
```

### Step 4: Launch Autoregressive Pre-Training
```bash
# Local CPU smoke training run
python scripts/train.py --config configs/smoke.yaml --train-bin data/train.bin --val-bin data/val.bin --max-steps 20

# Resuming from a checkpoint
python scripts/train.py --config configs/smoke.yaml --train-bin data/train.bin --val-bin data/val.bin --resume-from checkpoints/checkpoint_step_10.pt
```

### Step 5: Supervised Instruction Fine-Tuning (SFT)
```bash
# Fine-tune base checkpoint with prompt loss masking
python scripts/train_instruct.py \
    --base-checkpoint checkpoints/checkpoint_step_15.pt \
    --tokenizer-dir tokenizer_cache/smoke \
    --data-file data/coding_instructions.jsonl \
    --max-steps 15 \
    --output-dir checkpoints/instruct
```

### Step 6: Benchmark & Evaluate
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/instruct/checkpoint_instruct_final.pt \
    --tokenizer-dir tokenizer_cache/smoke \
    --val-bin data/val.bin
```

---

## 3. Training Checkpoint Format

Checkpoints are saved as PyTorch `.pt` files containing:
```python
{
    "step": global_step,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scaler_state_dict": scaler.state_dict(),
    "model_config": model.config.to_dict(),
    "training_config": asdict(training_config),
    "val_loss": val_loss,
}
```
Weights are tied between `tok_embeddings.weight` and `lm_head.weight`, saving memory during checkpoint writes.
