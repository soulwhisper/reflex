#!/usr/bin/env bash
# NVIDIA launcher for the RLCD fine-tune pipeline (CUDA, 1-2 GPUs).
# Kaggle 2xT4 equivalent: NPROC=2. Usage from repo root:
#   bash finetune/run_nvidia.sh
set -euo pipefail

MODEL_ID=${MODEL_ID:-convaiinnovations/laya}
DATA_DIR=${DATA_DIR:-finetune/datasets}
POLICIES=${POLICIES:-policies/routes.yaml}
BUILD_DIR=${BUILD_DIR:-finetune/build}
OUT_DIR=${OUT_DIR:-$BUILD_DIR/out}
NPROC=${NPROC:-2}
EPOCHS=${EPOCHS:-4}

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found; no NVIDIA host?"; exit 1; }
nvidia-smi -L
[ -f "$POLICIES" ] || { echo "run from the repo root; $POLICIES missing"; exit 1; }
ls "$DATA_DIR"/*.jsonl >/dev/null 2>&1 || { echo "no datasets in $DATA_DIR; see docs/FINETUNING.md"; exit 1; }

export USE_TF=0  # transformers TF probe can deadlock model construction
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python finetune/preprocess.py --model "$MODEL_ID" --datasets "$DATA_DIR" \
  --policies "$POLICIES" --out "$BUILD_DIR"

torchrun --standalone --nproc_per_node="$NPROC" finetune/train.py \
  "$MODEL_ID" "$BUILD_DIR/train_items.pt" "$OUT_DIR" \
  --holdout "$BUILD_DIR/holdout_items.pt" --epochs "$EPOCHS"

python finetune/evaluate.py --checkpoint "$OUT_DIR" \
  --items "$BUILD_DIR/holdout_items.pt" --baseline "$MODEL_ID"

echo "checkpoint: $OUT_DIR (temperatures.json + rl_agent_config.json included)"
