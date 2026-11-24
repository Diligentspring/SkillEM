#!/usr/bin/env bash
# Run in SkillRL's separate Linux/CUDA environment.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
SOURCE="$ROOT/materials/SkillRL-4d984d455b9b02f0bad283cbe9ab3d66a7e5b89d"
: "${MODEL_PATH:?Set MODEL_PATH to the common Search SFT initialization}"
: "${TRAIN_PARQUET:?Set TRAIN_PARQUET to the prepared training-only parquet}"
: "${VALIDATION_PARQUET:?Set VALIDATION_PARQUET to the held-out validation parquet}"
: "${TRAINING_STEPS:?Set an explicit optimizer-step budget}"
cd "$SOURCE"
bash examples/grpo_trainer/run_search_skills.sh vllm \
  "data.train_files=$TRAIN_PARQUET" "data.val_files=$VALIDATION_PARQUET" \
  trainer.total_epochs=100000 "trainer.total_training_steps=$TRAINING_STEPS" \
  trainer.logger='[console]' "$@"
