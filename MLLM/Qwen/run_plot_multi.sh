#!/bin/bash

# ========== model folders ==========
MODEL_DIRS=(
  "/root/personal/expriments/results_vqa_multilingual/Qwen-VL-v25/3B/results"
  "/root/personal/expriments/results_vqa_multilingual/Qwen-VL-v25/7B/results"
)

LABELS=(
  "Qwen2.5-VL-3B"
  "Qwen2.5-VL-7B"
)

# You only specify metrics names
# METRICS=("cka" "cosine" "cosine_norm")
METRICS=("cka")
OUT_DIR="/root/personal/plots"
mkdir -p "$OUT_DIR"

python visualization/plot_multi.py \
  --model_dirs "${MODEL_DIRS[@]}" \
  --labels "${LABELS[@]}" \
  --metrics "${METRICS[@]}" \
  --output_dir "$OUT_DIR"
