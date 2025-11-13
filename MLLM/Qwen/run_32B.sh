#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

DATA_PATH="/root/personal/datasets/PM4Bench/MDUR/traditional/test1.jsonl"
MODEL_PATH="/root/hf_models/QwenVL25/Qwen2.5-VL-7B-Instruct"
SAVE_DIR="/root/personal/expriments/results_vqa_multilingual/Qwen-VL-v25/7B/hidden_state"
LANGS="EN,KO"
MAX_SAMPLES=100
layer_interval=4
batch_size=2

python test_vqa_2.py \
  --data_path "$DATA_PATH" \
  --model_path "$MODEL_PATH" \
  --save_dir "$SAVE_DIR" \
  --langs "$LANGS" \
  --max_samples "$MAX_SAMPLES" \
  --layer_interval "$layer_interval" \
  --batch_size "$batch_size"

echo "✅ Finished! Results saved under $SAVE_DIR"
