#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

DATA_PATH="/workspace/MMalign/dataset/PM4Bench/MDUR_multilingual_for_qwenvl_base64.jsonl"
MODEL_PATH="/root/hf_models/QwenVL25/Qwen2.5-VL-7B-Instruct"
SAVE_DIR="/root/personal/expriments/results_vqa_multilingual/Qwen-VL-v25/7B/hidden_state"
LANGS="EN,KO,SR,HU,AR,CS,TH,ZH,RU,VI"
MAX_SAMPLES=0
layer_interval=2
batch_size=1

python test_vqa_1.py \
  --data_path "$DATA_PATH" \
  --model_path "$MODEL_PATH" \
  --save_dir "$SAVE_DIR" \
  --langs "$LANGS" \
  --max_samples "$MAX_SAMPLES" \
  --layer_interval "$layer_interval" \
  --batch_size "$batch_size"

echo "✅ Finished! Results saved under $SAVE_DIR"
