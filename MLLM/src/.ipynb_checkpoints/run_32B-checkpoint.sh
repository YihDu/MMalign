#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# === 参数定义 ===
DATA_PATH="/root/personal/datasets/PM4Bench/MDUR_multilingual_for_qwenvl_base64.jsonl"
MODEL_PATH="/root/personal/hf_models/Qwen-VL-2.5/7B-Instruct"
SAVE_DIR="test01"
LANGS="EN,ZH,AR"
MAX_SAMPLES=0
LAYERS="all"

# === 启动 ===
python qwen_vqa_single.py \
  --data_path "$DATA_PATH" \
  --model_path "$MODEL_PATH" \
  --save_dir "$SAVE_DIR" \
  --langs "$LANGS" \
  --max_samples "$MAX_SAMPLES" \
  --layers "$LAYERS"

echo "✅ Finished! Results saved under $SAVE_DIR"
