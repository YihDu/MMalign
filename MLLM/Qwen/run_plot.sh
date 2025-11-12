#!/bin/bash

DATA_DIR="/root/personal/expriments/results_vqa_multilingual/Qwen-VL-v25/7B/hidden_state/"
OUT_DIR="/root/personal/expriments/results_vqa_multilingual/Qwen-VL-v25/7B/results"
mkdir -p $OUT_DIR

# # 1️⃣ Compute metrics
echo ">>> Computing CKA..."
python compute_representation.py \
  --data_dir $DATA_DIR \
  --langs EN,KO,SR,HU \
  --metric cka \
  --save_path $OUT_DIR/results_cka.csv

# echo ">>> Computing normalized cosine..."
# python compute_representation.py \
#   --data_dir $DATA_DIR \
#   --langs EN,ZH,AR,VI \
#   --metric cosine_norm \
#   --save_path $OUT_DIR/results_cosine_norm.csv

echo ">>> Computing cosine..."
python compute_representation.py \
  --data_dir $DATA_DIR \
  --langs EN,KO,SR,HU \
  --metric cosine \
  --save_path $OUT_DIR/results_cosine.csv


# # 2️⃣ Plot results
python plot.py \
  --csv_path $OUT_DIR/results_cka.csv \
  --save_path $OUT_DIR/cka_plot.png

# python plot.py \
#   --csv_path $OUT_DIR/results_cosine_norm.csv \
#   --save_path $OUT_DIR/cosine_norm_plot.png

python plot.py \
  --csv_path $OUT_DIR/results_cosine.csv \
  --save_path $OUT_DIR/cosine_plot.png

