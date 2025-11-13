#!/bin/bash

DATA_DIR="/root/personal/expriments/results_vqa_multilingual/Qwen-VL-v25/7B/hidden_state/"
OUT_DIR="/root/personal/expriments/results_vqa_multilingual/Qwen-VL-v25/7B/results"
mkdir -p $OUT_DIR

# : << 'EOF'

# 计算 metric
echo "------ Computing CKA ------"
python compute_representation.py \
  --data_dir $DATA_DIR \
  --langs EN,KO,SR,HU,AR,CS,TH,ZH,RU,VI \
  --metric cka \
  --save_path $OUT_DIR/results_cka.csv

echo "------ Computing cosine ------"
python compute_representation.py \
  --data_dir $DATA_DIR \
  --langs EN,KO,SR,HU,AR,CS,TH,ZH,RU,VI \
  --metric cosine \
  --save_path $OUT_DIR/results_cosine.csv

echo "------ Computing normalized cosine ------"
python compute_representation.py \
  --data_dir $DATA_DIR \
  --langs EN,KO,SR,HU,AR,CS,TH,ZH,RU,VI \
  --metric cosine_norm \
  --save_path $OUT_DIR/results_cosine_norm.csv

# EOF

# 画图

python visualization/plot.py \
  --csv_path $OUT_DIR/results_cka.csv \
  --save_path $OUT_DIR/cka_plot.png

python  visualization/plot.py \
  --csv_path $OUT_DIR/results_cosine_norm.csv \
  --save_path $OUT_DIR/cosine_norm_plot.png

python  visualization/plot.py \
  --csv_path $OUT_DIR/results_cosine.csv \
  --save_path $OUT_DIR/cosine_plot.png

