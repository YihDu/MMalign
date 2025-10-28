#!/usr/bin/env bash
# ================================================================
# Script: run_geometry_pipeline.sh
# Purpose: Batch process multiple LLaVA checkpoints to trace
#          representation geometry (RankMe & αReQ).
# Author: Yihang-style research setup
# ================================================================

set -euo pipefail
export PYTHONUNBUFFERED=1

# ========================【用户可自定义区域】========================

# 1️⃣ 阶段选择: pretrain / sft / dpo / rlvr
STAGE="pretrain"

# 2️⃣ checkpoint 文件夹路径
#    若 checkpoint 命名非标准，可自行修改匹配模式
CKPT_DIR="./checkpoints/${STAGE}"

# 3️⃣ 输出目录
OUTPUT_DIR="./results/${STAGE}/geometry_evolution"

# 4️⃣ 配置文件路径 (可在 configs/ 下定义分析参数)
CONFIG="./configs/spectral_analysis.yaml"

# 5️⃣ GPU 选择（如为 CPU 环境可注释此行）
export CUDA_VISIBLE_DEVICES="0"

# 6️⃣ 每个 checkpoint 的采样数量
NUM_SAMPLES=5000

# 7️⃣ 指定分析层 (last / penultimate / all)
TARGET_LAYER="last"

# 8️⃣ token 选择: text_last / text_all / vision_all / all
#    如果在 extract_features.py 中实现多模态分离，这里可以切换
TOKEN_SCOPE="text_last"

# 9️⃣ 自动绘图选项
PLOT_AFTER_RUN=true

# 🔟 并行开关（可选）: 是否并行处理多个 checkpoint
PARALLEL=false
MAX_JOBS=2  # 并行最大任务数，仅当 PARALLEL=true 时有效

# ========================【系统配置部分】========================
mkdir -p "$OUTPUT_DIR"
LOG_FILE="${OUTPUT_DIR}/geometry_pipeline_$(date +'%Y%m%d_%H%M%S').log"

echo "==================================================" | tee "$LOG_FILE"
echo "[INFO] Stage          : ${STAGE}" | tee -a "$LOG_FILE"
echo "[INFO] Checkpoint dir : ${CKPT_DIR}" | tee -a "$LOG_FILE"
echo "[INFO] Output dir     : ${OUTPUT_DIR}" | tee -a "$LOG_FILE"
echo "[INFO] Config file    : ${CONFIG}" | tee -a "$LOG_FILE"
echo "[INFO] GPU setting    : ${CUDA_VISIBLE_DEVICES:-CPU}" | tee -a "$LOG_FILE"
echo "[INFO] Num samples    : ${NUM_SAMPLES}" | tee -a "$LOG_FILE"
echo "[INFO] Target layer   : ${TARGET_LAYER}" | tee -a "$LOG_FILE"
echo "[INFO] Token scope    : ${TOKEN_SCOPE}" | tee -a "$LOG_FILE"
echo "[INFO] Parallel mode  : ${PARALLEL}" | tee -a "$LOG_FILE"
echo "[INFO] Start time     : $(date)" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"

# ========================【主循环：逐 ckpt 运行】========================
job_count=0
for CKPT_PATH in "${CKPT_DIR}"/step_*; do
    if [[ -d "$CKPT_PATH" || -f "$CKPT_PATH" ]]; then
        echo "[RUN] Processing checkpoint: $CKPT_PATH" | tee -a "$LOG_FILE"

        CMD="python scripts/analysis/pipeline_geometry.py \
            --ckpt \"$CKPT_PATH\" \
            --output \"$OUTPUT_DIR\" \
            --config \"$CONFIG\" \
            --num_samples \"$NUM_SAMPLES\" \
            --layer \"$TARGET_LAYER\" \
            --token_scope \"$TOKEN_SCOPE\""

        if [ "$PARALLEL" = true ]; then
            eval "$CMD" 2>&1 | tee -a "$LOG_FILE" &
            ((job_count+=1))
            # 控制并行任务数量
            if [ "$job_count" -ge "$MAX_JOBS" ]; then
                wait
                job_count=0
            fi
        else
            eval "$CMD" 2>&1 | tee -a "$LOG_FILE"
        fi
    else
        echo "[WARN] No valid checkpoints found in $CKPT_DIR" | tee -a "$LOG_FILE"
    fi
done
wait

# ========================【绘图部分】========================
if [ "$PLOT_AFTER_RUN" = true ]; then
    echo "[INFO] Generating geometry evolution plot..." | tee -a "$LOG_FILE"
    python scripts/analysis/visualize_geometry.py \
        --input "$OUTPUT_DIR" \
        --output "${OUTPUT_DIR}/geometry_curve.png" \
        2>&1 | tee -a "$LOG_FILE"
fi

echo "==================================================" | tee -a "$LOG_FILE"
echo "[✅ DONE] All checkpoints processed successfully." | tee -a "$LOG_FILE"
echo "[INFO] Results saved to: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "[INFO] Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "=================================================="
