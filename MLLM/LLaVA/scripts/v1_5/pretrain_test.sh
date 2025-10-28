#!/usr/bin/env bash
set -euo pipefail

# ========== 环境变量部分 ==========
export PYTORCH_SHARING_STRATEGY=file_system
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_DEVICE_MAX_CONNECTIONS=1

mkdir -p /localdata/torch_tmp
export TMPDIR=/localdata/torch_tmp
export TORCH_HOME=/localdata/torch_tmp

# ======【参数：可按需改】======
IMG_DIR="/tmp/localdata/LLaVA_pretrain"
DATA_JSON="/workspace/MMalign/dataset/LLaVA/LLaVA-pretrain/blip_laion_cc_sbu_558k.json"
OUT_DIR="./checkpoints/llava-v1.5-7b-pretrain/20251028"
LOG_DIR="./logs"
mkdir -p "$LOG_DIR" "$OUT_DIR"

# ======【工具函数】======
require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "缺少命令: $1"; exit 1; }; }
bytes_to_gb() { python3 - <<'PY'
import sys; print(round(int(sys.stdin.read().strip())/1024/1024/1024,2))
PY
}

# ======【自检 1：共享内存 /dev/shm】======
echo "[CHECK] /dev/shm 容量："
df -h /dev/shm || true

# 强制用字节单位取可用空间（去掉非数字字符）
SHM_BYTES=$(df -B1 --output=avail /dev/shm 2>/dev/null | tail -1 | tr -dc '0-9')

# 兜底：如果上面没取到，再用 awk 解析常规 df 的第4列（可用）
if [[ -z "$SHM_BYTES" ]]; then
  SHM_BYTES=$(df -B1 /dev/shm 2>/dev/null | awk 'NR==2{print $4}')
fi

# 再兜底：实在拿不到就假定为 0
if [[ -z "$SHM_BYTES" ]]; then
  echo "[WARN] 读取 /dev/shm 可用空间失败，按 0 处理。"
  SHM_BYTES=0
fi

# 转 GB（用 awk，避免依赖 python/bc）
SHM_GB=$(awk -v b="$SHM_BYTES" 'BEGIN{printf "%.2f", b/1024/1024/1024}')

echo "[INFO] /dev/shm 可用空间: ${SHM_GB} GB"

# 小于 1GB 就提醒保持 file_system 策略
awk -v g="$SHM_GB" 'BEGIN{exit !(g<1.0)}'
if [[ $? -eq 0 ]]; then
  echo "[INFO] /dev/shm < 1GB，继续使用 PYTORCH_SHARING_STRATEGY=file_system。"
fi

# ======【自检 2：数据目录存在 & 非空】======
echo "[CHECK] 图像目录：$IMG_DIR"
if [[ ! -d "$IMG_DIR" ]]; then
  echo "[ERROR] 图像目录不存在：$IMG_DIR"; exit 1;
fi
IMG_CNT=$(ls -1 "$IMG_DIR" | wc -l || echo 0)
if [[ "$IMG_CNT" -le 0 ]]; then
  echo "[ERROR] 图像目录为空：$IMG_DIR"; exit 1;
fi
echo "[OK] 图像文件数：$IMG_CNT"

# ======【自检 3：数据 JSON】======
[[ -f "$DATA_JSON" ]] || { echo "[ERROR] data_path 不存在：$DATA_JSON"; exit 1; }
echo "[OK] data_path 存在：$DATA_JSON"

# ======【自检 4：输出目录磁盘空间】======
echo "[CHECK] 输出目录磁盘："
df -h "$OUT_DIR" || true

# ======【监控：GPU 采样到日志】======
require_cmd nvidia-smi
GPU_MON_LOG="$LOG_DIR/gpu_monitor_$(date +%Y%m%d_%H%M%S).log"
echo "[MONITOR] GPU 监控写入：$GPU_MON_LOG"
# 每 5s 记录一次：时间戳, GPU利用率, 显存占用(MiB), 温度, 功耗(W)
( while true; do
    echo -n "$(date '+%F %T'), "
    nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw \
               --format=csv,noheader,nounits
    sleep 5
  done ) >> "$GPU_MON_LOG" 2>&1 &
GPU_MON_PID=$!

# 结束时清理监控
cleanup() {
  echo "[CLEANUP] 停止 GPU 监控 (pid=$GPU_MON_PID)"
  kill ${GPU_MON_PID} >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

# ======【Deepspeed 每 rank 日志】======
DS_ARGS="--enable_each_rank_log"

# ======【启动训练（照你的参数，仅把日志与监控增强）】======
echo "[RUN] 启动训练..."
nohup deepspeed --num_gpus=4 /workspace/MMalign/MLLM/LLaVA/llava/train/train_mem.py \
  --deepspeed ../zero2.json \
  --model_name_or_path /tmp/models/vicuna-7b-v1.5 \
  --version plain \
  --data_path "$DATA_JSON" \
  --image_folder "$IMG_DIR" \
  --vision_tower /tmp/models/clip-vit-large-patch14-336  \
  --mm_projector_type mlp2x_gelu \
  --tune_mm_mlp_adapter True \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --bf16 True \
  --output_dir "$OUT_DIR" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 1000000 \
  --learning_rate 1e-3 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 0 \
  --lazy_preprocess True \
  --report_to wandb \
  2>&1 | tee "$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

echo "[DONE] 训练进程结束。最近 GPU 监控："
tail -n 10 "$GPU_MON_LOG" || true
