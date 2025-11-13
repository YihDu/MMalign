#!/bin/bash

LOGFILE="log(Qwen_7B_1113).log"

echo "===== GPU MONITOR START =====" > $LOGFILE
echo "TIME: $(date)" >> $LOGFILE
echo "" >> $LOGFILE

# 每 1 秒打印一次 GPU 全状态
while true; do
    {
        echo "----------------------------------------"
        echo "TIME: $(date)"
        nvidia-smi
        echo ""
        echo "Processes:"
        nvidia-smi pmon -c 1
        echo ""
    } >> $LOGFILE

    sleep 1
done