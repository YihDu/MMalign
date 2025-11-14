#!/bin/bash

LOGFILE="monitor.log"

echo "=== Monitoring started at $(date) ===" >> $LOGFILE

while true; do
    echo "-------------------- $(date) --------------------" >> $LOGFILE

    echo "[GPU MEMORY]" >> $LOGFILE
    nvidia-smi >> $LOGFILE 2>&1

    echo "" >> $LOGFILE

    echo "[GPU ERRORS]" >> $LOGFILE
    nvidia-smi -q | grep -Ei "xid|error|fail|ecc|critical" >> $LOGFILE 2>&1

    echo "" >> $LOGFILE

    # List python memory usage
    echo "[PYTHON MEMORY]" >> $LOGFILE
    ps aux | grep python | grep -v grep >> $LOGFILE 2>&1

    echo "" >> $LOGFILE

    echo "[SYSTEM MEMORY]" >> $LOGFILE
    free -h >> $LOGFILE 2>&1

    echo "" >> $LOGFILE

    sleep 1
done
