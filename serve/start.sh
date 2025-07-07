#!/bin/bash
LOG_FILE="/app/train_stdout.log"
echo "[start.sh] Starting at $(date)" | tee -a "$LOG_FILE"

python /app/train.py 2>&1 | tee -a "$LOG_FILE"

if [ $STATUS -ne 0 ]; then
    echo "[start.sh] Training failed with code $STATUS" | tee -a "$LOG_FILE"
    cat "$LOG_FILE"
    exit $STATUS
fi