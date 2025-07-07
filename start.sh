#!/bin/bash
set -e

LOG_FILE="/app/runtime.log"

echo "[start.sh] Starting FastAPI server at $(date)" | tee -a "$LOG_FILE"

# Run uvicorn with logs both streamed and saved
exec uvicorn api:app --host 0.0.0.0 --port 8000 2>&1 | tee -a "$LOG_FILE"
