#!/bin/bash
set -e

# Train dataset if existing model doesn't exist
if [ ! -f /app/model.h5 ]; then
    python /app/train.py
else
    echo "Model already exists. Skipping training."
fi

# Start the FastAPI server
echo "Starting FastAPI server..."
uvicorn api:app --host 0.0.0.0 --port 8000
