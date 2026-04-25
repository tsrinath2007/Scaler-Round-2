#!/bin/bash
# Start the training pipeline in the background
echo "Starting training pipeline in the background..."
python train_full_pipeline.py \
    --task task_medium \
    --ppo-timesteps 500000 \
    --expert-episodes 150 \
    --llm-epochs 3 \
    --hub-repo "${HUB_REPO:-}" \
    --skip-push > /app/training.log 2>&1 &

# Start the FastAPI server in the foreground
echo "Starting FastAPI server..."
uvicorn server.app:app --host 0.0.0.0 --port 7860
