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

# Start the Gradio Web UI in the foreground
echo "Starting Gradio UI..."
python app.py
