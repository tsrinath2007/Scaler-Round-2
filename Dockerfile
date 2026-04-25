# Closed-Loop Life Support OpenEnv — Training + Serving
# Runs on HuggingFace Docker Space with T4 GPU
FROM python:3.11-slim

# HF Spaces runs as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /app

# Install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=user . .

# HF Spaces requires port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/health').raise_for_status()"

# Full pipeline: PPO training → Expert data → LLM fine-tuning → Serve
# Override HUB_REPO with your HuggingFace username
# Set HF_TOKEN as a Space secret
CMD python train_full_pipeline.py \
        --task task_medium \
        --ppo-timesteps 500000 \
        --expert-episodes 150 \
        --llm-epochs 3 \
        --hub-repo ${HUB_REPO:-""} \
        --skip-push && \
    uvicorn server.app:app --host 0.0.0.0 --port 7860
