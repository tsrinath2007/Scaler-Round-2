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

# Health check — Gradio serves at /
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')"

# Full pipeline: PPO training → Expert data → LLM fine-tuning → Serve
# Training runs in background, server runs in foreground to pass health checks
COPY --chown=user start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
