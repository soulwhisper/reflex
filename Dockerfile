# Weights are baked at build time (no runtime downloads). Override
# CHECKPOINTS to bake fewer/more; english-only is the default (~808 MB).
FROM python:3.12-slim AS base
ARG CHECKPOINTS=english

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving ./serving
COPY policies ./policies

# Warm the checkpoint cache into the image layer.
ENV REFLEX_CHECKPOINTS=${CHECKPOINTS} \
    USE_TF=0 \
    HF_HOME=/app/.hf
RUN python -c "import os; import laya; laya.Router(preload=os.environ['REFLEX_CHECKPOINTS'].split(','))"

ENV REFLEX_POLICIES=/app/policies/routes.yaml
EXPOSE 9000

# single process: /decide + /healthz + /readyz + /mcp (Streamable HTTP)
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "9000"]
