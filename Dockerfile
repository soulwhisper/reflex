# Release format: ONNX runtime + the bundled English base checkpoint
# (tozp/laya-onnx repo root: model.onnx + tokenizer.json + rl_agent_config.json;
# the separately-tuned laya-typed-decisions artifact is NOT bundled — shadow
# data must come from the English base we intend to fine-tune).
# No torch/transformers at runtime — the default PyPI torch wheel alone added
# ~5 GB of CUDA libs on a CPU-only deployment; this image is ~2 GB (fp32) or
# ~800 MB (MODEL_FILE=model_int8.onnx). Fine-tuned checkpoints take the same
# path: finetune/export_onnx.py emits this exact layout, then build with
#   --build-arg MODEL_REPO=<your-hf-repo> [--build-arg MODEL_FILE=model.onnx]
ARG MODEL_REPO=tozp/laya-onnx
ARG MODEL_FILE=model.onnx

FROM python:3.14-slim@sha256:caaf356f40667c496d405780745b9ac25771c189a51dfcc42430d531ea09f8a2 AS base

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fetch the checkpoint in a throwaway stage so huggingface_hub (and its deps)
# never land in the final image. Hub LFS integrity is verified by the library.
FROM base AS model
ARG MODEL_REPO
ARG MODEL_FILE
RUN pip install --no-cache-dir huggingface-hub \
 && python - "$MODEL_REPO" "$MODEL_FILE" <<'PYEOF'
import os
import sys

from huggingface_hub import hf_hub_download

repo, model_file = sys.argv[1], sys.argv[2]
dest = "/model"
os.makedirs(dest, exist_ok=True)
for name in (model_file, "tokenizer.json", "rl_agent_config.json"):
    path = hf_hub_download(repo, name, local_dir=dest, token=os.environ.get("HF_TOKEN"))
    print("fetched", path)
# normalize to the runtime's expected filename
if model_file != "model.onnx":
    os.replace(os.path.join(dest, model_file), os.path.join(dest, "model.onnx"))
PYEOF

FROM base AS runtime
ARG MODEL_REPO
ENV REFLEX_MODEL_DIR=/app/model \
    REFLEX_MODEL_REPO=${MODEL_REPO} \
    REFLEX_POLICIES=/app/policies/routes.yaml
COPY app ./app
COPY policies ./policies
COPY --from=model /model /app/model
EXPOSE 9000

# single process: /decide + /healthz + /readyz + /mcp (Streamable HTTP)
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "9000"]
