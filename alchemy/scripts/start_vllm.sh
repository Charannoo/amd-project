#!/usr/bin/env bash
# Start Qwen2.5-72B via vLLM (run on GPU server with enough VRAM).
set -e
export VLLM_HOST="${VLLM_HOST:-http://localhost:8000}"
exec python -m vllm.entrypoints.openai.api_server \
  --model "${VLLM_MODEL:-Qwen/Qwen2.5-72B-Instruct}" \
  --dtype float16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.75 \
  --port "${VLLM_PORT:-8000}" \
  --api-key "${VLLM_API_KEY:-token-alchemy}" \
  "$@"
