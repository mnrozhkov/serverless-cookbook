#!/usr/bin/env bash
set -euo pipefail

python3 -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype float16 \
  --enable-lora \
  --max-loras 1 \
  --lora-modules tinyllama_adapter=/mnt/data/output/tinyllama-lora
