#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends build-essential
rm -rf /var/lib/apt/lists/*

python3 -m pip install --upgrade pip
python3 -m pip install --no-cache-dir \
  transformers==4.49.0 \
  datasets==3.3.2 \
  accelerate==1.4.0 \
  peft==0.14.0 \
  trl==0.15.1 \
  bitsandbytes==0.45.2 \
  sentencepiece==0.2.0 \
  mlflow==2.20.3

cp /mnt/data/fine_tune.py /tmp/fine_tune.py

python3 /tmp/fine_tune.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --output-dir /mnt/data/output/tinyllama-lora \
  --num-epochs 1 \
  --batch-size 1 \
  --gradient-accum-steps 4 \
  --logging-steps 1 \
  --max-seq-length 256 \
  --max-samples 32 \
  --skip-inference
