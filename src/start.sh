#!/usr/bin/env bash

echo "Worker Initiated"

echo "Starting WebUI API"
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"
export PYTHONUNBUFFERED=true

# Add environment variables to prevent issues
export COMMANDLINE_ARGS="--skip-torch-cuda-test --no-half --precision full --disable-nan-check"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

python /stable-diffusion-webui/webui.py \
  --xformers \
  --skip-python-version-check \
  --skip-torch-cuda-test \
  --skip-install \
  --ckpt /stable-diffusion-webui/models/Stable-diffusion/INIVerse_Max.safetensors \
  --opt-sdp-attention \
  --disable-safe-unpickle \
  --port 3000 \
  --api \
  --nowebui \
  --skip-version-check \
  --no-hashing \
  --no-download-sd-model \
  --medvram &

# Wait for WebUI to initialize
sleep 15

echo "Starting RunPod Handler"
python -u /handler.py
