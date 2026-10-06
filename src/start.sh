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

WEBUI_PID=$!

# Wait for WebUI to initialize (up to 60s)
echo "Waiting for WebUI to start..."
for i in $(seq 1 30); do
  if curl -s http://127.0.0.1:3000/sdapi/v1/sd-models > /dev/null 2>&1; then
    echo "WebUI is ready!"
    break
  fi
  if ! kill -0 $WEBUI_PID 2>/dev/null; then
    echo "ERROR: WebUI process died during startup"
    exit 1
  fi
  sleep 2
done

# Final check
if ! curl -s http://127.0.0.1:3000/sdapi/v1/sd-models > /dev/null 2>&1; then
  echo "ERROR: WebUI failed to start within timeout"
  exit 1
fi

echo "Starting RunPod Handler"
python -u /handler.py
