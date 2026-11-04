#!/usr/bin/env bash

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Suppress verbose logging from PyTorch and other libraries
export TORCH_LOGS="-all"
export TORCH_CPP_LOG_LEVEL="ERROR"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_LAUNCH_BLOCKING="0"
export PYTHONWARNINGS="ignore"

# Ensure ComfyUI-Manager runs in offline network mode inside the container
comfy-manager-set-mode offline || echo "worker-comfyui - Could not set ComfyUI-Manager network_mode" >&2

# Allow operators to tweak verbosity; default is INFO (less verbose than DEBUG)
: "${COMFY_LOG_LEVEL:=INFO}"

echo "worker-comfyui - Starting ComfyUI..."
python -u /comfyui/main.py \
    --disable-auto-launch \
    --disable-metadata \
    --verbose "${COMFY_LOG_LEVEL}" \
    --log-stdout \
    --use-ck-attention &

echo "worker-comfyui - Starting handler..."
exec python -u /handler.py
