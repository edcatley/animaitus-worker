# 1. SETUP THE BASE (CUDA 13.2)
ARG BASE_IMAGE=nvidia/cuda:13.2.1-devel-ubuntu24.04
FROM ${BASE_IMAGE} AS base

# 2. SETUP ARGS
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu132
ARG COMFYUI_VERSION=latest

# Added CMAKE_BUILD_PARALLEL_LEVEL back for faster compilation
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="12.0" \
    CMAKE_BUILD_PARALLEL_LEVEL=8 \
    UV_HTTP_TIMEOUT=600

# 3. INSTALL SYSTEM DEPS (Generic)
# FIXED: Added libgl1, libglib2.0-0, etc from the OLD dockerfile so OpenCV works
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-venv python3.12-dev \
    git wget build-essential ninja-build ffmpeg \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgoogle-perftools4t64 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 4. INSTALL PYTHON ENV
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && uv venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV="/opt/venv"

# 5. INSTALL COMFY & PYTORCH
RUN uv pip install comfy-cli pip setuptools wheel ninja

# Install PyTorch (This is fine to do manually to force the CUDA version)
RUN uv pip install --no-cache-dir torch==2.14.1 torchvision==0.29.1 --index-url ${PYTORCH_INDEX_URL}
# torchaudio has no cu132 build (its last release is 2.11, requiring torch>=2.11), so it comes from PyPI
RUN uv pip install --no-cache-dir torchaudio==2.11.1
# Install ComfyUI
RUN /usr/bin/yes | comfy --workspace /comfyui install --version "${COMFYUI_VERSION}" --nvidia \
    && uv pip install --no-cache-dir -r /comfyui/requirements.txt

WORKDIR /comfyui

# Copy the config telling Comfy to look at /network-volume
COPY src/extra_model_paths.yaml ./

WORKDIR /

# 6. SCRIPTS & CUSTOM NODES
# Add script to install custom nodes
COPY scripts/comfy-node-install.sh /usr/local/bin/comfy-node-install
RUN chmod +x /usr/local/bin/comfy-node-install

# Add script to configure Manager (Security) - RESTORED
COPY scripts/comfy-manager-set-mode.sh /usr/local/bin/comfy-manager-set-mode
RUN chmod +x /usr/local/bin/comfy-manager-set-mode

# Shared start script (used by all platform images)
COPY scripts/start.sh /start.sh
RUN chmod +x /start.sh

# Install Requirements (requests, websocket-client)
COPY requirements.txt .
RUN uv pip install --no-cache-dir -r requirements.txt

# Install SageAttention from source (must be after torch, requires --no-build-isolation)
# Its setup.py forces -std=c++17, but PyTorch 2.13+ headers need C++20. Removing the
# flag lets torch.utils.cpp_extension pick the standard for the installed PyTorch.
# Upstream issue: https://github.com/thu-ml/SageAttention/pull/405
RUN git clone --depth 1 --branch v2.2.0 https://github.com/thu-ml/SageAttention.git /tmp/SageAttention \
    && sed -i 's/"-std=c++17",[[:space:]]*//g' /tmp/SageAttention/setup.py \
    && ! grep -q 'std=c++17' /tmp/SageAttention/setup.py \
    && MAX_JOBS=1 CUDA_HOME=/usr/local/cuda uv pip install --no-build-isolation /tmp/SageAttention \
    && rm -rf /tmp/SageAttention

# Copy generic base handler (used by platform-specific wrappers)
COPY src/ /src/

# Add demo data for mock workflows to ComfyUI's input directory
COPY test-data/ /comfyui/input/demo/

# CUSTOM NODE INSTALL
ENV PIP_NO_INPUT=1
RUN comfy-node-install \
    comfyui-videohelpersuite \
    comfyui-kjnodes \
    comfyui-custom-scripts \
    comfyui-wan-vace-prep \
    tripleksampler

# Install forked frame interpolation node (abandoned upstream, using fork)
RUN git clone https://github.com/edcatley/ComfyUI-Frame-Interpolation /comfyui/custom_nodes/ComfyUI-Frame-Interpolation \
    && uv pip install --no-cache-dir --no-build-isolation -r /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/requirements-with-cupy.txt

# 7. EXPOSE THE STANDARD PORT
EXPOSE 8188

# 8. DEFAULT CMD (Just runs Comfy normally)
CMD ["python", "main.py", "--disable-metadata", "--listen", "0.0.0.0"]
