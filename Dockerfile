# =============================================================================
# Lyra-2 API — Docker Image
#
# Target: NVIDIA RTX Pro 6000 Blackwell (sm_120)
# CUDA 12.8  |  Python 3.10  |  PyTorch 2.7.1+cu128
#
# Build context: repo root (contains Lyra-2/ + lyra2_api.py + requirements-api.txt)
# Build:  docker build -t lyra2-api:latest .
# Run:    docker run --gpus all -p 52075:52075 \
#                  -v $HF_CACHE:/app/hf_cache \
#                  -v $(pwd)/Lyra-2/checkpoints:/app/Lyra-2/checkpoints:ro \
#                  lyra2-api:latest
# =============================================================================

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST="12.0"
ARG MAX_JOBS=4
ARG http_proxy=""
ARG https_proxy=""
ARG no_proxy="localhost,127.0.0.1"

ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    HTTP_PROXY=${http_proxy} \
    HTTPS_PROXY=${https_proxy} \
    no_proxy=${no_proxy} \
    NO_PROXY=${no_proxy}

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    NVIDIA_VISIBLE_DEVICES=all \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    HF_HOME=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache \
    HUGGINGFACE_HUB_CACHE=/app/hf_cache \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── System packages ──────────────────────────────────────────────────────────
# Note: Lyra-2 INSTALL.md pins conda gcc=13.3.0, but Ubuntu 22.04's default apt
# only ships gcc-11/12. We keep the build-essential default (gcc-11) — CUDA 12.8
# supports it as a host compiler and TRELLIS.2 (same Blackwell stack) builds
# fine with it. libeigen3-dev replaces conda's eigen requirement.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    build-essential ninja-build cmake git wget curl \
    libeigen3-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libegl1-mesa-dev libgles2-mesa-dev libgomp1 ffmpeg libjpeg-dev \
 && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python -m pip install --upgrade --no-cache-dir pip setuptools wheel

WORKDIR /app

# ── PyTorch 2.7.1 + cu128 ───────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

# ── Lyra-2 upstream Python deps (from Lyra-2/requirements.txt) ──────────────
# Install with --no-deps as upstream INSTALL.md instructs.
COPY Lyra-2/requirements.txt /tmp/lyra2-requirements.txt
RUN pip install --no-cache-dir --no-deps -r /tmp/lyra2-requirements.txt

# ── MoGe (separate install — has its own deps) ──────────────────────────────
RUN pip install --no-cache-dir "git+https://github.com/microsoft/MoGe.git"

# ── transformer_engine (Blackwell support: v1.13+, latest pulls v2.x) ───────
RUN pip install --no-cache-dir --no-build-isolation "transformer_engine[pytorch]"

# Symlink cuda_runtime → cudart for transformer_engine compatibility
RUN SITE=$(python -c "import site; print(site.getsitepackages()[0])") \
 && ln -sf "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"

# ── flash-attn — UNPINNED, latest version (Lyra's pin of 2.6.3 has no sm_120 kernel) ──
RUN MAX_JOBS=${MAX_JOBS} pip install --no-cache-dir --no-build-isolation flash-attn

# ── Vendored CUDA extensions ────────────────────────────────────────────────
COPY Lyra-2/lyra_2/_src/inference/vipe              /app/Lyra-2/lyra_2/_src/inference/vipe
COPY Lyra-2/lyra_2/_src/inference/depth_anything_3  /app/Lyra-2/lyra_2/_src/inference/depth_anything_3
RUN USE_SYSTEM_EIGEN=1 MAX_JOBS=${MAX_JOBS} pip install --no-cache-dir --no-build-isolation \
    -e /app/Lyra-2/lyra_2/_src/inference/vipe
RUN MAX_JOBS=${MAX_JOBS} pip install --no-cache-dir --no-build-isolation \
    -e '/app/Lyra-2/lyra_2/_src/inference/depth_anything_3[gs]'

# ── API deps ─────────────────────────────────────────────────────────────────
COPY requirements-api.txt /tmp/lyra2-api-requirements.txt
RUN pip install --no-cache-dir -r /tmp/lyra2-api-requirements.txt

# ── Re-lock PyTorch (transitive deps may have downgraded it) ────────────────
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cu128

# ── Application source ───────────────────────────────────────────────────────
# Copy the full Lyra-2 tree (will overwrite the vipe/da3 placeholders from above
# with identical contents — that's fine).
COPY Lyra-2/lyra_2 /app/Lyra-2/lyra_2
COPY Lyra-2/assets /app/Lyra-2/assets
COPY lyra2_api.py /app/lyra2_api.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV PYTHONPATH=/app/Lyra-2:${PYTHONPATH}
RUN mkdir -p /app/logs /app/outputs /app/hf_cache

EXPOSE 52075

# start_period is 2h to cover first-run checkpoint download (~50 GB).
# Subsequent restarts skip the download and become healthy within 5 min.
HEALTHCHECK \
    --interval=30s \
    --timeout=15s \
    --start-period=7200s \
    --retries=5 \
    CMD curl -f http://localhost:52075/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "lyra2_api:app", \
     "--host", "0.0.0.0", \
     "--port", "52075", \
     "--workers", "1", \
     "--log-level", "info"]
