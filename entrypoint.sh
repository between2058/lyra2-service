#!/bin/bash
# =============================================================================
# Lyra-2 container entrypoint
#
# Auto-downloads checkpoints from huggingface.co/nvidia/Lyra-2.0 on first start
# (when the bind-mounted host directory is empty), then exec's the supplied CMD
# (uvicorn). On subsequent starts the populated directory is detected and the
# download is skipped.
# =============================================================================
set -euo pipefail

CKPT_DIR="${LYRA2_CHECKPOINT_DIR:-/app/Lyra-2/checkpoints}"
DOWNLOAD_TARGET="$(dirname "$CKPT_DIR")"
MARKER_DIR="$CKPT_DIR/model"

if [[ -d "$MARKER_DIR" ]] && [[ -n "$(ls -A "$MARKER_DIR" 2>/dev/null)" ]]; then
    echo "[entrypoint] Checkpoints already present at $CKPT_DIR — skipping download."
else
    echo "[entrypoint] Checkpoints missing at $CKPT_DIR."
    echo "[entrypoint] Downloading nvidia/Lyra-2.0 (~50 GB; first run takes 1-3 hours depending on bandwidth)..."

    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "[entrypoint] WARNING: HF_TOKEN is not set."
        echo "[entrypoint]   nvidia/Lyra-2.0 is a gated repo. The download will fail with 401/403"
        echo "[entrypoint]   unless you have:"
        echo "[entrypoint]     1. accepted the license at https://huggingface.co/nvidia/Lyra-2.0"
        echo "[entrypoint]     2. set HF_TOKEN=hf_xxx in .env (or pre-downloaded the checkpoints"
        echo "[entrypoint]        into the bind-mounted host directory)."
    fi

    mkdir -p "$DOWNLOAD_TARGET"
    huggingface-cli download nvidia/Lyra-2.0 \
        --include "checkpoints/*" \
        --local-dir "$DOWNLOAD_TARGET" \
        ${HF_TOKEN:+--token "$HF_TOKEN"}

    echo "[entrypoint] Download complete. Checkpoints at $CKPT_DIR."
fi

# Hand off to the actual server process (uvicorn, supplied via Dockerfile CMD).
exec "$@"
