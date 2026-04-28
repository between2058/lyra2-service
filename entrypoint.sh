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

# Use a SPECIFIC small file as the "fully downloaded" marker rather than just
# "is the model dir non-empty". Lyra-2's HF repo contains 6 subdirs (image_encoder,
# lora, model, recon, text_encoder, vae) and a partial download (e.g. interrupted
# manual hf-cli) can leave model/ populated while text_encoder/ is still missing,
# which causes inference to crash with FileNotFoundError on negative_prompt.pt.
# Pick a small file we know is needed by every inference path.
MARKER_FILE="$CKPT_DIR/text_encoder/negative_prompt.pt"

if [[ -f "$MARKER_FILE" ]]; then
    echo "[entrypoint] Checkpoints already complete at $CKPT_DIR (marker: $MARKER_FILE) — skipping download."
else
    echo "[entrypoint] Checkpoints missing or incomplete at $CKPT_DIR (marker file $MARKER_FILE not found)."
    echo "[entrypoint] Downloading nvidia/Lyra-2.0 (~50 GB; first run takes 1-3 hours depending on bandwidth; resumable)..."

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
