# Lyra-2 Service

FastAPI Docker wrapper around [Lyra-2](https://github.com/nv-tlabs/lyra) for image → 3D Gaussian Splatting world generation. Targets NVIDIA RTX Pro 6000 Blackwell (sm_120).

## What it does

Three endpoints, all queue-based with polling:

| Endpoint | Input | Output | Runtime (no DMD / with DMD) |
|---|---|---|---|
| `POST /jobs/image-to-gs`    | image + caption / trajectory | `.ply` + `.mp4` | ~10 min / ~2 min |
| `POST /jobs/image-to-video` | image + caption / trajectory | `.mp4`         | ~9 min / ~35 s |
| `POST /jobs/video-to-gs`    | mp4 video                    | `.ply`         | ~1 min |

Submit a job → get `job_id` → poll `GET /jobs/{job_id}` → download via `GET /download/{job_id}/{file}`.

## Quickstart

Defaults assume the standard `/pegaai/model_team/huggingface_cache` layout, so
on a machine with that path the only file you need to touch is `.env` (for
your HuggingFace token).

```bash
# 1. Get an HF token (read access) at https://huggingface.co/settings/tokens
#    AND accept the gated-model license at https://huggingface.co/nvidia/Lyra-2.0
cp .env.example .env
$EDITOR .env       # paste HF_TOKEN=hf_xxx; override paths/GPU/port if needed

# 2. Build + run — first start auto-downloads checkpoints (~50 GB, 1-3 hours).
docker compose up -d --build
docker compose logs -f lyra2     # watch download progress

# 3. Verify
curl http://localhost:52075/health
```

The container's entrypoint (`entrypoint.sh`) checks `LYRA2_CHECKPOINTS_PATH/model/`
on every start. If empty, it runs `huggingface-cli download nvidia/Lyra-2.0
--include "checkpoints/*"` into the bind-mounted host directory. If already
populated, it skips download and starts uvicorn immediately.

### Optional: pre-download to skip the first-start wait

If you'd rather not block container start for hours, populate the host
directory before `docker compose up`:

```bash
mkdir -p /pegaai/model_team/huggingface_cache/Lyra-2.0
cd /pegaai/model_team/huggingface_cache/Lyra-2.0
pip install huggingface_hub
huggingface-cli login                              # or export HF_TOKEN
huggingface-cli download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir .
```

After this, the entrypoint's existence check passes and download is skipped
(no `HF_TOKEN` needed in `.env` either).

## Integration with phidias-model

This repo ships its own `docker-compose.yml` and `nginx/nginx.conf` for standalone use. To fold into `phidias-model/docker-compose.yml`, copy the `lyra2:` service block and the nginx upstream block into the larger orchestration.
