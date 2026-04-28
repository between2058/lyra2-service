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

Defaults assume the standard `/pegaai/model_team/huggingface_cache` layout. On a
machine with that path, no `.env` editing is required — `docker compose up -d`
just works after the one-time checkpoint download.

```bash
# 1. Download checkpoints (one-time, ~50 GB) into the shared HF cache
mkdir -p /pegaai/model_team/huggingface_cache/Lyra-2.0
cd /pegaai/model_team/huggingface_cache/Lyra-2.0
pip install huggingface_hub
huggingface-cli download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir .

# 2. Build + run (no .env needed if using the default paths)
cd <wherever-you-cloned-this-repo>
docker compose up -d --build
curl http://localhost:52071/health
```

If your machine uses different paths, copy and edit the env file:

```bash
cp .env.example .env
$EDITOR .env   # override HF_CACHE_HOST_PATH / LYRA2_CHECKPOINTS_PATH / GPU id
docker compose up -d --build
```

## Integration with phidias-model

This repo ships its own `docker-compose.yml` and `nginx/nginx.conf` for standalone use. To fold into `phidias-model/docker-compose.yml`, copy the `lyra2:` service block and the nginx upstream block into the larger orchestration.
