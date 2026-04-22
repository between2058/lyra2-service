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

```bash
# 1. Download checkpoints (one-time, ~50 GB)
cd Lyra-2
pip install huggingface_hub
huggingface-cli download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir .
cd ..

# 2. Configure
cp .env.example .env
$EDITOR .env   # set HF_CACHE_HOST_PATH and any GPU/path overrides

# 3. Build + run
docker compose up -d --build
curl http://localhost:52071/health
```

## Integration with phidias-model

This repo ships its own `docker-compose.yml` and `nginx/nginx.conf` for standalone use. To fold into `phidias-model/docker-compose.yml`, copy the `lyra2:` service block and the nginx upstream block into the larger orchestration.
