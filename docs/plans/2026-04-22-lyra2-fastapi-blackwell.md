# Lyra-2 FastAPI Service for Blackwell RTX Pro 6000 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package Lyra-2 (image → exploration video → 3D Gaussian Splatting) as a single FastAPI Docker service deployable on RTX Pro 6000 Blackwell (sm_120). Delivered as a **self-contained repo** at `/Users/between2058/Documents/code/lyra2-service/` with its own `docker-compose.yml`, `nginx.conf`, and `.env.example`. Integration into the larger `phidias-model` orchestration is left as a downstream cherry-pick (out of scope here).

**Architecture:** One container, one GPU, three job endpoints (`/jobs/image-to-gs` for the full image→video→GS pipeline, `/jobs/video-to-gs` for video-only GS reconstruction, `/jobs/image-to-video` for video-only generation). Inference is serialised through an `asyncio.Queue` + `asyncio.Semaphore(1)` GPU worker; clients submit jobs and poll `GET /jobs/{id}`. All weights are lazy-loaded once, shared across endpoints. Pattern is lifted directly from `/Users/between2058/Documents/code/phidias-model/ReconViaGen/reconviagen_api.py`.

**Tech Stack:**
- Base: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`
- Runtime: Python 3.10, PyTorch 2.7.1 + cu128, FastAPI 0.115, uvicorn 0.32, slowapi
- Lyra-2 deps: flash-attn (latest, NOT 2.6.3 — 2.6.3 has no sm_120 kernels), transformer_engine[pytorch] ≥ 2.x, MoGe, vendored `vipe` + `depth_anything_3` CUDA extensions (built with `TORCH_CUDA_ARCH_LIST=12.0`)
- Orchestration: docker-compose (single profile / always-on), nginx reverse proxy + rate limit, autoheal sidecar

**Working directory for this plan:** `/Users/between2058/Documents/code/lyra2-service/` (fresh git repo, branch `main`, currently empty).

**Reference files (read these before starting — they live in OTHER repos, copy patterns from them):**
- `/Users/between2058/Documents/code/phidias-model/ReconViaGen/reconviagen_api.py` — canonical FastAPI + queue pattern to copy
- `/Users/between2058/Documents/code/phidias-model/TRELLIS.2/Dockerfile` — Blackwell build pattern (pure pip, no conda, re-lock torch)
- `/Users/between2058/Documents/code/phidias-model/nginx/nginx.conf` — pattern for upstream + rate-limit zone
- `/Users/between2058/Documents/code/phidias-model/docker-compose.yml` — pattern for service block + nginx orchestration
- `/Users/between2058/Documents/code/lyra/Lyra-2/INSTALL.md` — upstream install steps (DO NOT follow conda flow; translate to pip+apt)
- `/Users/between2058/Documents/code/lyra/Lyra-2/lyra_2/_src/inference/lyra2_zoomgs_inference.py` — Step 1 (preset trajectory) CLI
- `/Users/between2058/Documents/code/lyra/Lyra-2/lyra_2/_src/inference/lyra2_custom_traj_inference.py` — Step 1 (custom trajectory) CLI
- `/Users/between2058/Documents/code/lyra/Lyra-2/lyra_2/_src/inference/vipe_da3_gs_recon.py` — Step 2 (GS reconstruction) CLI

**Target file structure (final state of `/Users/between2058/Documents/code/lyra2-service/`):**

```
lyra2-service/                            # fresh git repo
├── Lyra-2/                               # imported from upstream nv-tlabs/lyra
│   ├── (rest of upstream files unchanged)
│   └── lyra_2/_src/inference/
│       ├── _runner_types.py              # NEW
│       ├── lyra2_zoomgs_inference.py     # MODIFY: expose run_zoomgs(...)
│       ├── lyra2_custom_traj_inference.py # MODIFY: expose run_custom_traj(...)
│       └── vipe_da3_gs_recon.py          # MODIFY: expose run_gs_recon(...)
├── Dockerfile                            # NEW: Blackwell-targeted image, build context = repo root
├── requirements-api.txt                  # NEW: FastAPI + slowapi extras
├── lyra2_api.py                          # NEW: FastAPI app
├── tests/
│   └── test_api_submission.py            # NEW
├── docker-compose.yml                    # NEW: standalone orchestration (lyra2 + nginx + autoheal)
├── nginx/
│   └── nginx.conf                        # NEW
├── .env.example                          # NEW
├── .gitignore                            # NEW
├── README.md                             # NEW
├── UPSTREAM.md                           # NEW
└── docs/plans/2026-04-22-lyra2-fastapi-blackwell.md   # THIS FILE
```

**Out of scope (do NOT do in this plan):**
- Interactive GUI (Lyra-2 README "Coming Soon" — not in upstream code yet)
- Training pipeline, data toolkit
- HuggingFace model auto-download into image (mount HF cache instead)
- Multi-GPU sharding within one job
- Switching between DMD and non-DMD checkpoints based on quota
- Integration into `phidias-model/docker-compose.yml` (left as user-side cherry-pick)

---

## Task 1: Bootstrap repo and import Lyra-2 source

**Why:** Set up the empty repo with upstream Lyra-2 source, baseline `.gitignore`, and `UPSTREAM.md` tracking the imported commit.

**Files:**
- Create directory: `/Users/between2058/Documents/code/lyra2-service/Lyra-2/`
- Create: `/Users/between2058/Documents/code/lyra2-service/.gitignore`
- Create: `/Users/between2058/Documents/code/lyra2-service/UPSTREAM.md`
- Create: `/Users/between2058/Documents/code/lyra2-service/README.md`

**Decision: copy, not submodule.** The user has a working clone of nv-tlabs/lyra at `/Users/between2058/Documents/code/lyra`. Copy the `Lyra-2/` subtree from there into this repo. Document the upstream commit in `UPSTREAM.md` so we can re-sync later.

- [ ] **Step 1: Verify upstream clone is current and record SHA.**

```bash
cd /Users/between2058/Documents/code/lyra
git fetch origin
git log --oneline -1 main
```

Expected: a short SHA + commit message (e.g. `52e5079 Add DMD`). Note this SHA — you will write it into `UPSTREAM.md` in Step 3.

- [ ] **Step 2: Copy Lyra-2 subtree into this repo.**

```bash
cd /Users/between2058/Documents/code/lyra2-service
cp -R /Users/between2058/Documents/code/lyra/Lyra-2 ./Lyra-2
find ./Lyra-2 -name '.git' -prune -exec rm -rf {} +
```

Expected: `lyra2-service/Lyra-2/` exists with `lyra_2/`, `assets/`, `requirements.txt`, `INSTALL.md`, `README.md`.

- [ ] **Step 3: Create `UPSTREAM.md`** at repo root with the SHA recorded above:

```markdown
# Upstream tracking

Source: https://github.com/nv-tlabs/lyra
Path imported: `Lyra-2/`
Pinned commit: <SHA from Step 1>
Date imported: 2026-04-22

To re-sync: re-run the copy command in this repo's
`docs/plans/2026-04-22-lyra2-fastapi-blackwell.md` Task 1.

Local modifications applied to upstream source:
- `Lyra-2/lyra_2/_src/inference/_runner_types.py` — added in Task 2
- `Lyra-2/lyra_2/_src/inference/lyra2_zoomgs_inference.py` — refactored to expose `run_zoomgs()` (Task 2)
- `Lyra-2/lyra_2/_src/inference/lyra2_custom_traj_inference.py` — refactored to expose `run_custom_traj()` (Task 2)
- `Lyra-2/lyra_2/_src/inference/vipe_da3_gs_recon.py` — refactored to expose `run_gs_recon()` (Task 2)
```

- [ ] **Step 4: Create `.gitignore`** at repo root:

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.coverage
.venv/
venv/

# Lyra-2 large outputs / weights
Lyra-2/checkpoints/
Lyra-2/outputs/

# Service runtime
outputs/
logs/
hf_cache/

# Local secrets
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
.DS_Store
```

- [ ] **Step 5: Create `README.md`** at repo root (brief, explains how to use):

```markdown
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
```

- [ ] **Step 6: Initial commit.**

```bash
cd /Users/between2058/Documents/code/lyra2-service
git add .gitignore README.md UPSTREAM.md docs/ Lyra-2/
git commit -m "chore: initial import of Lyra-2 source + repo scaffolding"
```

---

## Task 2: Refactor Lyra-2 inference scripts to expose `run()` functions

**Why:** Current scripts only have `main()` driven by `argparse`. FastAPI cannot import argparse Namespaces cleanly, and the GPU worker needs a callable that accepts plain args and returns a result dict.

**Files:**
- Create: `Lyra-2/lyra_2/_src/inference/_runner_types.py`
- Modify: `Lyra-2/lyra_2/_src/inference/lyra2_zoomgs_inference.py`
- Modify: `Lyra-2/lyra_2/_src/inference/lyra2_custom_traj_inference.py`
- Modify: `Lyra-2/lyra_2/_src/inference/vipe_da3_gs_recon.py`

(All paths relative to `/Users/between2058/Documents/code/lyra2-service/`.)

**Refactor strategy:** Each script keeps its `parse_args()` and `if __name__ == "__main__"` block intact (so CLI usage from upstream README still works). We add a new `run_*(...)` function that takes a dataclass of params and returns `{"output_path": ..., "metadata": {...}}`. `main()` becomes a thin wrapper that calls `run_*(...)` with args parsed from CLI.

- [ ] **Step 1: Create shared param dataclasses** at `Lyra-2/lyra_2/_src/inference/_runner_types.py`:

```python
"""Typed parameter objects for in-process invocation of Lyra-2 inference.

Each dataclass mirrors the argparse arguments of one inference CLI script.
Defaults match the script defaults; required fields have no default.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class ZoomGSParams:
    input_image_path: str
    prompt_dir: Optional[str] = None
    prompt: str = ""
    sample_id: Optional[int] = None
    experiment: str = "lyra2"
    checkpoint_dir: str = "checkpoints/model"
    output_path: str = "outputs/zoomgs"
    num_frames_zoom_in: int = 81
    num_frames_zoom_out: int = 241
    zoom_in_strength: float = 0.5
    zoom_out_strength: float = 1.5
    use_dmd: bool = False
    seed: int = 1
    # Add the rest of the argparse args verbatim (with same defaults)
    # — see lyra2_zoomgs_inference.parse_arguments() lines 135–230 for the
    # exhaustive list. Do NOT omit any field — we need them for parameter passthrough.


@dataclass
class CustomTrajParams:
    input_image_path: str
    trajectory_path: str               # path to .npz with w2c, intrinsics, image_height, image_width
    captions_path: str                 # path to .json (frame_idx -> caption)
    experiment: str = "lyra2"
    checkpoint_dir: str = "checkpoints/model"
    output_path: str = "outputs/custom_traj"
    num_frames: int = 481
    use_dmd: bool = False
    pose_scale: float = 1.0
    seed: int = 1
    # Likewise: copy any additional args from lyra2_custom_traj_inference.parse_arguments()


@dataclass
class GSReconParams:
    input_video_path: str
    output_dir: Optional[str] = None   # default: alongside input video
```

- [ ] **Step 2: Refactor `lyra2_zoomgs_inference.py`.** Just above `if __name__ == "__main__"`:

```python
def run_zoomgs(params: "ZoomGSParams") -> dict:
    """In-process entry point for FastAPI. Returns paths produced by inference."""
    from argparse import Namespace
    from lyra_2._src.inference._runner_types import ZoomGSParams  # noqa: F401
    args = Namespace(**params.__dict__)
    return _execute(args)


def _execute(args) -> dict:
    # Move the body of the existing main() function here verbatim,
    # except: at the end, return the dict described below.
    ...


def main() -> None:
    args = parse_arguments()
    _execute(args)
```

The `_execute(args)` body must be the existing `main()` body verbatim, with one addition: just before returning, return:

```python
return {
    "output_dir": str(output_root),                     # e.g. outputs/zoomgs/<sample_id>
    "video_path": str(combined_video_path),             # outputs/zoomgs/videos/<sample_id>.mp4
    "zoom_in_path": str(zoom_in_path),
    "zoom_out_path": str(zoom_out_path),
}
```

(Refer to the existing `main()` to find the variable names — they will be named slightly differently. Use whatever variable already holds the final mp4 path.)

- [ ] **Step 3: Repeat the same refactor for `lyra2_custom_traj_inference.py`.** Add `run_custom_traj(params: CustomTrajParams) -> dict` returning `{"output_dir": str, "video_path": str}`.

- [ ] **Step 4: Repeat for `vipe_da3_gs_recon.py`.** Add `run_gs_recon(params: GSReconParams) -> dict` returning:

```python
return {
    "output_dir": str(output_dir),                      # <input_basename>_gs_ours/
    "ply_path": str(output_dir / "reconstructed_scene.ply"),
    "trajectory_video_path": str(output_dir / "gs_trajectory.mp4"),
}
```

- [ ] **Step 5: Smoke-test that `_runner_types.py` imports cleanly** (importability check only — no GPU calls):

```bash
cd /Users/between2058/Documents/code/lyra2-service/Lyra-2
python -c "
import importlib, sys
sys.path.insert(0, '.')
importlib.import_module('lyra_2._src.inference._runner_types')
print('OK')
"
```

Expected: `OK`. (The other modules import torch and require the full env; deferring real-load test to in-Docker test in Task 9.)

- [ ] **Step 6: Commit.**

```bash
cd /Users/between2058/Documents/code/lyra2-service
git add Lyra-2/lyra_2/_src/inference/_runner_types.py \
        Lyra-2/lyra_2/_src/inference/lyra2_zoomgs_inference.py \
        Lyra-2/lyra_2/_src/inference/lyra2_custom_traj_inference.py \
        Lyra-2/lyra_2/_src/inference/vipe_da3_gs_recon.py
git commit -m "refactor(lyra2): expose run_*() entry points for in-process use"
```

---

## Task 3: API skeleton (logging, app, job store, GPU worker, /health)

**Why:** Establish the bones of `lyra2_api.py` mirroring `reconviagen_api.py`. No endpoint logic yet; just plumbing that a /health probe can talk to.

**Files:**
- Create: `requirements-api.txt`
- Create: `lyra2_api.py`

(All paths relative to `/Users/between2058/Documents/code/lyra2-service/`.)

- [ ] **Step 1: Write `requirements-api.txt`.** Mirror `phidias-model/ReconViaGen/requirements-api.txt`'s philosophy (torch / cuda-bound packages installed in Dockerfile, NOT here). Lyra-2 already pins most of its deps in `Lyra-2/requirements.txt`; this file only adds the API layer.

```
# =============================================================================
# Lyra-2 API — Python dependencies
#
# NOT listed here (installed in Dockerfile or Lyra-2/requirements.txt):
#   torch / torchvision    — Dockerfile (cu128 wheels)
#   flash-attn             — Dockerfile (Blackwell needs latest, NOT 2.6.3)
#   transformer_engine     — Dockerfile (--no-build-isolation)
#   MoGe                   — Dockerfile (git install)
#   vipe / depth_anything_3 — Dockerfile (vendored, --no-build-isolation -e)
# =============================================================================

# ── Web framework ─────────────────────────────────────────────────────────────
fastapi==0.115.5
uvicorn[standard]==0.32.1
python-multipart==0.0.17
slowapi==0.1.9

# ── Health / GPU monitoring ──────────────────────────────────────────────────
nvidia-ml-py
pydantic>=2.0.0
```

- [ ] **Step 2: Create `lyra2_api.py` skeleton.** Copy the structural pattern from `/Users/between2058/Documents/code/phidias-model/ReconViaGen/reconviagen_api.py` lines 1–460 (logging block, `JobRecord` dataclass, queue, GPU worker loop, helpers, startup/shutdown, `/health`). Adapt:

  - App title: `"Lyra-2 API (image → 3DGS world)"`
  - Output dir env: `OUTPUT_DIR = os.environ.get("LYRA2_OUTPUT_DIR", "/app/outputs")`
  - Two model handles instead of one: `_video_pipeline` (Step 1), `_gs_pipeline` (Step 2)
  - `ensure_model_loaded()` becomes `ensure_video_pipeline_loaded()` and `ensure_gs_pipeline_loaded()`, both lazy
  - `QUEUE_MAX_SIZE` default: `int(os.environ.get("QUEUE_MAX_SIZE", "3"))` — Step 1 jobs run for minutes, queue should be small
  - VRAM leak threshold env: `LYRA2_TORCH_VRAM_LEAK_LIMIT_GB` (default 80.0 — Step 1 video model is large)

- [ ] **Step 3: Verify the file imports cleanly** (no GPU required, just syntax check):

```bash
cd /Users/between2058/Documents/code/lyra2-service
python -c "import ast; ast.parse(open('lyra2_api.py').read()); print('parse OK')"
```

Expected: `parse OK`.

- [ ] **Step 4: Commit.**

```bash
cd /Users/between2058/Documents/code/lyra2-service
git add requirements-api.txt lyra2_api.py
git commit -m "feat(lyra2): add FastAPI skeleton with logging, queue, /health"
```

---

## Task 4: Job submission endpoints

**Why:** Three POST endpoints that build a sync_fn closure over the refactored `run_*()` functions and submit to the GPU worker.

**Files:**
- Modify: `lyra2_api.py`
- Create: `tests/test_api_submission.py`

- [ ] **Step 1: Add `POST /jobs/image-to-video`.** Accepts:
  - `mode: Literal["preset", "custom"]` (form field)
  - `image: UploadFile` (required, jpg/png)
  - `caption: Optional[str]` (form, used in preset mode if no caption file)
  - `trajectory: Optional[UploadFile]` (required when mode=custom; .npz)
  - `captions_json: Optional[UploadFile]` (required when mode=custom; .json)
  - Common params: `num_frames_zoom_in`, `num_frames_zoom_out`, `num_frames` (custom), `zoom_in_strength`, `zoom_out_strength`, `pose_scale`, `use_dmd`, `seed`

  Body of handler:

  ```python
  request_id = str(uuid.uuid4())
  req_dir = os.path.join(OUTPUT_DIR, request_id)
  os.makedirs(req_dir, exist_ok=True)

  image_path = os.path.join(req_dir, "input.png")
  with open(image_path, "wb") as f: shutil.copyfileobj(image.file, f)

  if mode == "preset":
      params = ZoomGSParams(
          input_image_path=image_path,
          prompt=caption or "",
          output_path=req_dir,
          num_frames_zoom_in=num_frames_zoom_in,
          num_frames_zoom_out=num_frames_zoom_out,
          zoom_in_strength=zoom_in_strength,
          zoom_out_strength=zoom_out_strength,
          use_dmd=use_dmd,
          seed=seed,
      )
      def sync_fn():
          ensure_video_pipeline_loaded()
          return run_zoomgs(params)
  else:  # mode == "custom"
      traj_path = os.path.join(req_dir, "trajectory.npz")
      cap_path  = os.path.join(req_dir, "captions.json")
      with open(traj_path, "wb") as f: shutil.copyfileobj(trajectory.file, f)
      with open(cap_path, "wb")  as f: shutil.copyfileobj(captions_json.file, f)
      params = CustomTrajParams(
          input_image_path=image_path,
          trajectory_path=traj_path,
          captions_path=cap_path,
          output_path=req_dir,
          num_frames=num_frames,
          use_dmd=use_dmd,
          pose_scale=pose_scale,
          seed=seed,
      )
      def sync_fn():
          ensure_video_pipeline_loaded()
          return run_custom_traj(params)

  queue_pos = await _submit_to_gpu_worker(sync_fn, request_id, req_dir)
  return {"job_id": request_id, "status": "queued", "queue_position": queue_pos}
  ```

- [ ] **Step 2: Add `POST /jobs/video-to-gs`.** Accepts:
  - `video: UploadFile` (mp4)

  ```python
  request_id = str(uuid.uuid4())
  req_dir = os.path.join(OUTPUT_DIR, request_id)
  os.makedirs(req_dir, exist_ok=True)
  video_path = os.path.join(req_dir, "input.mp4")
  with open(video_path, "wb") as f: shutil.copyfileobj(video.file, f)

  params = GSReconParams(input_video_path=video_path, output_dir=req_dir)
  def sync_fn():
      ensure_gs_pipeline_loaded()
      return run_gs_recon(params)

  queue_pos = await _submit_to_gpu_worker(sync_fn, request_id, req_dir)
  return {"job_id": request_id, "status": "queued", "queue_position": queue_pos}
  ```

- [ ] **Step 3: Add `POST /jobs/image-to-gs`** (chains Step 1 + Step 2). Same input shape as `/jobs/image-to-video`. Body builds a `sync_fn` that calls `run_zoomgs` (or `run_custom_traj`) AND then `run_gs_recon` on its output:

  ```python
  def sync_fn():
      ensure_video_pipeline_loaded()
      step1 = run_zoomgs(params_zoom)  # or run_custom_traj
      ensure_gs_pipeline_loaded()
      step2_params = GSReconParams(input_video_path=step1["video_path"], output_dir=req_dir)
      step2 = run_gs_recon(step2_params)
      return {**step1, **step2}
  ```

- [ ] **Step 4: Verify with mocked GPU.** Add a unit test at `tests/test_api_submission.py`:

  ```python
  """Unit tests for job submission — GPU calls are mocked.
  Run: pytest tests/test_api_submission.py -v
  """
  import io
  import sys, os
  import pytest
  from unittest.mock import patch
  from fastapi.testclient import TestClient

  # Make the repo root importable so `import lyra2_api` works
  sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


  @pytest.fixture
  def client():
      with patch("lyra2_api.run_zoomgs", return_value={"video_path": "/tmp/x.mp4", "output_dir": "/tmp"}), \
           patch("lyra2_api.run_custom_traj", return_value={"video_path": "/tmp/y.mp4", "output_dir": "/tmp"}), \
           patch("lyra2_api.run_gs_recon", return_value={"ply_path": "/tmp/z.ply", "output_dir": "/tmp"}), \
           patch("lyra2_api.ensure_video_pipeline_loaded"), \
           patch("lyra2_api.ensure_gs_pipeline_loaded"), \
           patch("lyra2_api._check_vram_available", return_value=True):
          import lyra2_api
          with TestClient(lyra2_api.app) as c:
              yield c

  def test_image_to_video_preset(client):
      png = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
      r = client.post(
          "/jobs/image-to-video",
          data={"mode": "preset", "caption": "test", "use_dmd": "false"},
          files={"image": ("test.png", png, "image/png")},
      )
      assert r.status_code == 200, r.text
      assert r.json()["status"] == "queued"
      assert "job_id" in r.json()

  def test_video_to_gs(client):
      mp4 = io.BytesIO(b"\x00" * 1024)
      r = client.post(
          "/jobs/video-to-gs",
          files={"video": ("test.mp4", mp4, "video/mp4")},
      )
      assert r.status_code == 200, r.text
      assert r.json()["status"] == "queued"

  def test_image_to_gs_preset(client):
      png = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
      r = client.post(
          "/jobs/image-to-gs",
          data={"mode": "preset", "caption": "test"},
          files={"image": ("test.png", png, "image/png")},
      )
      assert r.status_code == 200, r.text
      assert r.json()["status"] == "queued"
  ```

  Run:
  ```bash
  cd /Users/between2058/Documents/code/lyra2-service
  pip install fastapi httpx pytest python-multipart slowapi  # one-time, host machine
  pytest tests/test_api_submission.py -v
  ```
  Expected: 3 PASS.

- [ ] **Step 5: Commit.**

```bash
cd /Users/between2058/Documents/code/lyra2-service
git add lyra2_api.py tests/
git commit -m "feat(lyra2): add image-to-video, video-to-gs, image-to-gs endpoints"
```

---

## Task 5: Job lifecycle endpoints (status, cancel, download)

**Why:** Clients need to poll, cancel queued jobs, and fetch outputs.

**Files:**
- Modify: `lyra2_api.py`
- Modify: `tests/test_api_submission.py` (extend)

- [ ] **Step 1: Copy `GET /jobs/{job_id}`, `DELETE /jobs/{job_id}`, `GET /download/{request_id}/{file_name}` from `/Users/between2058/Documents/code/phidias-model/ReconViaGen/reconviagen_api.py` lines 795–900 verbatim.** No structural changes needed — the `_jobs` dict and `_cancelled_jobs` set are identical in both.

  Allowed download extensions for Lyra-2: `.mp4`, `.ply`, `.json`, `.png`. Edit the file extension whitelist accordingly.

- [ ] **Step 2: Extend `tests/test_api_submission.py`:**

```python
def test_poll_unknown_job(client):
    r = client.get("/jobs/does-not-exist")
    assert r.status_code == 404
    assert r.json()["detail"]["error_code"] == "JOB_NOT_FOUND"

def test_cancel_unknown_job(client):
    r = client.delete("/jobs/does-not-exist")
    assert r.status_code == 404

def test_full_lifecycle(client):
    """Submit, poll until completed (mocked GPU returns instantly)."""
    import time
    r = client.post(
        "/jobs/video-to-gs",
        files={"video": ("test.mp4", io.BytesIO(b"\x00" * 1024), "video/mp4")},
    )
    job_id = r.json()["job_id"]
    for _ in range(30):
        s = client.get(f"/jobs/{job_id}")
        if s.json()["status"] == "completed":
            break
        time.sleep(0.1)
    assert s.json()["status"] == "completed"
    assert s.json()["result"]["ply_path"] == "/tmp/z.ply"
```

Run:
```bash
cd /Users/between2058/Documents/code/lyra2-service
pytest tests/test_api_submission.py -v
```
Expected: 6 PASS.

- [ ] **Step 3: Commit.**

```bash
git add lyra2_api.py tests/test_api_submission.py
git commit -m "feat(lyra2): add /jobs/{id} polling, cancel, /download endpoints"
```

---

## Task 6: Hardening (OOM detection, rate limit, queue cap, VRAM guard)

**Why:** Lyra-2 jobs are minutes-long and large; we MUST detect OOM/leaks and let the autoheal sidecar restart us. We also need slowapi to enforce in-process rate limits before the queue.

**Files:**
- Modify: `lyra2_api.py`

- [ ] **Step 1: Lift the OOM / force-restart pattern from `/Users/between2058/Documents/code/phidias-model/ReconViaGen/reconviagen_api.py` lines 322–410** (`flush_gpu`, `_set_force_restart`, `_check_vram_available`, `_get_vram_used_gb`, `_get_torch_vram_gb`, the `_force_restart` global). This block is environment-agnostic — copy verbatim. The `_gpu_worker_loop` exception handler that calls `_set_force_restart` is already in the skeleton from Task 3; ensure it remains.

- [ ] **Step 2: Add slowapi rate limiter.** Mirror the pattern at `reconviagen_api.py` (search for `_limiter`):

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

_limiter = Limiter(key_func=get_remote_address)
app.state.limiter = _limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
```

Decorate each POST endpoint with `@_limiter.limit("10/minute")` (lower than reconviagen's 15/min — Lyra-2 jobs run longer).

- [ ] **Step 3: Verify rate limit fires.** Add to test file:

```python
def test_rate_limit_engages(client):
    """11th POST in a minute should 429."""
    last = None
    for _ in range(12):
        last = client.post(
            "/jobs/image-to-video",
            data={"mode": "preset", "caption": "x"},
            files={"image": ("test.png", io.BytesIO(b"\x00" * 100), "image/png")},
        )
    assert last.status_code == 429
```

Run: `pytest tests/test_api_submission.py::test_rate_limit_engages -v`. Expected: PASS.

- [ ] **Step 4: Commit.**

```bash
git add lyra2_api.py tests/test_api_submission.py
git commit -m "feat(lyra2): add OOM detection, VRAM guard, slowapi rate limiting"
```

---

## Task 7: Dockerfile

**Why:** This is the load-bearing piece — translates Lyra-2's conda-based INSTALL.md into a pure pip + apt Dockerfile that builds for sm_120.

**Files:**
- Create: `Dockerfile` (at repo root; build context = repo root)

- [ ] **Step 1: Write Dockerfile.** Mirror `/Users/between2058/Documents/code/phidias-model/TRELLIS.2/Dockerfile` line-by-line — only the deps differ. Note: build context is the repo root (`lyra2-service/`), so `COPY` paths use the `Lyra-2/` prefix:

```dockerfile
# =============================================================================
# Lyra-2 API — Docker Image
#
# Target: NVIDIA RTX Pro 6000 Blackwell (sm_120)
# CUDA 12.8  |  Python 3.10  |  PyTorch 2.7.1+cu128
#
# Build context: repo root (contains Lyra-2/ + lyra2_api.py + requirements-api.txt)
# Build:  docker build -t lyra2-api:latest .
# Run:    docker run --gpus all -p 52071:52071 \
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
# gcc-13 replaces conda's gcc=13.3.0; libeigen3-dev replaces conda's eigen.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    build-essential ninja-build cmake git wget curl \
    gcc-13 g++-13 \
    libeigen3-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libegl1-mesa-dev libgles2-mesa-dev libgomp1 ffmpeg libjpeg-dev \
 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 \
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

ENV PYTHONPATH=/app/Lyra-2:${PYTHONPATH}
RUN mkdir -p /app/logs /app/outputs /app/hf_cache

EXPOSE 52071

HEALTHCHECK \
    --interval=30s \
    --timeout=15s \
    --start-period=300s \
    --retries=5 \
    CMD curl -f http://localhost:52071/health || exit 1

CMD ["python", "-m", "uvicorn", "lyra2_api:app", \
     "--host", "0.0.0.0", \
     "--port", "52071", \
     "--workers", "1", \
     "--log-level", "info"]
```

- [ ] **Step 2: Lint with hadolint** (optional but cheap):

```bash
docker run --rm -i hadolint/hadolint < /Users/between2058/Documents/code/lyra2-service/Dockerfile
```

Expected: no SC2046/DL3008 errors. Warnings about pinning apt versions are OK to ignore (matches TRELLIS.2 pattern).

- [ ] **Step 3: Commit.**

```bash
cd /Users/between2058/Documents/code/lyra2-service
git add Dockerfile
git commit -m "feat(lyra2): add Dockerfile targeting Blackwell sm_120"
```

---

## Task 8: Self-contained orchestration (docker-compose, nginx, .env)

**Why:** This repo ships standalone — `docker compose up -d` should bring up nginx + lyra2 + autoheal without needing phidias-model. (Cherry-picking into phidias-model later is a separate exercise.)

**Files:**
- Create: `docker-compose.yml`
- Create: `nginx/nginx.conf`
- Create: `.env.example`

- [ ] **Step 1: Create `docker-compose.yml`.** Three services: `nginx` (entry point, rate limit), `lyra2` (the API), `autoheal` (restarts unhealthy containers).

```yaml
version: "3.8"

# =============================================================================
# Lyra-2 Service — standalone orchestration
#
# Quick start:
#   cp .env.example .env
#   $EDITOR .env
#   docker compose up -d
#
# Endpoints (via nginx):
#   POST http://localhost:52071/jobs/image-to-gs
#   POST http://localhost:52071/jobs/image-to-video
#   POST http://localhost:52071/jobs/video-to-gs
#   GET  http://localhost:52071/jobs/{id}
#   GET  http://localhost:52071/health
#   GET  http://localhost:52071/docs
#
# Bypass nginx for debugging: uncomment the lyra2 ports section.
# =============================================================================

networks:
  lyra2-net:
    driver: bridge

services:
  nginx:
    image: nginx:1.27-alpine
    container_name: lyra2-nginx
    ports:
      - "${LYRA2_PORT:-52071}:52071"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
    networks:
      - lyra2-net
    depends_on:
      - lyra2
    restart: unless-stopped

  autoheal:
    image: willfarrell/autoheal:latest
    container_name: lyra2-autoheal
    environment:
      - AUTOHEAL_CONTAINER_LABEL=autoheal
      - AUTOHEAL_INTERVAL=30
      - AUTOHEAL_START_PERIOD=300       # Lyra-2 first-load is slow
      - AUTOHEAL_DEFAULT_STOP_TIMEOUT=30
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    restart: unless-stopped

  lyra2:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        TORCH_CUDA_ARCH_LIST: "${TORCH_CUDA_ARCH_LIST:-12.0}"
        MAX_JOBS: "${MAX_JOBS:-4}"
        http_proxy: "${HTTP_PROXY:-}"
        https_proxy: "${HTTPS_PROXY:-}"
        no_proxy: "localhost,127.0.0.1"

    container_name: lyra2-api

    # Ports are NOT exposed to host — traffic goes through nginx.
    # Uncomment for direct debug access:
    # ports:
    #   - "127.0.0.1:52071:52071"

    volumes:
      - ${HF_CACHE_HOST_PATH}:/app/hf_cache:rw
      - ${LYRA2_CHECKPOINTS_PATH:-./Lyra-2/checkpoints}:/app/Lyra-2/checkpoints:ro
      - ./logs:/app/logs
      - ./outputs:/app/outputs

    environment:
      - HF_HOME=/app/hf_cache
      - TRANSFORMERS_CACHE=/app/hf_cache
      - HUGGINGFACE_HUB_CACHE=/app/hf_cache
      - QUEUE_MAX_SIZE=${LYRA2_QUEUE_MAX_SIZE:-3}
      - NVML_GPU_INDEX=${LYRA2_GPU_ID:-0}
      - TORCH_VRAM_LEAK_LIMIT_GB=${LYRA2_TORCH_VRAM_LEAK_LIMIT_GB:-80.0}
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["${LYRA2_GPU_ID:-0}"]
              capabilities: [gpu]

    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:52071/health || exit 1"]
      interval: 30s
      timeout: 15s
      retries: 5
      start_period: 300s

    labels:
      autoheal: "true"

    shm_size: "8gb"
    networks:
      - lyra2-net
    restart: unless-stopped
```

- [ ] **Step 2: Create `nginx/nginx.conf`.** Single upstream → lyra2 backend, with rate limit and long read timeout for minutes-long inference.

```nginx
events {
    worker_connections 1024;
}

http {
    # POST-only rate limit (GETs for /jobs/{id} polling are not limited)
    map $request_method $lyra2_rate_key { POST $binary_remote_addr; default ""; }
    limit_req_zone $lyra2_rate_key zone=lyra2_limit:10m rate=60r/m;

    # Conn limit (POST only)
    map $request_method $conn_limit_key { POST $binary_remote_addr; default ""; }
    limit_conn_zone $conn_limit_key zone=conn_limit:10m;

    # Docker internal DNS
    resolver 127.0.0.11 valid=30s ipv6=off;

    client_max_body_size 200m;

    proxy_set_header Host              $host;
    proxy_set_header X-Real-IP         $remote_addr;
    proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_next_upstream off;
    proxy_buffering off;
    proxy_request_buffering off;

    limit_req_status  429;
    limit_conn_status 429;

    # ---------------------------------------------------------------------------
    # lyra2  :52071
    # Inference: ~9 min (Step 1 no DMD) / ~2 min (full pipeline + DMD)
    # Queue maxsize=3
    # ---------------------------------------------------------------------------
    server {
        listen 52071;

        error_page 429 @rate_limited;
        error_page 504 @gateway_timeout;

        location / {
            limit_req  zone=lyra2_limit burst=10 nodelay;
            limit_conn conn_limit 2;

            set $upstream_lyra2 http://lyra2:52071;
            proxy_pass            $upstream_lyra2$request_uri;
            proxy_read_timeout    1800s;
            proxy_connect_timeout 10s;
            proxy_send_timeout    60s;
        }

        location @rate_limited {
            default_type application/json;
            return 429 '{"detail":"rate_limit_exceeded","message":"Too many requests, please slow down"}';
        }

        location @gateway_timeout {
            default_type application/json;
            return 504 '{"detail":"gateway_timeout","message":"Inference exceeded server time limit"}';
        }
    }
}
```

- [ ] **Step 3: Create `.env.example`** at repo root:

```bash
# =============================================================================
# Lyra-2 Service — machine-specific configuration
#
# Usage:
#   cp .env.example .env
#   vim .env
#   docker compose up -d
#
# .env is gitignored — never commit it.
# =============================================================================

# ── Required: HuggingFace cache on host (for model downloads) ────────────────
HF_CACHE_HOST_PATH=/data/hf_cache

# ── Required: Lyra-2 checkpoint directory ────────────────────────────────────
# Pre-download with:
#   cd Lyra-2 && pip install huggingface_hub
#   huggingface-cli download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir .
LYRA2_CHECKPOINTS_PATH=./Lyra-2/checkpoints

# ── Build settings ───────────────────────────────────────────────────────────
# 12.0 = RTX Pro 6000 Blackwell only (faster build).
# Multi-GPU compatibility: TORCH_CUDA_ARCH_LIST=8.0;8.6;8.9;9.0;10.0;12.0
TORCH_CUDA_ARCH_LIST=12.0

# Parallel compile jobs (raise on multi-core build machines)
MAX_JOBS=4

# Optional corporate proxy (leave empty otherwise)
HTTP_PROXY=
HTTPS_PROXY=

# ── Runtime ──────────────────────────────────────────────────────────────────

# GPU index (run `nvidia-smi` to list)
LYRA2_GPU_ID=0

# Public port (nginx listens here)
LYRA2_PORT=52071

# Inference queue depth — each job runs minutes
LYRA2_QUEUE_MAX_SIZE=3

# Torch VRAM ceiling for leak detection (GB) — Lyra-2 ~70 GB peak
LYRA2_TORCH_VRAM_LEAK_LIMIT_GB=80.0
```

- [ ] **Step 4: Validate compose syntax locally** (no images built yet — just syntax):

```bash
cd /Users/between2058/Documents/code/lyra2-service
HF_CACHE_HOST_PATH=/tmp/x docker compose config > /dev/null
```

Expected: no errors.

- [ ] **Step 5: Validate nginx config syntax:**

```bash
docker run --rm -v /Users/between2058/Documents/code/lyra2-service/nginx/nginx.conf:/etc/nginx/nginx.conf:ro \
    nginx:1.27-alpine nginx -t
```

Expected: `nginx: configuration file /etc/nginx/nginx.conf test is successful`. (The `lyra2:52071` upstream resolution warning is expected — it's resolved at runtime when the lyra2 container starts.)

- [ ] **Step 6: Commit.**

```bash
cd /Users/between2058/Documents/code/lyra2-service
git add docker-compose.yml nginx/ .env.example
git commit -m "feat(lyra2): add standalone docker-compose, nginx, env"
```

---

## Task 9: Build & smoke test on Blackwell host

**Why:** Local mac has no GPU. Build and runtime verification MUST happen on the actual RTX Pro 6000 box. This task is a hand-off checklist.

**Prerequisites on the Blackwell host:**
- Docker + nvidia-container-toolkit installed
- This repo cloned/synced to the Blackwell box at `<HOST_PATH>/lyra2-service`
- HuggingFace cache path exists (set `HF_CACHE_HOST_PATH`)
- Lyra-2 checkpoints downloaded:
  ```bash
  cd <HOST_PATH>/lyra2-service/Lyra-2
  pip install huggingface_hub
  huggingface-cli download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir .
  ```

**Files:** none (verification only)

- [ ] **Step 1: Configure and build.**

```bash
cd <HOST_PATH>/lyra2-service
cp .env.example .env
$EDITOR .env                           # set HF_CACHE_HOST_PATH at minimum
docker compose build lyra2 2>&1 | tee /tmp/lyra2-build.log
```

Expected: build completes (~30–60 min on first build). Watch for these failure modes:

| Failure | Fix |
|---|---|
| `flash-attn` compile error referencing sm_120 | confirm using latest flash-attn (not 2.6.3); may need `git+https://github.com/Dao-AILab/flash-attention.git@main` |
| `transformer_engine` error about cudart not found | confirm the symlink step ran; check `ls /usr/local/lib/python3.10/dist-packages/nvidia/cudart` |
| `vipe` build can't find Eigen | confirm `apt install libeigen3-dev` succeeded; `USE_SYSTEM_EIGEN=1` is set |
| torch downgraded by transitive dep | confirm the final `--force-reinstall torch==2.7.1` step ran |

- [ ] **Step 2: Start service and verify /health.**

```bash
docker compose up -d
sleep 60
curl -s http://localhost:52071/health | python -m json.tool
```

Expected: `{"status": "ok", "model_loaded": false, "gpu_busy": false, ...}` (model_loaded=false is expected — lazy load).

- [ ] **Step 3: Smoke-test `/jobs/video-to-gs`** (Step 2 only — fastest path, ~1 min):

```bash
curl -X POST http://localhost:52071/jobs/video-to-gs \
    -F "video=@/path/to/test.mp4" | tee /tmp/job1.json
JOB_ID=$(jq -r .job_id /tmp/job1.json)

until [ "$(curl -s http://localhost:52071/jobs/$JOB_ID | jq -r .status)" = "completed" ]; do
    sleep 10
    curl -s http://localhost:52071/jobs/$JOB_ID | jq .
done

PLY=$(curl -s http://localhost:52071/jobs/$JOB_ID | jq -r '.result.ply_path | split("/")[-1]')
curl -O http://localhost:52071/download/$JOB_ID/$PLY
```

Expected: a non-zero `reconstructed_scene.ply` lands in CWD. Open in Meshlab/CloudCompare to eyeball.

- [ ] **Step 4: Smoke-test `/jobs/image-to-gs`** with `use_dmd=true` (full pipeline, ~2 min):

```bash
curl -X POST http://localhost:52071/jobs/image-to-gs \
    -F "mode=preset" \
    -F "caption=a tabletop scene with toy cars" \
    -F "use_dmd=true" \
    -F "image=@<HOST_PATH>/lyra2-service/Lyra-2/assets/samples/sample_0.jpg"
```
(Poll as in Step 3.) Expected: `result` contains both `video_path` and `ply_path`. Eyeball both outputs.

- [ ] **Step 5: Verify nginx rate limiting** (POST 11 times rapidly):

```bash
for i in $(seq 1 11); do
  curl -s -o /dev/null -w "%{http_code} " -X POST http://localhost:52071/jobs/video-to-gs \
      -F "video=@/path/to/test.mp4"
done; echo
```

Expected: ten `200`s then one `429` (or earlier if queue fills first → 503).

- [ ] **Step 6: Document any deviations** in `UPSTREAM.md` so re-syncs don't lose the fix.

- [ ] **Step 7: Commit any post-build fixes** (e.g. version pin updates in Dockerfile).

---

## Notes on potential post-launch follow-ups (NOT in this plan)

- **Cherry-pick into phidias-model orchestration**: copy `lyra2:` block from this repo's `docker-compose.yml` (under a `profiles: ["lyra2"]`), copy the nginx upstream block, and add LYRA2_* env vars. Use `${LYRA2_BUILD_CONTEXT:-../lyra2-service}` as build context.
- **Dual-GPU split**: today both Step 1 and Step 2 share one GPU and lock. With a second card, spin a second `lyra2-gs-only` service on a different port using the same image; have a thin gateway that routes `/jobs/image-to-gs` → Step 1 on GPU0, then proxies result video to Step 2 service on GPU1.
- **DMD vs non-DMD model variants**: today `use_dmd` toggles a LoRA at request time. If user demand splits cleanly, consider two warm processes.
- **Background HF cache pre-warm at build time** (currently lazy on first request — adds ~3 min to first /jobs/* call).
