import os

# Must be set before importing torch — switches CUDA allocator to expandable
# segments (VMM-backed), which allows much finer-grained memory return to the
# driver after empty_cache().  Without this, PyTorch holds large fixed-size
# pool blocks that cannot be partially released, causing nvidia-smi to show
# far more VRAM used than torch.cuda.memory_allocated() reports.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:512",
)

import shutil
import sys
import uuid
import gc
import asyncio
import datetime
import time
from dataclasses import dataclass
import logging
import logging.handlers
from typing import Any, Literal, Optional

# torch may not be installed on the host — guard so the module imports cleanly
# during host-side syntax checks.  In Docker this always succeeds.
try:
    import torch
except (ImportError, ModuleNotFoundError):
    torch = None

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

try:
    import pynvml
except (ImportError, ModuleNotFoundError):
    pynvml = None

# ---------------------------------------------------------------------------
# Refactored Lyra-2 entry points (from Task 2 — do NOT call here, just import)
#
# These imports load torch / heavy deps when reconciled inside Docker.  On the
# host machine they will fail because torch isn't installed.  We swallow the
# error so `python -c "import lyra2_api"` succeeds during host-side checks.
# The API server will refuse jobs at runtime if these are still None.
# ---------------------------------------------------------------------------
sys.path.insert(0, "/app/Lyra-2")  # for in-Docker runtime; harmless on host
try:
    from lyra_2._src.inference.lyra2_zoomgs_inference import run_zoomgs
    from lyra_2._src.inference.lyra2_custom_traj_inference import run_custom_traj
    from lyra_2._src.inference.vipe_da3_gs_recon import run_gs_recon
    from lyra_2._src.inference._runner_types import (
        ZoomGSParams,
        CustomTrajParams,
        GSReconParams,
    )
except (ImportError, ModuleNotFoundError):
    # Heavy deps (torch, transformer_engine, etc.) only available inside Docker.
    # API server will refuse jobs at runtime if these are None.
    run_zoomgs = run_custom_traj = run_gs_recon = None
    ZoomGSParams = CustomTrajParams = GSReconParams = None


# =============================================================================
# Logging 設定
# =============================================================================

os.makedirs("/app/logs", exist_ok=True) if os.path.isdir("/app") else os.makedirs(
    os.environ.get("LYRA2_LOG_DIR", "/tmp/lyra2_logs"), exist_ok=True
)
_LOG_DIR = "/app/logs" if os.path.isdir("/app") else os.environ.get(
    "LYRA2_LOG_DIR", "/tmp/lyra2_logs"
)


class TaiwanFormatter(logging.Formatter):
    """Timestamps always in UTC+8 (Taiwan), independent of system timezone."""
    _TZ = datetime.timezone(datetime.timedelta(hours=8))

    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created, tz=self._TZ)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S")


class TaiwanAccessFormatter(logging.Formatter):
    """Format uvicorn.access records to match app log style.

    uvicorn passes (client_addr, method, full_path, http_version, status_code)
    as positional args on the LogRecord.  We parse them here so the output
    matches the app log timestamp/levelname layout.
    """
    _TZ = datetime.timezone(datetime.timedelta(hours=8))

    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created, tz=self._TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def format(self, record):
        ts = self.formatTime(record)
        try:
            client_addr, method, full_path, http_version, status_code = record.args
            if isinstance(client_addr, tuple):
                client_addr = "%s:%s" % client_addr
            elif not client_addr:
                client_addr = "(unknown)"
            return (
                f'{ts} [ACCESS  ] {client_addr} - '
                f'"{method} {full_path} HTTP/{http_version}" {status_code}'
            )
        except Exception:
            return f"{ts} [ACCESS  ] {record.getMessage()}"


class HealthCheckFilter(logging.Filter):
    """Suppress uvicorn.access entries for GET /health (stops healthcheck log spam)."""
    def filter(self, record):
        return "GET /health" not in record.getMessage()


def _rotating_file_handler(filename: str, formatter: logging.Formatter) -> logging.Handler:
    handler = logging.handlers.TimedRotatingFileHandler(
        f"{_LOG_DIR}/{filename}",
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    return handler


def _reconfigure_uvicorn_access_logger() -> None:
    """Apply our formatter to uvicorn.access AFTER uvicorn's dictConfig has run.

    uvicorn calls logging.config.dictConfig() during startup, replacing any
    handlers set at import time.  This function must be called from the startup
    event handler so our formatter takes effect on the final handler list.
    """
    _access_fmt = TaiwanAccessFormatter()
    uv = logging.getLogger("uvicorn.access")
    uv.handlers.clear()
    uv.filters.clear()
    uv.propagate = False
    uv.addFilter(HealthCheckFilter())
    ch = logging.StreamHandler()
    ch.setFormatter(_access_fmt)
    uv.addHandler(ch)
    uv.addHandler(_rotating_file_handler("access.log", _access_fmt))


# Formatters
_fmt = TaiwanFormatter("%(asctime)s [%(levelname)-8s] %(message)s")

# app logger — business logic, stdout + rotating file
app_logger = logging.getLogger("lyra2_api")
app_logger.setLevel(logging.DEBUG)
app_logger.propagate = False
_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)
app_logger.addHandler(_ch)
app_logger.addHandler(_rotating_file_handler("app.log", _fmt))

# uvicorn.error (server startup / shutdown messages) → file
# Note: uvicorn.access is configured in the startup event (see _reconfigure_uvicorn_access_logger)
logging.getLogger("uvicorn").addHandler(_rotating_file_handler("uvicorn.log", _fmt))

# =============================================================================

app = FastAPI(title="Lyra-2 API (image → 3DGS world)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Per-IP rate limiting (slowapi)
#
# Limiter is initialised here so app.state.limiter is available, but specific
# endpoint decorators are wired in a later task.
# ---------------------------------------------------------------------------
_limiter = Limiter(key_func=get_remote_address, default_limits=[])
app.state.limiter = _limiter
app.add_middleware(SlowAPIMiddleware)


async def _rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": {"error_code": "RATE_LIMITED", "message": f"Rate limit exceeded: {exc.detail}"}},
        headers={"Retry-After": "60"},
    )


app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

OUTPUT_DIR = os.environ.get("LYRA2_OUTPUT_DIR", "/app/outputs")
try:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except OSError as _e:
    # /app is read-only on host machines used for syntax checks; in Docker the
    # bind-mount/volume always lets us create this directory.  Fail loudly only
    # at job-submit time if the dir is still missing then.
    app_logger.warning(f"Could not create OUTPUT_DIR={OUTPUT_DIR}: {_e}")
app_logger.info(f"Output directory: {OUTPUT_DIR}")

# --- 全局模型變數 ---
# Lyra-2 has TWO heavy pipelines that may be loaded independently:
#   _video_pipeline : video diffusion (zoom-out / custom trajectory)
#   _gs_pipeline    : VIPE+DA3 GS reconstruction
_video_pipeline = None
_gs_pipeline = None

# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------
@dataclass
class JobRecord:
    job_id: str
    status: str          # "queued" | "processing" | "completed" | "failed"
    created_at: datetime.datetime
    result: Optional[Any] = None
    error: Optional[dict] = None
    output_dir: Optional[str] = None

_jobs: dict[str, JobRecord] = {}

# Jobs marked for cancellation (set before worker dequeues them)
_cancelled_jobs: set[str] = set()


# ---------------------------------------------------------------------------
# GPU request queue (single queue — Lyra-2 serialises everything on one GPU)
# ---------------------------------------------------------------------------
QUEUE_MAX_SIZE: int = int(os.environ.get("QUEUE_MAX_SIZE", "3"))

_infer_queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
_gpu_semaphore = asyncio.Semaphore(1)
_gpu_processing: bool = False


async def _gpu_worker_loop() -> None:
    """Single GPU worker: serialises all inference tasks through the Semaphore."""
    global _gpu_processing
    while True:
        job_id, sync_fn = await _infer_queue.get()

        # Skip jobs that were cancelled while waiting in queue
        if job_id in _cancelled_jobs:
            _cancelled_jobs.discard(job_id)
            _infer_queue.task_done()
            continue

        async with _gpu_semaphore:
            _gpu_processing = True
            _jobs[job_id].status = "processing"
            try:
                result = await run_in_threadpool(sync_fn)
                _jobs[job_id].status = "completed"
                _jobs[job_id].result = result
            except Exception as exc:
                error_msg = str(exc)
                # Treat torch.cuda.OutOfMemoryError (when torch is available) and any
                # exception whose message mentions "out of memory" as OOM-class events
                # that warrant flagging the container for autoheal restart.
                is_oom = False
                if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
                    is_oom = True
                elif "out of memory" in error_msg.lower():
                    is_oom = True

                if is_oom:
                    _set_force_restart(
                        f"CUDA OOM during inference (job={job_id}): {error_msg} — "
                        "possible memory fragmentation; autoheal will restart this container"
                    )
                    _jobs[job_id].status = "failed"
                    _jobs[job_id].error = {"error_code": "GPU_OOM", "message": error_msg}
                else:
                    _jobs[job_id].status = "failed"
                    _jobs[job_id].error = {"error_code": "INFERENCE_ERROR", "message": error_msg}
            finally:
                _gpu_processing = False
                _infer_queue.task_done()


async def _submit_to_gpu_worker(sync_fn: Any, request_id: str, req_dir: str) -> int:
    """Enqueue an inference job and return its 1-based queue position.

    Raises HTTPException(429) if the queue is full.
    Creates a JobRecord in _jobs with status='queued'.
    """
    if _infer_queue.full():
        raise HTTPException(
            status_code=429,
            detail={"error_code": "QUEUE_FULL", "message": "Server busy, please retry later"},
            headers={"Retry-After": "30"},
        )
    if not _check_vram_available(limit_gb=88.0):
        raise HTTPException(
            status_code=503,
            detail={"error_code": "VRAM_LIMIT_EXCEEDED", "message": "GPU memory limit reached, please retry later"},
            headers={"Retry-After": "30"},
        )
    _jobs[request_id] = JobRecord(
        job_id=request_id,
        status="queued",
        created_at=datetime.datetime.now(),
        output_dir=req_dir,
    )
    await _infer_queue.put((request_id, sync_fn))
    return _infer_queue.qsize()


async def _cleanup_loop() -> None:
    """Periodically delete completed/failed jobs older than 90 minutes."""
    while True:
        await asyncio.sleep(30 * 60)
        cutoff = datetime.datetime.now() - datetime.timedelta(minutes=90)
        to_delete = [
            jid for jid, job in list(_jobs.items())
            if job.status in ("completed", "failed") and job.created_at < cutoff
        ]
        for jid in to_delete:
            job = _jobs.pop(jid, None)
            if job and job.output_dir:
                shutil.rmtree(job.output_dir, ignore_errors=True)
        if to_delete:
            app_logger.info(f"[TTL Cleanup] Removed {len(to_delete)} expired job(s)")


# =============================================================================
# GPU 記憶體追蹤
# =============================================================================

# Physical GPU index for pynvml queries.
# pynvml ignores CUDA_VISIBLE_DEVICES and always addresses physical GPUs directly.
# Set NVML_GPU_INDEX to match the device_ids value in docker-compose.yml.
_NVML_GPU_INDEX = int(os.environ.get("NVML_GPU_INDEX", "0"))


def log_gpu_memory(label: str):
    if torch is None or not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    rsvd  = torch.cuda.memory_reserved()  / 1024**3
    driver = -1.0
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(_NVML_GPU_INDEX)
            info   = pynvml.nvmlDeviceGetMemoryInfo(handle)
            driver = info.used / 1024**3
        except Exception:
            driver = -1.0
    app_logger.info(
        f"GPU memory [{label}]: "
        f"torch_alloc={alloc:.2f} GB  "
        f"torch_rsvd={rsvd:.2f} GB  "
        f"driver_used={driver:.2f} GB"
    )


def flush_gpu() -> None:
    """Generic CUDA cache flush: empty_cache() + gc.collect().

    Lyra-2's two pipelines don't expose a custom cleanup hook, so we keep this
    body intentionally simple.  empty_cache() returns the PyTorch allocator's
    cached blocks back to the driver; gc.collect() drops Python references that
    might still be holding GPU tensors alive.
    """
    gc.collect()
    gc.collect()
    if torch is None or not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    try:
        torch._C._cuda_clearCublasWorkspaces()
    except Exception:
        pass
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()  # second pass after cuBLAS blocks are returned
    log_gpu_memory("after flush")


def _check_vram_available(limit_gb: float) -> bool:
    """Return True if used VRAM on the physical GPU is below limit_gb. Fail-open on errors."""
    if pynvml is None:
        return True
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(_NVML_GPU_INDEX)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_gb = info.used / 1024 ** 3
        return used_gb < limit_gb
    except Exception:
        return True  # nvml unavailable — let the request through


def _get_vram_used_gb() -> Optional[float]:
    """Return driver-reported TOTAL VRAM usage on the physical GPU (all processes).

    Used for monitoring/observability only — do NOT use for unhealthy detection on
    shared GPUs, because it includes other containers' allocations.
    """
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(_NVML_GPU_INDEX)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**3
    except Exception:
        return None


def _get_torch_vram_gb() -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (allocated_gb, reserved_gb, fragmentation_ratio) for THIS process.

    All values are per-process (PyTorch's view of this container's GPU allocations).
    fragmentation_ratio = (reserved - allocated) / reserved.  A high value while
    idle means the allocator has cached-but-fragmented blocks that may prevent large
    contiguous allocations — the classic sign of GPU memory fragmentation.
    """
    if torch is None or not torch.cuda.is_available():
        return None, None, None
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved  = torch.cuda.memory_reserved()  / 1024**3
    frag = (reserved - allocated) / reserved if reserved > 0 else 0.0
    return allocated, reserved, frag


# Per-process VRAM ceiling for leak detection.
# Uses torch.cuda.memory_reserved() (this container only), so it is safe for
# shared-GPU deployments — NOT affected by other containers' allocations.
# Tune for Lyra-2: video pipeline + GS pipeline together can sit ≈ 50–70 GB on
# Blackwell, so the threshold is set well above that.
TORCH_VRAM_LEAK_LIMIT_GB = float(os.environ.get("LYRA2_TORCH_VRAM_LEAK_LIMIT_GB", "80.0"))

# Force-restart flag: set by the GPU worker on CUDA OOM (fragmentation or true OOM).
# /health returns 503 whenever this is True, which triggers autoheal restart.
_force_restart: bool = False
_force_restart_reason: str = ""


def _set_force_restart(reason: str) -> None:
    global _force_restart, _force_restart_reason
    _force_restart = True
    _force_restart_reason = reason
    app_logger.error(f"[ForceRestart] Flagged for autoheal restart: {reason}")


# =============================================================================
# Model Loading (sentinel markers — warm-load deferred to first inference)
# =============================================================================

def ensure_video_pipeline_loaded() -> None:
    """Mark the video pipeline as ready.

    The actual model load happens lazily inside run_zoomgs() / run_custom_traj() on
    first use. This sentinel sets a module flag so /health correctly reports
    model_loaded=True. Warm-load (separating load() from infer()) is a future
    optimisation that requires a deeper Lyra-2 refactor.
    """
    global _video_pipeline
    if _video_pipeline is None:
        app_logger.info("Video pipeline marker set (warm-load deferred to first inference call)")
        _video_pipeline = "ready"  # sentinel


def ensure_gs_pipeline_loaded() -> None:
    """Mark the GS reconstruction pipeline as ready.

    Same lazy-marker pattern as ensure_video_pipeline_loaded() — the heavy load
    happens inside run_gs_recon() on first call.
    """
    global _gs_pipeline
    if _gs_pipeline is None:
        app_logger.info("GS pipeline marker set (warm-load deferred to first inference call)")
        _gs_pipeline = "ready"  # sentinel


# =============================================================================
# Lifecycle
# =============================================================================

@app.on_event("startup")
async def _start_gpu_worker():
    asyncio.create_task(_gpu_worker_loop())
    asyncio.create_task(_cleanup_loop())
    app_logger.info(f"GPU worker started (queue maxsize={QUEUE_MAX_SIZE})")
    # Must run AFTER uvicorn's dictConfig to take effect (see _reconfigure_uvicorn_access_logger)
    _reconfigure_uvicorn_access_logger()


@app.on_event("shutdown")
async def cleanup():
    try:
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    except Exception:
        pass


# =============================================================================
# Health
# =============================================================================

@app.get("/health")
async def health_check():
    torch_alloc_gb, torch_rsvd_gb, frag_ratio = _get_torch_vram_gb()
    nvml_total_gb = _get_vram_used_gb()  # total GPU usage (all processes) — informational only

    # Per-process VRAM leak: torch-reserved exceeds expected ceiling while idle.
    # Using torch stats (not pynvml) makes this safe for shared-GPU deployments.
    torch_leaked = (
        torch_rsvd_gb is not None
        and torch_rsvd_gb > TORCH_VRAM_LEAK_LIMIT_GB
        and not _gpu_processing
    )
    body = {
        # Lyra-2 has TWO pipelines — report healthy if EITHER is loaded so the
        # endpoint reflects "any model in memory" rather than requiring both.
        "model_loaded": (_video_pipeline is not None) or (_gs_pipeline is not None),
        "gpu_busy": _gpu_processing,
        "queue_size": _infer_queue.qsize(),
        "queue_capacity": QUEUE_MAX_SIZE,
        "active_jobs": len(_jobs),
        # Per-process torch stats (this container only)
        "torch_allocated_gb": round(torch_alloc_gb, 2) if torch_alloc_gb is not None else None,
        "torch_reserved_gb":  round(torch_rsvd_gb,  2) if torch_rsvd_gb  is not None else None,
        "torch_vram_leak_limit_gb": TORCH_VRAM_LEAK_LIMIT_GB,
        "fragmentation_ratio": round(frag_ratio, 3) if frag_ratio is not None else None,
        # Total GPU VRAM from driver (all processes on this GPU — informational)
        "nvml_total_gpu_vram_gb": round(nvml_total_gb, 2) if nvml_total_gb is not None else None,
    }

    # Priority 1: force restart (CUDA OOM / fragmentation triggered during inference)
    if _force_restart:
        app_logger.error(f"[Health] Returning unhealthy (force_restart): {_force_restart_reason}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "reason": "force_restart_flagged",
                "force_restart_reason": _force_restart_reason,
                **body,
            },
        )

    # Priority 2: per-process VRAM leak detected while idle
    if torch_leaked:
        app_logger.warning(
            f"[Health] VRAM leak: torch_reserved={torch_rsvd_gb:.2f} GB "
            f"> {TORCH_VRAM_LEAK_LIMIT_GB} GB threshold"
        )
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "reason": "torch_vram_leak_detected", **body},
        )

    return {"status": "ok", **body}


# =============================================================================
# Job submission endpoints
# =============================================================================

@app.post("/jobs/image-to-video")
async def image_to_video(
    mode: str = Form(...),
    image: UploadFile = File(...),
    caption: Optional[str] = Form(None),
    trajectory: Optional[UploadFile] = File(None),
    captions_json: Optional[UploadFile] = File(None),
    num_frames_zoom_in: int = Form(81),
    num_frames_zoom_out: int = Form(241),
    num_frames: int = Form(481),
    zoom_in_strength: float = Form(0.5),
    zoom_out_strength: float = Form(1.5),
    pose_scale: float = Form(1.0),
    use_dmd: bool = Form(False),
    seed: int = Form(1),
):
    """Generate a video from an image using Lyra-2.

    Two modes:
      - "preset": use built-in zoom-in / zoom-out trajectory + caption text
      - "custom": user-supplied trajectory.npz + captions.json
    """
    if mode not in ("preset", "custom"):
        raise HTTPException(
            status_code=400,
            detail={"error_code": "BAD_MODE", "message": "mode must be 'preset' or 'custom'"},
        )

    request_id = str(uuid.uuid4())
    req_dir = os.path.join(OUTPUT_DIR, request_id)
    os.makedirs(req_dir, exist_ok=True)

    image_path = os.path.join(req_dir, "input.png")
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    if mode == "preset":
        if ZoomGSParams is None or run_zoomgs is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error_code": "PIPELINE_UNAVAILABLE",
                    "message": "Lyra-2 inference modules not loaded — check container env",
                },
            )
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
        if CustomTrajParams is None or run_custom_traj is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error_code": "PIPELINE_UNAVAILABLE",
                    "message": "Lyra-2 inference modules not loaded — check container env",
                },
            )
        if trajectory is None or captions_json is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "MISSING_FIELDS",
                    "message": "mode=custom requires trajectory and captions_json files",
                },
            )
        traj_path = os.path.join(req_dir, "trajectory.npz")
        cap_path = os.path.join(req_dir, "captions.json")
        with open(traj_path, "wb") as f:
            shutil.copyfileobj(trajectory.file, f)
        with open(cap_path, "wb") as f:
            shutil.copyfileobj(captions_json.file, f)
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


@app.post("/jobs/video-to-gs")
async def video_to_gs(
    video: UploadFile = File(...),
):
    """Reconstruct a 3DGS scene from an input video using VIPE + DA3."""
    if GSReconParams is None or run_gs_recon is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error_code": "PIPELINE_UNAVAILABLE",
                "message": "Lyra-2 inference modules not loaded — check container env",
            },
        )

    request_id = str(uuid.uuid4())
    req_dir = os.path.join(OUTPUT_DIR, request_id)
    os.makedirs(req_dir, exist_ok=True)

    video_path = os.path.join(req_dir, "input.mp4")
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    params = GSReconParams(input_video_path=video_path, output_dir=req_dir)

    def sync_fn():
        ensure_gs_pipeline_loaded()
        return run_gs_recon(params)

    queue_pos = await _submit_to_gpu_worker(sync_fn, request_id, req_dir)
    return {"job_id": request_id, "status": "queued", "queue_position": queue_pos}


@app.post("/jobs/image-to-gs")
async def image_to_gs(
    mode: str = Form(...),
    image: UploadFile = File(...),
    caption: Optional[str] = Form(None),
    trajectory: Optional[UploadFile] = File(None),
    captions_json: Optional[UploadFile] = File(None),
    num_frames_zoom_in: int = Form(81),
    num_frames_zoom_out: int = Form(241),
    num_frames: int = Form(481),
    zoom_in_strength: float = Form(0.5),
    zoom_out_strength: float = Form(1.5),
    pose_scale: float = Form(1.0),
    use_dmd: bool = Form(False),
    seed: int = Form(1),
):
    """End-to-end: generate a video from an image, then reconstruct a 3DGS scene.

    Chains run_zoomgs() / run_custom_traj() (Step 1) into run_gs_recon() (Step 2)
    inside a single sync closure so they run back-to-back on the GPU worker.
    """
    if mode not in ("preset", "custom"):
        raise HTTPException(
            status_code=400,
            detail={"error_code": "BAD_MODE", "message": "mode must be 'preset' or 'custom'"},
        )

    request_id = str(uuid.uuid4())
    req_dir = os.path.join(OUTPUT_DIR, request_id)
    os.makedirs(req_dir, exist_ok=True)

    image_path = os.path.join(req_dir, "input.png")
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    params_zoom = None
    params_custom = None

    if mode == "preset":
        if ZoomGSParams is None or run_zoomgs is None or GSReconParams is None or run_gs_recon is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error_code": "PIPELINE_UNAVAILABLE",
                    "message": "Lyra-2 inference modules not loaded — check container env",
                },
            )
        params_zoom = ZoomGSParams(
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
    else:  # mode == "custom"
        if CustomTrajParams is None or run_custom_traj is None or GSReconParams is None or run_gs_recon is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "error_code": "PIPELINE_UNAVAILABLE",
                    "message": "Lyra-2 inference modules not loaded — check container env",
                },
            )
        if trajectory is None or captions_json is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "MISSING_FIELDS",
                    "message": "mode=custom requires trajectory and captions_json files",
                },
            )
        traj_path = os.path.join(req_dir, "trajectory.npz")
        cap_path = os.path.join(req_dir, "captions.json")
        with open(traj_path, "wb") as f:
            shutil.copyfileobj(trajectory.file, f)
        with open(cap_path, "wb") as f:
            shutil.copyfileobj(captions_json.file, f)
        params_custom = CustomTrajParams(
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
        if mode == "preset":
            step1 = run_zoomgs(params_zoom)
        else:
            step1 = run_custom_traj(params_custom)
        ensure_gs_pipeline_loaded()
        step2_params = GSReconParams(
            input_video_path=step1["video_path"],
            output_dir=req_dir,
        )
        step2 = run_gs_recon(step2_params)
        return {**step1, **step2}

    queue_pos = await _submit_to_gpu_worker(sync_fn, request_id, req_dir)
    return {"job_id": request_id, "status": "queued", "queue_position": queue_pos}
