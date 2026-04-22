"""Unit tests for job submission — GPU calls are mocked.
Run: pytest tests/test_api_submission.py -v
"""
import io
import sys
import os
import tempfile
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

# Point the API at a writable temp dir BEFORE importing lyra2_api — on macOS the
# default /app/outputs is unwritable.
os.environ.setdefault(
    "LYRA2_OUTPUT_DIR", os.path.join(tempfile.gettempdir(), "lyra2_test_outputs")
)
os.environ.setdefault(
    "LYRA2_LOG_DIR", os.path.join(tempfile.gettempdir(), "lyra2_test_logs")
)

# Make the repo root importable so `import lyra2_api` works
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
# And the Lyra-2 dir so `from lyra_2._src...` works on host machines (no Docker)
sys.path.insert(0, os.path.join(_REPO_ROOT, "Lyra-2"))


@pytest.fixture
def client():
    with patch("lyra2_api.run_zoomgs", return_value={"video_path": "/tmp/x.mp4", "output_dir": "/tmp"}), \
         patch("lyra2_api.run_custom_traj", return_value={"video_path": "/tmp/y.mp4", "output_dir": "/tmp"}), \
         patch("lyra2_api.run_gs_recon", return_value={"ply_path": "/tmp/z.ply", "output_dir": "/tmp"}), \
         patch("lyra2_api.ensure_video_pipeline_loaded"), \
         patch("lyra2_api.ensure_gs_pipeline_loaded"), \
         patch("lyra2_api._check_vram_available", return_value=True):
        # Also patch the dataclasses so the None-check guards in handlers don't trip
        # (they're None on host machines without torch). Set them to real classes.
        # The actual classes are importable — even on host — because the dataclass
        # module doesn't import torch. But the lyra2_api try/except may have set
        # them to None. Re-import directly for the test:
        from lyra_2._src.inference._runner_types import ZoomGSParams, CustomTrajParams, GSReconParams
        with patch("lyra2_api.ZoomGSParams", ZoomGSParams), \
             patch("lyra2_api.CustomTrajParams", CustomTrajParams), \
             patch("lyra2_api.GSReconParams", GSReconParams):
            import lyra2_api
            with TestClient(lyra2_api.app) as c:
                yield c


def test_image_to_video_preset(client):
    png = io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    r = client.post(
        "/jobs/image-to-video",
        data={"mode": "preset", "caption": "test"},
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
