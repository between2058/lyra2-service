# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RECON_DA3_MODEL_PATH = str(_REPO_ROOT / "checkpoints" / "recon" / "model.pt")


@dataclass
class ZoomGSParams:
    input_image_path: str
    num_samples: int = 10
    sample_start_idx: int = 0
    sample_id: Optional[int] = None
    prompt: str = ""
    prompt_dir: Optional[str] = None
    prompt_suffix: str = ""
    experiment: str = "lyra_framepack_spatial"
    checkpoint_dir: str = "checkpoints/model"
    output_path: str = "inference/lyra2_zoomgs"
    guidance: float = 5.0
    shift: float = 5.0
    num_sampling_step: int = 50
    seed: int = 1
    fps: int = 16
    num_frames: int = 161
    num_frames_zoom_in: int = 81
    num_frames_zoom_out: int = 241
    resolution: str = "480,832"
    context_parallel_size: int = 1
    lora_paths: List[str] = field(default_factory=lambda: [
        "checkpoints/lora/realism_boost.safetensors",
        "checkpoints/lora/detail_enhancer.safetensors",
    ])
    lora_weights: List[float] = field(default_factory=lambda: [0.4, 0.4])
    offload: bool = False
    offload_when_prompt: bool = False
    zoom_in_trajectory: str = "horizontal_zoom"
    zoom_out_trajectory: str = "horizontal_zoom"
    zoom_in_direction: str = "right"
    zoom_out_direction: str = "left"
    zoom_in_strength: float = 0.5
    zoom_out_strength: float = 1.5
    use_moge_scale: bool = True
    ground_plane_align: bool = False
    ground_plane_bottom_frac: float = 0.4
    zoom_out_upward_shift: float = 0.05
    zoom_out_upward_ratio: float = 0.15
    depth_backend: str = "da3"
    da3_model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    da3_model_path_custom: str = "checkpoints/recon/model.pt"
    da3_frame_interval: int = 8
    da3_max_history_frames: int = 10
    da3_include_ar_chunk_last_frames: bool = False
    da3_use_predicted_pose: bool = False
    da3_predicted_pose_continuation: bool = False
    use_dmd: bool = False
    ablate_same_t5: bool = False
    use_dmd_scheduler: bool = False
    warp_chunk_size: Optional[int] = None
    num_retrieval_views: int = 1
    disable_cache_update: bool = False
    multiview_ids: Optional[List[int]] = None
    offload_da3_diffusion: bool = False


@dataclass
class CustomTrajParams:
    input_image_path: str
    trajectory_path: str
    num_samples: int = 10
    sample_start_idx: int = 0
    prompt: str = ""
    prompt_dir: Optional[str] = None
    captions_path: Optional[str] = None
    prompt_suffix: str = ""
    experiment: str = "lyra_framepack_spatial"
    checkpoint_dir: str = "checkpoints/model"
    output_path: str = "inference/lyra2_custom_traj"
    guidance: float = 5.0
    shift: float = 5.0
    num_sampling_step: int = 35
    seed: int = 1
    fps: int = 16
    num_frames: int = 161
    pose_scale: float = 1.1
    resolution: str = "480,832"
    context_parallel_size: int = 1
    lora_paths: Optional[List[str]] = None
    lora_weights: Optional[List[float]] = None
    offload: bool = False
    offload_when_prompt: bool = False
    debug: bool = False
    use_moge_scale: bool = True
    depth_backend: str = "da3"
    da3_model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    da3_model_path_custom: Optional[str] = None
    da3_frame_interval: int = 8
    da3_max_history_frames: int = 10
    da3_include_ar_chunk_last_frames: bool = False
    da3_use_predicted_pose: bool = False
    da3_predicted_pose_continuation: bool = False
    use_dmd: bool = False
    ablate_same_t5: bool = False
    use_dmd_scheduler: bool = False
    warp_chunk_size: Optional[int] = None
    num_retrieval_views: int = 1
    disable_cache_update: bool = False
    multiview_ids: Optional[List[int]] = None
    offload_da3_diffusion: bool = False


@dataclass
class GSReconParams:
    input_video_path: str
    output_dir: Optional[str] = None
    force: bool = False
    device: Optional[str] = None
    no_vipe: bool = False
    vipe_overrides: Optional[List[str]] = None
    vipe_full_mode: bool = False
    max_frames: int = 0
    da3_max_frames: int = 128
    da3_model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    da3_model_path_custom: str = _DEFAULT_RECON_DA3_MODEL_PATH
    da3_process_res: Optional[int] = None
    da3_process_method: str = "upper_bound_resize"
    max_resolution: int = 0
    gs_down_ratio: int = 2
    gs_scale_extra_multiplier: float = 1.0
    gs_ply_prune_opacity_percentile: Optional[float] = None
    gs_ds_feature_mode: bool = True
    use_da3_render_pose: bool = True
    render_fps: Optional[float] = None
    render_chunk_size: int = 1
