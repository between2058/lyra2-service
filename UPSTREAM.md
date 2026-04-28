# Upstream tracking

Source: https://github.com/nv-tlabs/lyra
Path imported: `Lyra-2/`
Pinned commit: 52e5079
Date imported: 2026-04-22

To re-sync: re-run the copy command in this repo's
`docs/plans/2026-04-22-lyra2-fastapi-blackwell.md` Task 1.

Local modifications applied to upstream source:
- `Lyra-2/lyra_2/_src/inference/_runner_types.py` — added in Task 2
- `Lyra-2/lyra_2/_src/inference/lyra2_zoomgs_inference.py` — refactored to expose `run_zoomgs()` (Task 2)
- `Lyra-2/lyra_2/_src/inference/lyra2_custom_traj_inference.py` — refactored to expose `run_custom_traj()` (Task 2)
- `Lyra-2/lyra_2/_src/inference/vipe_da3_gs_recon.py` — refactored to expose `run_gs_recon()` (Task 2)

Vendored submodule contents (upstream lists these as git submodules; we flatten
them into this repo so a fresh `git clone` is buildable without
`--recurse-submodules`):

- `Lyra-2/lyra_2/_src/inference/depth_anything_3/`
    - source: https://github.com/frankshen07/Depth-Anything-3
    - pinned commit: `1ed6cb8eee386a3c94077d907b09c7aa1c312cd8`
- `Lyra-2/lyra_2/_src/inference/depth_anything_3/da3_streaming/loop_utils/salad/`
    - source: https://github.com/serizba/salad
    - pinned commit: `6aede13a3f6c25750bf7fde10209c06cb73060bb`
- `Lyra-2/lyra_2/_src/inference/vipe/`
    - source: https://github.com/nv-tlabs/vipe
    - pinned commit: `b7cac64616763bc755009db79a1815c7cdc9b130`

To re-sync these submodules: in the upstream lyra clone, run
`git submodule update --init --recursive` then re-copy the populated dirs over
the empty placeholders here.
