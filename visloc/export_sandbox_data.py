"""
Sandbox data export.

Unlike the main dashboard (Phase 4), the parameter sandbox needs to run
the UKF fusion *live* in the browser as users move sliders - re-running
the CV pipeline (ORB matching, optical flow) client-side isn't feasible
without porting OpenCV to JS, but the fusion math itself is small 4x4
matrix algebra and ports cleanly.

So: precompute a few noise-level presets (the one piece that genuinely
needs Python/OpenCV), each with a *dense* set of VPS fix candidates
(every 5 frames) so the in-browser "VPS rate" slider can pick any
multiple of 5 by subsampling, without needing to re-run the localizer.
The actual filter tuning (process noise, soft-correction frames, gate
threshold, VPS rate) then runs for real in JS - see docs/index.html.
"""
from __future__ import annotations

import json

import numpy as np

from visloc.world import generate_world
from visloc.simulator import FrameSimulator, make_path
from visloc.odometry import RelativeOdometry
from visloc.localizer import AbsoluteLocalizer

WORLD_SEED = 42
N_FRAMES = 200
CROP_SIZE = 200
PATH_KIND = "loop"
SCENARIO_SEED = 7
CANDIDATE_RATE = 5  # dense fix attempts every 5 frames

PRESETS = {
    "calm":      {"noise_std": 0.5, "max_yaw_deg": 0.5},
    "standard":  {"noise_std": 1.5, "max_yaw_deg": 2.0},   # matches Phases 1-4
    "turbulent": {"noise_std": 3.0, "max_yaw_deg": 5.0},
}

OUT_PATH = "docs/assets/sandbox_data.json"
WORLD_DISPLAY_MAX = 1000


def export_preset(world, noise_std, max_yaw_deg):
    path = make_path(PATH_KIND, n_frames=N_FRAMES, world_w=world.shape[1], world_h=world.shape[0])
    sim = FrameSimulator(world, path, crop_size=CROP_SIZE, noise_std=noise_std,
                          max_yaw_deg=max_yaw_deg, seed=SCENARIO_SEED)
    frames = sim.generate()

    odo = RelativeOdometry()
    loc = AbsoluteLocalizer(world, detector="orb")

    vio_local = odo.track_path(frames, start_xy=(0.0, 0.0))
    gt = np.array([[f.gt_x, f.gt_y] for f in frames])

    # Deliberately NOT pre-scaled for display here - the fusion math
    # (run live in JS) must operate in the same raw/unscaled coordinate
    # frame as the noise parameters (vps_noise_std, process_noise_std,
    # etc., all defined in raw pixels), exactly mirroring the Python
    # implementation. An earlier version pre-scaled this data but left
    # the noise parameters unscaled, which silently distorted the
    # Mahalanobis gate's distance calculation by scale^2 (innovation
    # scaled, covariance didn't) - it only showed up as a real numerical
    # divergence under a strict gate threshold, since looser thresholds
    # never hit the now-wrong boundary. The display scale is applied
    # only at render time in the frontend instead.
    vps_candidates = []
    for i, f in enumerate(frames):
        if i % CANDIDATE_RATE == 0:
            res = loc.localize(f.image)
            if res.success:
                vps_candidates.append({"frame": i, "x": round(res.x, 2),
                                        "y": round(res.y, 2), "inliers": res.n_inliers})

    return {
        "gt_path": gt.round(2).tolist(),
        "vio_local_path": vio_local.round(2).tolist(),
        "vps_candidates": vps_candidates,
    }


def main():
    world = generate_world(width=1800, height=1800, seed=WORLD_SEED, n_blobs=200, n_roads=14)
    scale = WORLD_DISPLAY_MAX / max(world.shape[1], world.shape[0])

    data = {
        "meta": {
            "n_frames": N_FRAMES,
            "candidate_rate": CANDIDATE_RATE,
            "scale": round(scale, 6),  # for the frontend to apply at render time only
        },
        "presets": {},
    }
    for name, params in PRESETS.items():
        print(f"Generating preset '{name}' ({params})...")
        data["presets"][name] = export_preset(world, **params)
        n_fixes = len(data["presets"][name]["vps_candidates"])
        print(f"  -> {n_fixes} VPS fix candidates")

    with open(OUT_PATH, "w") as fp:
        json.dump(data, fp)

    import os
    print(f"\nWrote {OUT_PATH} ({os.path.getsize(OUT_PATH)/1024:.1f} KB)")


if __name__ == "__main__":
    main()
