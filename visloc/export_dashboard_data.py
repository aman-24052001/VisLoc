"""
Dashboard data export.

Runs the canonical scenario (same as evaluate_fusion.py) and exports
everything the static dashboard needs to play back without a backend:
  - a resized world map image
  - a sprite sheet of all per-frame camera crops (one image, not N requests)
  - a JSON file with all path/error arrays + sprite sheet layout metadata
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import cv2

from visloc.world import generate_world
from visloc.simulator import FrameSimulator, make_path
from visloc.fusion import run_fusion
from visloc.localizer import AbsoluteLocalizer

WORLD_SEED = 42
N_FRAMES = 200
CROP_SIZE = 200
NOISE_STD = 1.5
MAX_YAW_DEG = 2.0
PATH_KIND = "loop"
SCENARIO_SEED = 7
VPS_RATE = 10

WORLD_DISPLAY_MAX = 1000   # max dimension of the exported world image
CROP_THUMB_SIZE = 110      # each sprite cell, downsized from CROP_SIZE
SPRITE_COLS = 20           # 200 frames / 20 cols = 10 rows

OUT_DIR = "docs/assets"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    world = generate_world(width=1800, height=1800, seed=WORLD_SEED, n_blobs=200, n_roads=14)
    path = make_path(PATH_KIND, n_frames=N_FRAMES, world_w=world.shape[1], world_h=world.shape[0])
    sim = FrameSimulator(world, path, crop_size=CROP_SIZE, noise_std=NOISE_STD,
                          max_yaw_deg=MAX_YAW_DEG, seed=SCENARIO_SEED)
    frames = sim.generate()

    # Re-run localization separately here too, just to capture inlier
    # confidence per fix for the dashboard's VPS readout (run_fusion
    # doesn't expose this, it only returns the frame indices/fix success).
    loc = AbsoluteLocalizer(world, detector="orb")
    vps_confidence = {}
    for i, f in enumerate(frames):
        if i % VPS_RATE == 0:
            res = loc.localize(f.image)
            if res.success:
                vps_confidence[i] = res.n_inliers

    result = run_fusion(frames, world, vps_rate=VPS_RATE)
    fused, vio, gt = result["fused_path"], result["vio_global_path"], result["gt_path"]
    err_fused = np.hypot(fused[:, 0] - gt[:, 0], fused[:, 1] - gt[:, 1])
    err_vio = np.hypot(vio[:, 0] - gt[:, 0], vio[:, 1] - gt[:, 1])

    # --- World image (resized for web) ---
    orig_h, orig_w = world.shape[:2]
    scale = WORLD_DISPLAY_MAX / max(orig_w, orig_h)
    disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
    world_disp = cv2.resize(world, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{OUT_DIR}/world.jpg", world_disp, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # --- Sprite sheet of all camera crops ---
    n_rows = math.ceil(len(frames) / SPRITE_COLS)
    sprite = np.zeros((n_rows * CROP_THUMB_SIZE, SPRITE_COLS * CROP_THUMB_SIZE, 3), dtype=np.uint8)
    for i, f in enumerate(frames):
        thumb = cv2.resize(f.image, (CROP_THUMB_SIZE, CROP_THUMB_SIZE), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, SPRITE_COLS)
        sprite[r * CROP_THUMB_SIZE:(r + 1) * CROP_THUMB_SIZE, c * CROP_THUMB_SIZE:(c + 1) * CROP_THUMB_SIZE] = thumb
    cv2.imwrite(f"{OUT_DIR}/crops_sprite.jpg", sprite, [cv2.IMWRITE_JPEG_QUALITY, 80])

    # --- Data JSON (all coordinates in DISPLAY scale, pre-scaled here so
    #     the frontend never needs to know the original world size) ---
    def scaled(arr):
        return (np.asarray(arr) * scale).round(2).tolist()

    data = {
        "meta": {
            "n_frames": len(frames),
            "scenario": {
                "path_kind": PATH_KIND, "seed": SCENARIO_SEED, "vps_rate": VPS_RATE,
                "noise_std": NOISE_STD, "max_yaw_deg": MAX_YAW_DEG,
            },
            "world_display": {"w": disp_w, "h": disp_h},
            "sprite": {"cols": SPRITE_COLS, "rows": n_rows, "cell": CROP_THUMB_SIZE},
        },
        "gt_path": scaled(gt),
        "vio_path": scaled(vio),
        "fused_path": scaled(fused),
        "vps_fix_frames": result["vps_fix_frames"],
        "vps_confidence": vps_confidence,
        "err_vio": np.round(err_vio, 2).tolist(),
        "err_fused": np.round(err_fused, 2).tolist(),
    }

    with open(f"{OUT_DIR}/data.json", "w") as fp:
        json.dump(data, fp)

    print(f"World image: {disp_w}x{disp_h} -> {OUT_DIR}/world.jpg")
    print(f"Sprite sheet: {SPRITE_COLS}x{n_rows} cells of {CROP_THUMB_SIZE}px -> {OUT_DIR}/crops_sprite.jpg")
    print(f"Data JSON -> {OUT_DIR}/data.json "
          f"({os.path.getsize(f'{OUT_DIR}/data.json')/1024:.1f} KB)")
    print(f"VPS fixes: {len(result['vps_fix_frames'])}, "
          f"final drift reduction: {100*(1-err_fused[-1]/err_vio[-1]):.1f}%")


if __name__ == "__main__":
    main()
