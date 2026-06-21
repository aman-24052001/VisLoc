"""
Fusion evaluation.

Generates the "after" picture for the project: runs the full VIO+VPS+UKF
pipeline on the same standard scenario used for Phase 2's drift baseline,
and produces the headline result - % drift reduction from fusion vs. raw
odometry alone.

Produces in assets/:
  - fusion_path_overlay.png: ground truth vs. raw-odometry vs. fused path
  - fusion_path_zoom.png: zoomed view of the same divergence point used in
    Phase 2's drift chart, now showing the fused path staying close
  - fusion_error_over_time.png: error vs. frame index, raw vs. fused
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from visloc.world import generate_world
from visloc.simulator import FrameSimulator, make_path
from visloc.fusion import run_fusion

# Same standard scenario as evaluate_drift.py, so the two are directly
# comparable.
WORLD_SEED = 42
N_FRAMES = 200
CROP_SIZE = 200
NOISE_STD = 1.5
MAX_YAW_DEG = 2.0
PATH_KIND = "loop"
SCENARIO_SEED = 7
VPS_RATE = 10


def run_fusion_scenario():
    world = generate_world(width=1800, height=1800, seed=WORLD_SEED, n_blobs=200, n_roads=14)
    path = make_path(PATH_KIND, n_frames=N_FRAMES, world_w=world.shape[1], world_h=world.shape[0])
    sim = FrameSimulator(world, path, crop_size=CROP_SIZE, noise_std=NOISE_STD,
                          max_yaw_deg=MAX_YAW_DEG, seed=SCENARIO_SEED)
    frames = sim.generate()
    result = run_fusion(frames, world, vps_rate=VPS_RATE)
    return world, result


def plot_three_way_zoom(world, gt, vio, fused, out_path, tail_frac=0.35, pad=40):
    n = len(gt)
    start = int(n * (1 - tail_frac))
    gt_t, vio_t, fused_t = gt[start:], vio[start:], fused[start:]
    valid = ~np.isnan(fused_t[:, 0])

    all_x = np.concatenate([gt_t[:, 0], vio_t[:, 0], fused_t[valid, 0]])
    all_y = np.concatenate([gt_t[:, 1], vio_t[:, 1], fused_t[valid, 1]])
    x_min, x_max = int(all_x.min() - pad), int(all_x.max() + pad)
    y_min, y_max = int(all_y.min() - pad), int(all_y.max() + pad)

    world_rgb = cv2.cvtColor(world, cv2.COLOR_BGR2RGB)
    crop = world_rgb[max(y_min, 0):y_max, max(x_min, 0):x_max]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(crop, extent=[x_min, x_max, y_max, y_min])
    ax.plot(gt_t[:, 0], gt_t[:, 1], color="white", linewidth=3, label="Ground truth")
    ax.plot(vio_t[:, 0], vio_t[:, 1], color="#e35454", linewidth=2, linestyle="--", label="Raw odometry (drifts)")
    ax.plot(fused_t[valid, 0], fused_t[valid, 1], color="#fbbf24", linewidth=2.5, linestyle="-.", label="UKF fused (corrected)")
    ax.set_title(f"Zoomed: last {int(tail_frac*100)}% of flight - fusion stays close to truth")
    ax.legend(loc="best", framealpha=0.9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_error_comparison(err_vio, err_fused, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(err_vio, color="#e35454", linewidth=2, label="Raw odometry only")
    ax.plot(err_fused, color="#fbbf24", linewidth=2, label="UKF fused")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Position error (px)")
    ax.set_title("Drift: Raw Odometry vs. UKF Fusion")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    world, result = run_fusion_scenario()
    fused, vio, gt = result["fused_path"], result["vio_global_path"], result["gt_path"]
    valid = ~np.isnan(fused[:, 0])

    err_fused = np.hypot(fused[:, 0] - gt[:, 0], fused[:, 1] - gt[:, 1])
    err_vio = np.hypot(vio[:, 0] - gt[:, 0], vio[:, 1] - gt[:, 1])

    plot_three_way_zoom(world, gt, vio, fused, "assets/fusion_path_zoom.png")
    plot_error_comparison(err_vio, err_fused, "assets/fusion_error_over_time.png")

    print(f"Scenario: {PATH_KIND}, seed={SCENARIO_SEED}, {N_FRAMES} frames, "
          f"yaw=±{MAX_YAW_DEG}deg, noise std={NOISE_STD}px, vps_rate={VPS_RATE}")
    print()
    print(f"{'Metric':<20}{'Raw odometry':>15}{'UKF fused':>15}{'Reduction':>12}")
    for label, v_val, f_val in [
        ("Final drift", err_vio[-1], err_fused[-1]),
        ("Mean error", err_vio[valid].mean(), err_fused[valid].mean()),
        ("Max error", err_vio[valid].max(), err_fused[valid].max()),
    ]:
        reduction = 100 * (1 - f_val / v_val)
        print(f"{label:<20}{v_val:>14.2f}{f_val:>15.2f}{reduction:>11.1f}%")
