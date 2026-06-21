"""
Drift evaluation.

Generates the "before" picture for the project: a flight path tracked
using only raw frame-to-frame odometry (no absolute correction), showing
how error compounds over time. This baseline is what Phase 3's UKF fusion
will be benchmarked against.

Produces two charts in assets/:
  - drift_path_overlay.png: ground truth vs. raw-odometry-only path,
    overlaid on the world map
  - drift_error_over_time.png: position error vs. frame index
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

from visloc.world import generate_world
from visloc.simulator import FrameSimulator, make_path
from visloc.odometry import RelativeOdometry

# Standard scenario for the project: a small amount of yaw (the camera
# isn't perfectly gimbal-stabilized) is the realistic drift source for
# translation-only LK, matching the real system's documented limitation -
# see odometry.py and the test suite for how this was derived empirically.
WORLD_SEED = 42
N_FRAMES = 200
CROP_SIZE = 200
NOISE_STD = 1.5
MAX_YAW_DEG = 2.0
PATH_KIND = "loop"


def run_drift_baseline():
    world = generate_world(width=1800, height=1800, seed=WORLD_SEED, n_blobs=200, n_roads=14)
    path = make_path(PATH_KIND, n_frames=N_FRAMES, world_w=world.shape[1], world_h=world.shape[0])
    sim = FrameSimulator(world, path, crop_size=CROP_SIZE, noise_std=NOISE_STD, max_yaw_deg=MAX_YAW_DEG, seed=7)
    frames = sim.generate()

    odo = RelativeOdometry()
    est_path = odo.track_path(frames)
    gt_path = np.array([[f.gt_x, f.gt_y] for f in frames])
    errors = np.hypot(est_path[:, 0] - gt_path[:, 0], est_path[:, 1] - gt_path[:, 1])

    return world, gt_path, est_path, errors


def plot_path_overlay(world, gt_path, est_path, out_path):
    world_rgb = cv2.cvtColor(world, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(world_rgb)
    ax.plot(gt_path[:, 0], gt_path[:, 1], color="white", linewidth=2.5, label="Ground truth (actual flight)")
    ax.plot(est_path[:, 0], est_path[:, 1], color="#e35454", linewidth=2, linestyle="--", label="Raw odometry only (drifts)")
    ax.scatter(*gt_path[0], color="lime", s=60, zorder=5, label="Start")
    ax.set_title("VisLoc — Phase 2: Raw Odometry Drift vs. Ground Truth")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_path_overlay_zoom(world, gt_path, est_path, out_path, tail_frac=0.35, pad=40):
    """Zoomed view of the last portion of the flight, where drift is
    largest - the full-world view makes 10-20px of drift invisible
    against an 1800px map, which defeats the point of the chart."""
    n = len(gt_path)
    start = int(n * (1 - tail_frac))
    gt_tail = gt_path[start:]
    est_tail = est_path[start:]

    x_min = int(min(gt_tail[:, 0].min(), est_tail[:, 0].min()) - pad)
    x_max = int(max(gt_tail[:, 0].max(), est_tail[:, 0].max()) + pad)
    y_min = int(min(gt_tail[:, 1].min(), est_tail[:, 1].min()) - pad)
    y_max = int(max(gt_tail[:, 1].max(), est_tail[:, 1].max()) + pad)

    world_rgb = cv2.cvtColor(world, cv2.COLOR_BGR2RGB)
    crop = world_rgb[max(y_min, 0):y_max, max(x_min, 0):x_max]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(crop, extent=[x_min, x_max, y_max, y_min])
    ax.plot(gt_tail[:, 0], gt_tail[:, 1], color="white", linewidth=3, label="Ground truth")
    ax.plot(est_tail[:, 0], est_tail[:, 1], color="#e35454", linewidth=2.5, linestyle="--", label="Raw odometry (drifted)")
    ax.scatter(*gt_tail[-1], color="white", s=70, marker="o", zorder=5, edgecolors="black")
    ax.scatter(*est_tail[-1], color="#e35454", s=70, marker="X", zorder=5, edgecolors="black")
    ax.set_title(f"Zoomed: last {int(tail_frac*100)}% of flight — divergence visible")
    ax.legend(loc="best", framealpha=0.9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_error_over_time(errors, out_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(errors, color="#e35454", linewidth=2)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Position error (px)")
    ax.set_title("Raw Odometry Drift Over Time (no correction)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    world, gt_path, est_path, errors = run_drift_baseline()

    plot_path_overlay(world, gt_path, est_path, "assets/drift_path_overlay.png")
    plot_path_overlay_zoom(world, gt_path, est_path, "assets/drift_path_zoom.png")
    plot_error_over_time(errors, "assets/drift_error_over_time.png")

    print(f"Frames: {N_FRAMES}, path: {PATH_KIND}, yaw: ±{MAX_YAW_DEG}deg, noise std: {NOISE_STD}px")
    print(f"Final drift (last-frame error): {errors[-1]:.1f}px")
    print(f"Mean error: {errors.mean():.1f}px")
    print(f"Max error: {errors.max():.1f}px")
