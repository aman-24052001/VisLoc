"""
Frame simulator.

Simulates a downward-facing camera moving along a synthetic flight path
over a static "world" image, producing a sequence of cropped frames plus
ground-truth (x, y, yaw) for each frame. This is the data source for both
the absolute localizer and the odometry tracker.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2


@dataclass
class Frame:
    """A single simulated camera frame plus its ground-truth pose."""
    index: int
    image: np.ndarray      # cropped BGR frame
    gt_x: float            # ground-truth centre x in world coordinates
    gt_y: float            # ground-truth centre y in world coordinates
    gt_yaw_deg: float      # ground-truth yaw in degrees


def make_path(
    kind: str,
    n_frames: int,
    world_w: int,
    world_h: int,
    margin: int = 250,
) -> np.ndarray:
    """Generate ground-truth (x, y) waypoints for a named path shape.

    Returns an array of shape (n_frames, 2).
    """
    t = np.linspace(0, 1, n_frames)
    cx, cy = world_w / 2, world_h / 2
    rx, ry = world_w / 2 - margin, world_h / 2 - margin

    if kind == "loop":
        theta = t * 2 * np.pi
        x = cx + rx * np.cos(theta)
        y = cy + ry * np.sin(theta)
    elif kind == "zigzag":
        x = margin + t * (world_w - 2 * margin)
        # Triangle wave across the y-range.
        period = 0.2
        phase = (t / period) % 1.0
        tri = np.where(phase < 0.5, phase * 2, 2 - phase * 2)
        y = margin + tri * (world_h - 2 * margin)
    elif kind == "straight":
        x = margin + t * (world_w - 2 * margin)
        y = np.full_like(t, cy)
    else:
        raise ValueError(f"Unknown path kind: {kind!r}")

    return np.stack([x, y], axis=1)


class FrameSimulator:
    """Generates a sequence of camera-like crops along a flight path."""

    def __init__(
        self,
        world: np.ndarray,
        path_xy: np.ndarray,
        crop_size: int = 220,
        noise_std: float = 4.0,
        max_yaw_deg: float = 0.0,
        seed: int = 0,
    ):
        self.world = world
        self.path_xy = path_xy
        self.crop_size = crop_size
        self.noise_std = noise_std
        self.max_yaw_deg = max_yaw_deg
        self.rng = np.random.default_rng(seed)
        self.world_h, self.world_w = world.shape[:2]

    def __len__(self) -> int:
        return len(self.path_xy)

    def _crop_at(self, cx: float, cy: float, yaw_deg: float) -> np.ndarray:
        """Extract a (possibly rotated) square crop centred at (cx, cy)."""
        size = self.crop_size
        half = size / 2.0

        if abs(yaw_deg) < 1e-6:
            x0, y0 = int(cx - half), int(cy - half)
            x0 = np.clip(x0, 0, self.world_w - size)
            y0 = np.clip(y0, 0, self.world_h - size)
            return self.world[y0:y0 + size, x0:x0 + size].copy()

        # Rotated crop: rotate a padded region then centre-crop, so the
        # output simulates a yawed camera rather than just shifting it.
        pad = int(size * 1.5)
        x0 = int(np.clip(cx - pad / 2, 0, self.world_w - pad))
        y0 = int(np.clip(cy - pad / 2, 0, self.world_h - pad))
        region = self.world[y0:y0 + pad, x0:x0 + pad]
        m = cv2.getRotationMatrix2D((pad / 2, pad / 2), yaw_deg, 1.0)
        rotated = cv2.warpAffine(region, m, (pad, pad))
        c0 = int(pad / 2 - half)
        return rotated[c0:c0 + size, c0:c0 + size].copy()

    def generate(self) -> list[Frame]:
        frames = []
        for i, (px, py) in enumerate(self.path_xy):
            nx = px + self.rng.normal(0, self.noise_std)
            ny = py + self.rng.normal(0, self.noise_std)
            yaw = self.rng.uniform(-self.max_yaw_deg, self.max_yaw_deg) if self.max_yaw_deg else 0.0

            crop = self._crop_at(nx, ny, yaw)
            # Ground truth = the actual (jittered) pose the crop was rendered
            # at, not the idealized waypoint (px, py). The waypoint is just
            # the design intent; what the camera really saw is centred on
            # (nx, ny). Using the waypoint as "ground truth" would silently
            # mismeasure frame-to-frame odometry error later, since odometry
            # estimates motion between the *actual* rendered frames.
            frames.append(Frame(index=i, image=crop, gt_x=nx, gt_y=ny, gt_yaw_deg=yaw))
        return frames


if __name__ == "__main__":
    from visloc.world import generate_world

    world = generate_world()
    path = make_path("loop", n_frames=300, world_w=world.shape[1], world_h=world.shape[0])
    sim = FrameSimulator(world, path, crop_size=220, noise_std=3.0, max_yaw_deg=8.0)
    frames = sim.generate()
    print(f"Generated {len(frames)} frames, first crop shape: {frames[0].image.shape}")
    cv2.imwrite("assets/sample_frame_000.png", frames[0].image)
    cv2.imwrite("assets/sample_frame_150.png", frames[150].image)
