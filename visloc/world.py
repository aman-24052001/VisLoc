"""
Procedural "world" map generator.

Generates a synthetic aerial/satellite-style reference image with enough
texture and structure (blobs, roads, edges) for ORB/AKAZE feature matching
to work reliably. This stands in for a real satellite tile in v1 — swap in
an actual aerial photo later by pointing the simulator at a different file;
nothing downstream needs to change.

Deterministic given a seed, so demos and tests are reproducible.
"""
from __future__ import annotations

import numpy as np
import cv2


def generate_world(
    width: int = 2000,
    height: int = 2000,
    seed: int = 42,
    n_blobs: int = 220,
    n_roads: int = 14,
) -> np.ndarray:
    """Generate a synthetic aerial-style world map.

    Returns a BGR uint8 image (OpenCV convention) of shape (height, width, 3).
    """
    rng = np.random.default_rng(seed)

    # Base terrain: low-frequency noise field, colour-mapped to look
    # vaguely like mixed terrain (greens/browns) rather than flat noise.
    base = rng.normal(loc=0.5, scale=0.15, size=(height // 8, width // 8))
    base = cv2.resize(base, (width, height), interpolation=cv2.INTER_CUBIC)
    base = np.clip(base, 0, 1)

    terrain = np.zeros((height, width, 3), dtype=np.float32)
    terrain[..., 0] = 60 + base * 40   # B
    terrain[..., 1] = 90 + base * 70   # G
    terrain[..., 2] = 70 + base * 50   # R
    img = terrain.astype(np.uint8)

    # Structures: irregular blobs (buildings/fields) with varied size,
    # colour and rotation so local neighbourhoods are visually distinct
    # (this is what gives ORB/AKAZE something to lock onto).
    # Earth-tone palette (BGR) to mimic aerial/satellite imagery: dull
    # greens (fields), browns/tans (bare soil, rooftops), greys (concrete/
    # roads), muted blues (water). Each draw gets per-channel jitter so
    # blobs aren't flat/identical, which keeps them feature-rich for ORB.
    palette_bgr = [
        (40, 90, 60),     # field green
        (55, 110, 80),    # lighter field green
        (70, 95, 110),    # tan / bare soil
        (60, 80, 95),     # brown rooftop
        (90, 90, 90),     # concrete grey
        (120, 75, 55),    # muted blue (water/shadow)
        (45, 70, 75),     # olive
        (100, 100, 85),   # pale tan
    ]

    for _ in range(n_blobs):
        cx = float(rng.integers(0, width))
        cy = float(rng.integers(0, height))
        size_x = float(rng.integers(15, 90))
        size_y = float(rng.integers(15, 90))
        angle = float(rng.uniform(0, 180))
        base_color = np.array(palette_bgr[rng.integers(0, len(palette_bgr))], dtype=np.int32)
        jitter = rng.integers(-12, 12, size=3)
        color = tuple(int(c) for c in np.clip(base_color + jitter, 0, 255))

        rect = ((cx, cy), (size_x, size_y), angle)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillConvexPoly(img, box, color)
        # Thin border so blob edges create strong corner features.
        cv2.polylines(img, [box], isClosed=True, color=(25, 35, 30), thickness=1)

    # Roads: a handful of long straight/segmented lines crossing the map,
    # which create strong, repeatable linear features.
    for _ in range(n_roads):
        x1, y1 = rng.integers(0, width), rng.integers(0, height)
        x2, y2 = rng.integers(0, width), rng.integers(0, height)
        thickness = rng.integers(4, 9)
        cv2.line(img, (x1, y1), (x2, y2), (50, 50, 55), thickness)

    # Light texture noise on top so flat blob interiors aren't textureless.
    speckle = rng.normal(0, 6, size=(height, width, 3))
    img = np.clip(img.astype(np.float32) + speckle, 0, 255).astype(np.uint8)

    return img


def save_world(path: str, **kwargs) -> np.ndarray:
    img = generate_world(**kwargs)
    cv2.imwrite(path, img)
    return img


if __name__ == "__main__":
    world = save_world("assets/world.png")
    print(f"Generated world map: {world.shape}")
