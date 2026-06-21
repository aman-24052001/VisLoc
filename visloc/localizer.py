"""
Absolute localizer.

Matches a single camera-crop frame against the full reference world image
using ORB or AKAZE keypoints, filters matches with RANSAC homography, and
recovers an estimated (x, y) position of the crop's centre in world
coordinates, plus a confidence score (inlier count).

This is the "visual GPS fix" stage: slow (only run every N frames in
practice) but absolute — it doesn't drift over time the way frame-to-frame
odometry does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2


@dataclass
class LocalizationResult:
    success: bool
    x: float | None = None
    y: float | None = None
    yaw_deg: float | None = None
    n_inliers: int = 0
    n_matches: int = 0


class AbsoluteLocalizer:
    """Locates a camera crop within a larger reference image."""

    def __init__(
        self,
        world: np.ndarray,
        detector: str = "orb",
        n_features: int = 1500,
        world_n_features: int = 20000,
        min_inliers: int = 12,
        ransac_thresh: float = 4.0,
    ):
        self.world = world
        self.min_inliers = min_inliers
        self.ransac_thresh = ransac_thresh

        if detector == "orb":
            # Defaults (edgeThreshold=31, fastThreshold=20) are tuned for
            # full-size photos and starve small ~200px crops of keypoints
            # (observed as few as 5 kp on a 220x220 crop). Lowering both
            # gives hundreds-to-low-thousands of keypoints on the same crop.
            #
            # Separately: world_n_features needs to be much larger than
            # n_features. A 2000x2000 world map at n_features=1500 gives
            # only ~1 keypoint per ~50x50px region — too sparse for a
            # 220px crop to reliably find enough matches. ~20000 features
            # (~1 per 14x14px) eliminates failures in testing.
            self.crop_detector = cv2.ORB_create(
                nfeatures=n_features, edgeThreshold=10, fastThreshold=7
            )
            self.world_detector = cv2.ORB_create(
                nfeatures=world_n_features, edgeThreshold=10, fastThreshold=7
            )
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif detector == "akaze":
            self.crop_detector = cv2.AKAZE_create()
            self.world_detector = cv2.AKAZE_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            raise ValueError(f"Unknown detector: {detector!r}")

        gray_world = cv2.cvtColor(world, cv2.COLOR_BGR2GRAY)
        self.world_kp, self.world_des = self.world_detector.detectAndCompute(gray_world, None)

    def localize(self, crop: np.ndarray, k: int = 2, ratio: float = 0.75) -> LocalizationResult:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        kp, des = self.crop_detector.detectAndCompute(gray, None)

        if des is None or len(kp) < 4 or self.world_des is None:
            return LocalizationResult(success=False)

        knn = self.matcher.knnMatch(des, self.world_des, k=k)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

        if len(good) < 4:
            return LocalizationResult(success=False, n_matches=len(good))

        src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([self.world_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)
        if H is None:
            return LocalizationResult(success=False, n_matches=len(good))

        n_inliers = int(mask.sum())
        if n_inliers < self.min_inliers:
            return LocalizationResult(success=False, n_matches=len(good), n_inliers=n_inliers)

        h, w = gray.shape
        centre = np.float32([[w / 2, h / 2]]).reshape(-1, 1, 2)
        world_centre = cv2.perspectiveTransform(centre, H)[0, 0]

        # Recover yaw from the homography's rotation component.
        yaw_deg = float(np.degrees(np.arctan2(H[1, 0], H[0, 0])))

        return LocalizationResult(
            success=True,
            x=float(world_centre[0]),
            y=float(world_centre[1]),
            yaw_deg=yaw_deg,
            n_inliers=n_inliers,
            n_matches=len(good),
        )


if __name__ == "__main__":
    from visloc.world import generate_world
    from visloc.simulator import FrameSimulator, make_path

    world = generate_world()
    path = make_path("loop", n_frames=300, world_w=world.shape[1], world_h=world.shape[0])
    sim = FrameSimulator(world, path, crop_size=220, noise_std=3.0, max_yaw_deg=8.0)
    frames = sim.generate()

    loc = AbsoluteLocalizer(world, detector="orb")

    errs = []
    n_fail = 0
    for f in frames[::10]:  # simulate the 1-2Hz absolute-fix rate
        res = loc.localize(f.image)
        if not res.success:
            n_fail += 1
            continue
        err = np.hypot(res.x - f.gt_x, res.y - f.gt_y)
        errs.append(err)

    print(f"Tested {len(frames[::10])} sampled frames")
    print(f"Failures: {n_fail}")
    print(f"Mean localization error: {np.mean(errs):.2f}px, max: {np.max(errs):.2f}px")
