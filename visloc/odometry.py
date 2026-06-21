"""
Relative odometry (frame-to-frame motion).

Tracks sparse features between consecutive camera-crop frames using
pyramidal Lucas-Kanade optical flow, and estimates the camera's relative
world-space displacement between frames. This is the fast (every frame)
but drift-prone signal that the UKF will later fuse with the slow-but-
absolute localizer fix.

Sign convention (verified empirically, see commit history): when the
camera moves by +delta in world space, the ground content in the image
shifts by -delta (content moves opposite to camera motion, since the crop
is centred on the camera). So: world_delta = -median(tracked_point_shift).

Assumes a non-rotating (or near-zero-yaw) camera, matching the constraint
the original ArduPilot project also had for its optical-flow approach
("requires bottom facing gimbal stabilized cameras"). Rotation handling is
out of scope for this module - the UKF/yaw correction layer would need to
account for it separately.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2


@dataclass
class OdometryResult:
    success: bool
    dx: float = 0.0
    dy: float = 0.0
    n_tracked: int = 0


class RelativeOdometry:
    """Frame-to-frame motion estimator using sparse Lucas-Kanade flow.

    Constraint (found during testing, also true of real LK trackers): the
    camera's per-frame displacement needs to stay comfortably within the
    tracker's capture range relative to crop size - e.g. ~100px/frame
    against a 200px crop overwhelmed default-window LK and produced wrong-
    sign, wrong-magnitude estimates. In practice this just means the
    simulated frame rate needs to be high enough relative to flight speed,
    which mirrors why real VIO systems care about camera frame rate vs.
    platform velocity.
    """

    def __init__(
        self,
        max_corners: int = 300,
        quality_level: float = 0.01,
        min_distance: int = 7,
        win_size: tuple[int, int] = (21, 21),
        min_tracked: int = 8,
    ):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.win_size = win_size
        self.min_tracked = min_tracked

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def estimate(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> OdometryResult:
        prev_gray = self._to_gray(prev_frame)
        curr_gray = self._to_gray(curr_frame)

        pts0 = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
        )
        if pts0 is None or len(pts0) < self.min_tracked:
            return OdometryResult(success=False, n_tracked=0 if pts0 is None else len(pts0))

        pts1, status, _err = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts0, None, winSize=self.win_size
        )
        status = status.reshape(-1).astype(bool)
        n_tracked = int(status.sum())
        if n_tracked < self.min_tracked:
            return OdometryResult(success=False, n_tracked=n_tracked)

        disp = (pts1[status] - pts0[status]).reshape(-1, 2)
        # Median over mean: robust to a handful of mistracked points
        # (occlusion, repeated texture) without needing explicit outlier
        # rejection like RANSAC.
        median_disp = np.median(disp, axis=0)

        world_dx, world_dy = -median_disp[0], -median_disp[1]
        return OdometryResult(success=True, dx=float(world_dx), dy=float(world_dy), n_tracked=n_tracked)

    def track_path(self, frames: list, start_xy: tuple[float, float] | None = None) -> np.ndarray:
        """Integrate frame-to-frame estimates into an absolute path.

        By default, starts at frames[0]'s ground-truth position (mirrors
        how the real system bootstraps from a single fix) and accumulates
        relative deltas from there with no further correction - this is
        the "raw odometry only" baseline that's expected to drift.

        Pass start_xy explicitly to integrate from an arbitrary local
        origin instead (e.g. (0, 0)) - used when VIO's own unanchored
        local frame is needed before it's been aligned to world
        coordinates via a VPS fix (see visloc/fusion.py).

        Known limitation (two fixes tried and rejected - see commit
        history): a single catastrophic LK mistracking event at a sharp
        direction-reversal point (confirmed on a zigzag corner) gets
        permanently baked into this cumulative sum, with no way to undo
        it - and can eventually grow large enough that a downstream
        Mahalanobis gate (fusion.py) starts rejecting the correction
        too, since a corrupted state and a bad measurement look
        identical to it. Two outlier-rejection attempts were tried:
        (1) substituting a rolling-median delta on large deviation -
        rejected because a *genuine* sharp turn deviates from recent
        history by a similar magnitude to a bad estimate, so real turns
        got suppressed too, sometimes permanently; (2) median-filtering
        the whole delta sequence before integrating - rejected because
        median filtering isn't bias-preserving (it favours whichever
        value is locally more frequent), introducing a small systematic
        bias on *every* normal frame that compounds via cumsum and was
        worse than the rare event it targeted. Left unfiltered: rare
        single-frame corruption is a known, undocumented-away limitation
        rather than a silently-introduced bias on the common case. The
        real ngps_flight project this is modeled on has the same open
        gap (listed in its own TODO as "no fallback VO pipeline yet").

        Returns an array of shape (len(frames), 2) of estimated (x, y).
        On a failed estimate between two frames, holds position (zero
        displacement) rather than crashing, and that frame's contribution
        to error should be interpreted accordingly.
        """
        path = np.zeros((len(frames), 2), dtype=np.float64)
        path[0] = start_xy if start_xy is not None else [frames[0].gt_x, frames[0].gt_y]

        for i in range(1, len(frames)):
            res = self.estimate(frames[i - 1].image, frames[i].image)
            if res.success:
                path[i] = path[i - 1] + [res.dx, res.dy]
            else:
                path[i] = path[i - 1]
        return path


if __name__ == "__main__":
    from visloc.world import generate_world
    from visloc.simulator import FrameSimulator, make_path

    world = generate_world(width=1600, height=1600, seed=42, n_blobs=180, n_roads=12)
    path = make_path("loop", n_frames=200, world_w=world.shape[1], world_h=world.shape[0])
    sim = FrameSimulator(world, path, crop_size=200, noise_std=1.5, max_yaw_deg=0.0, seed=7)
    frames = sim.generate()

    odo = RelativeOdometry()
    est_path = odo.track_path(frames)
    gt_path = np.array([[f.gt_x, f.gt_y] for f in frames])

    errors = np.hypot(est_path[:, 0] - gt_path[:, 0], est_path[:, 1] - gt_path[:, 1])
    print(f"Frames: {len(frames)}")
    print(f"Final drift (last-frame error): {errors[-1]:.1f}px")
    print(f"Mean error: {errors.mean():.1f}px, Max: {errors.max():.1f}px")
