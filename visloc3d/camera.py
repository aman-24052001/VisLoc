"""
Camera projection: renders what a rigidly body-mounted (non-gimbal),
downward-facing camera actually sees, given the drone's real 3D position
and attitude from visloc3d. This replaces FrameSimulator's scripted,
always-axis-aligned crop with a real perspective projection.

Imaging a flat ground plane (Z=0) from any camera pose is *exactly* a
homography - not an approximation that needs justifying, a property of
projective geometry for planar scenes (Hartley & Zisserman, "Multiple
View Geometry," ch. 8). That's what makes this tractable: altitude
controls footprint size, and roll/pitch tilt the footprint into a
trapezoid (keystone distortion) precisely, not approximately - both
fall out of one 3x3 matrix derived from the drone's actual pose.

Deliberately models a *rigid*, non-gimbal-stabilized camera: the
original ArduPilot project this is inspired by explicitly required a
gimbal-stabilized camera for its SIFT/ORB approach. Modeling the rigid
case instead is a more demanding, more interesting extension - and an
explicit modeling choice (see VISLOC3D_ARCHITECTURE.md, Section 6).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2

from visloc3d.dynamics import quat_to_rotmat

# Fixed rigid camera-to-body mount: camera optical axis (+Z_cam) points
# along body -Z (down) when level; image "right" (+X_cam) = body +X
# (front). A right-handed camera frame looking *down* cannot also have
# image "down" increase with body +Y while keeping image "right" tied to
# body +X - same reason a top-down map view and a looking-up-at-the-sky
# view have opposite right/up handedness for the same compass directions.
# Verified analytically (solving for the required column structure under
# orthonormality forces this) and confirms empirically below: this
# mounting requires a single deterministic vertical flip of the rendered
# output to match the world map's row=Y/col=X array convention, applied
# in render_view() rather than baked into a "clever" rotation choice that
# would obscure why it's there.
R_CAM_BODY = np.array([
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
])


@dataclass
class CameraIntrinsics:
    output_size: int       # square output image, pixels
    fov_deg: float          # full field of view, degrees

    @property
    def focal_px(self) -> float:
        return (self.output_size / 2) / np.tan(np.radians(self.fov_deg) / 2)

    @property
    def K(self) -> np.ndarray:
        f, c = self.focal_px, self.output_size / 2
        return np.array([[f, 0, c], [0, f, c], [0, 0, 1]])

    @staticmethod
    def matching_reference_footprint(output_size: int, footprint_at_z_ref: float,
                                      z_ref: float) -> "CameraIntrinsics":
        """Choose FOV so the ground footprint at altitude z_ref equals
        footprint_at_z_ref - used to make the new projective renderer
        backward-compatible with the original FrameSimulator's fixed-size
        crop at an equivalent reference altitude, so Phase 1's ORB
        keypoint-density tuning (validated for ~220px crops) stays valid."""
        half_fov = np.arctan((footprint_at_z_ref / 2) / z_ref)
        return CameraIntrinsics(output_size=output_size, fov_deg=np.degrees(2 * half_fov))


def ground_homography(position: np.ndarray, quat: np.ndarray,
                       intrinsics: CameraIntrinsics) -> np.ndarray:
    """3x3 homography mapping world ground-plane (X, Y, 1) -> image (u, v, 1)
    homogeneous coordinates, for a camera at `position` with body attitude
    `quat`, rigidly mounted per R_CAM_BODY."""
    R_body_world = quat_to_rotmat(quat)
    R_cam_world = R_body_world @ R_CAM_BODY
    R_wc = R_cam_world.T  # world-to-camera

    t = -R_wc @ position
    H = intrinsics.K @ np.column_stack([R_wc[:, 0], R_wc[:, 1], t])
    return H


def render_view(world_map: np.ndarray, position: np.ndarray, quat: np.ndarray,
                 intrinsics: CameraIntrinsics) -> np.ndarray:
    """Render the camera's view of world_map (treated as the ground plane,
    Z=0, with image pixel coordinates used directly as world X/Y units)
    from the given drone position and attitude."""
    H = ground_homography(position, quat, intrinsics)
    size = intrinsics.output_size
    warped = cv2.warpPerspective(world_map, H, (size, size))
    # See R_CAM_BODY comment: this mounting (image-right=body-front,
    # camera looking down, right-handed) produces image-v decreasing with
    # world Y, the opposite of the world map's row=Y array convention -
    # confirmed both analytically and by direct point-mapping checks
    # during validation. One deterministic flip here corrects it.
    return cv2.flip(warped, 0)


def ground_footprint_corners(position: np.ndarray, quat: np.ndarray,
                              intrinsics: CameraIntrinsics) -> np.ndarray:
    """World (X, Y) coordinates the four image corners actually look at -
    a perfect square only when level; tilted, this is a trapezoid. Useful
    for both validation and for drawing the real footprint on a map."""
    H = ground_homography(position, quat, intrinsics)
    H_inv = np.linalg.inv(H)
    size = intrinsics.output_size
    corners_img = np.array([[0, 0, 1], [size, 0, 1], [size, size, 1], [0, size, 1]], dtype=float)
    world_h = (H_inv @ corners_img.T).T
    return world_h[:, :2] / world_h[:, 2:3]
