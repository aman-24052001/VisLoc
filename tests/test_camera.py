import numpy as np
import pytest
import cv2

from visloc3d.camera import (
    CameraIntrinsics, ground_homography, render_view, ground_footprint_corners,
)
from visloc3d.dynamics import quat_from_euler
from visloc.world import generate_world
from visloc.simulator import FrameSimulator, make_path


@pytest.fixture(scope="module")
def world():
    return generate_world(width=1800, height=1800, seed=42, n_blobs=200, n_roads=14)


def test_level_footprint_is_centered_axis_aligned_square():
    intrinsics = CameraIntrinsics(output_size=220, fov_deg=53.13)
    corners = ground_footprint_corners(np.array([0, 0, 200]), np.array([1, 0, 0, 0]), intrinsics)
    # axis-aligned: only two distinct x-values and two distinct y-values
    assert len(set(np.round(corners[:, 0], 4))) == 2
    assert len(set(np.round(corners[:, 1], 4))) == 2
    # centered at origin
    assert np.isclose(corners[:, 0].mean(), 0, atol=1e-6)
    assert np.isclose(corners[:, 1].mean(), 0, atol=1e-6)


def test_footprint_scales_linearly_with_altitude():
    intrinsics = CameraIntrinsics.matching_reference_footprint(220, 220, 200.0)
    widths = {}
    for z in [100, 200, 400]:
        corners = ground_footprint_corners(np.array([900, 900, z]), np.array([1, 0, 0, 0]), intrinsics)
        widths[z] = corners[:, 0].max() - corners[:, 0].min()
    assert np.isclose(widths[100], 110.0, atol=0.1)
    assert np.isclose(widths[200], 220.0, atol=0.1)
    assert np.isclose(widths[400], 440.0, atol=0.1)


def test_tilt_produces_nonrectangular_trapezoid():
    intrinsics = CameraIntrinsics.matching_reference_footprint(220, 220, 200.0)
    quat_tilted = quat_from_euler(np.radians(15), 0, 0)
    corners = ground_footprint_corners(np.array([900, 900, 200]), quat_tilted, intrinsics)
    sides = [np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    # not a rectangle: opposite sides should differ once tilted
    assert not np.isclose(sides[0], sides[2], atol=1.0)


def test_level_footprint_matches_old_axis_aligned_crop_structurally(world):
    """Validates geometric correspondence, not raw pixel equality - the
    world map's per-pixel speckle noise decorrelates under resampling
    (confirmed: raw diff 7.7/255, but the same comparison after blurring
    both images to average out that noise drops to 1.8/255), so a tight
    raw-pixel tolerance would be the wrong test for this content. Blur
    first, matching the validation approach used to establish this."""
    z_ref = 200.0
    intrinsics = CameraIntrinsics.matching_reference_footprint(
        output_size=220, footprint_at_z_ref=220, z_ref=z_ref)
    position = np.array([900, 900, z_ref])
    rendered = render_view(world, position, np.array([1, 0, 0, 0]), intrinsics)

    path = make_path("straight", n_frames=2, world_w=1800, world_h=1800)
    sim = FrameSimulator(world, path, crop_size=220, noise_std=0.0, max_yaw_deg=0.0)
    old_crop = sim._crop_at(900, 900, 0.0)

    blur_a = cv2.GaussianBlur(rendered, (9, 9), 0).astype(float)
    blur_b = cv2.GaussianBlur(old_crop, (9, 9), 0).astype(float)
    assert np.abs(blur_a - blur_b).mean() < 4.0


def test_higher_altitude_gives_visibly_more_zoomed_out_view(world):
    intrinsics = CameraIntrinsics.matching_reference_footprint(220, 220, 200.0)
    near = render_view(world, np.array([900, 900, 100]), np.array([1, 0, 0, 0]), intrinsics)
    far = render_view(world, np.array([900, 900, 400]), np.array([1, 0, 0, 0]), intrinsics)
    # not a content check (different footprints, can't pixel-compare) -
    # just confirms both render successfully at very different scales
    # without degenerate/empty output.
    assert near.shape == far.shape == (220, 220, 3)
    assert near.std() > 0 and far.std() > 0


def test_homography_is_well_conditioned_for_realistic_tilt(world):
    """A degenerate or near-singular homography would indicate the
    camera model breaks down well before realistic flight attitudes -
    check it stays well-conditioned up to a meaningfully large tilt."""
    intrinsics = CameraIntrinsics.matching_reference_footprint(220, 220, 200.0)
    quat_tilted = quat_from_euler(np.radians(30), np.radians(20), 0)
    H = ground_homography(np.array([900, 900, 200]), quat_tilted, intrinsics)
    cond = np.linalg.cond(H)
    assert cond < 1e6


def test_tilt_shifts_image_center_ground_point_by_altitude_times_tan_tilt():
    """The key finding from integrating this with the ORB localizer: its
    growing position error under tilt (1.2px at 0deg -> 116px at 30deg)
    looked at first like an ORB/matching weakness, but turned out to be a
    pure, deterministic geometric fact, confirmed here using the *exact*
    homography with no feature matching involved at all - a tilted rigid
    (non-gimbal) camera's image center looks at a ground point shifted
    from the drone's actual (x,y) by altitude*tan(tilt), simple
    trigonometry. This is exactly *why* the original ArduPilot project
    this work is modeled on required a gimbal-stabilized camera -
    confirmed quantitatively from first principles rather than just
    citing it. A localizer wanting true drone position (not "where the
    camera is pointing") from a rigid mount would need to know/estimate
    attitude and subtract this offset explicitly - left as a documented
    follow-on, not fixed here.
    """
    intrinsics = CameraIntrinsics.matching_reference_footprint(220, 220, 200.0)
    z = 200.0
    position = np.array([900.0, 900.0, z])
    for tilt_deg in [2, 5, 10, 15]:
        quat = quat_from_euler(np.radians(tilt_deg), 0, 0)
        H = ground_homography(position, quat, intrinsics)
        center_world_h = np.linalg.inv(H) @ np.array([110.0, 110.0, 1.0])
        center_world = center_world_h[:2] / center_world_h[2]
        observed_offset = center_world[1] - 900.0  # roll tilts the Y component
        expected_offset = z * np.tan(np.radians(tilt_deg))
        assert np.isclose(observed_offset, expected_offset, rtol=0.01)
