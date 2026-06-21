import numpy as np

from visloc.world import generate_world
from visloc.simulator import FrameSimulator, make_path
from visloc.odometry import RelativeOdometry, OdometryResult


def _frames(n_frames=80, noise_std=1.5, max_yaw_deg=0.0, seed=7):
    world = generate_world(width=1400, height=1400, seed=42, n_blobs=160, n_roads=10)
    path = make_path("straight", n_frames=n_frames, world_w=1400, world_h=1400)
    sim = FrameSimulator(world, path, crop_size=200, noise_std=noise_std, max_yaw_deg=max_yaw_deg, seed=seed)
    return sim.generate()


def test_estimate_recovers_known_translation_direction():
    # n_frames chosen so per-frame displacement (~24px) stays well within
    # what default-window LK can track against a 200px crop - mirrors a
    # realistic camera frame rate relative to flight speed, not a stress
    # test of LK's large-displacement limits.
    frames = _frames(n_frames=60, noise_std=0.0, max_yaw_deg=0.0)
    odo = RelativeOdometry()

    res = odo.estimate(frames[5].image, frames[6].image)
    assert isinstance(res, OdometryResult)
    assert res.success
    assert res.n_tracked >= 8

    true_dx = frames[6].gt_x - frames[5].gt_x
    true_dy = frames[6].gt_y - frames[5].gt_y
    # "straight" path only moves in x - estimate should agree in sign and
    # be close in magnitude.
    assert abs(res.dx - true_dx) < 3.0
    assert abs(res.dy - true_dy) < 3.0


def test_estimate_fails_gracefully_on_blank_frames():
    odo = RelativeOdometry()
    blank_a = np.full((200, 200, 3), 128, dtype=np.uint8)
    blank_b = np.full((200, 200, 3), 128, dtype=np.uint8)
    res = odo.estimate(blank_a, blank_b)
    assert not res.success


def test_track_path_matches_ground_truth_under_zero_noise_zero_yaw():
    frames = _frames(n_frames=60, noise_std=0.0, max_yaw_deg=0.0)
    odo = RelativeOdometry()
    est_path = odo.track_path(frames)
    gt_path = np.array([[f.gt_x, f.gt_y] for f in frames])

    err = np.hypot(est_path[:, 0] - gt_path[:, 0], est_path[:, 1] - gt_path[:, 1])
    # Pure translation, no rotation, no noise - should track almost
    # perfectly with negligible accumulated error.
    assert err[-1] < 3.0


def test_drift_grows_under_yaw_lk_cannot_model():
    """Translation-only LK can't account for rotation - error should grow
    roughly monotonically as small uncorrected yaw accumulates, mirroring
    the real system's documented limitation."""
    frames = _frames(n_frames=150, noise_std=1.5, max_yaw_deg=2.0, seed=7)
    odo = RelativeOdometry()
    est_path = odo.track_path(frames)
    gt_path = np.array([[f.gt_x, f.gt_y] for f in frames])
    err = np.hypot(est_path[:, 0] - gt_path[:, 0], est_path[:, 1] - gt_path[:, 1])

    # Error late in the sequence should clearly exceed error early on -
    # this is the "drift" the UKF fusion stage exists to correct.
    early_err = err[10:30].mean()
    late_err = err[-30:].mean()
    assert late_err > early_err * 1.5


def test_track_path_output_shape():
    frames = _frames(n_frames=40)
    odo = RelativeOdometry()
    path = odo.track_path(frames)
    assert path.shape == (40, 2)
