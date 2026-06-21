import numpy as np
import pytest

from visloc.world import generate_world
from visloc.simulator import FrameSimulator, make_path
from visloc.fusion import UKFFusion, FusionConfig, run_fusion


def _scenario(path_kind="loop", n_frames=200, noise_std=1.5, max_yaw_deg=2.0, seed=7):
    world = generate_world(width=1800, height=1800, seed=42, n_blobs=200, n_roads=14)
    path = make_path(path_kind, n_frames=n_frames, world_w=1800, world_h=1800)
    sim = FrameSimulator(world, path, crop_size=200, noise_std=noise_std, max_yaw_deg=max_yaw_deg, seed=seed)
    return world, sim.generate()


def test_step_before_bootstrap_raises():
    ukf = UKFFusion()
    with pytest.raises(RuntimeError):
        ukf.step(np.array([0, 0, 0, 0]), None)


def test_bootstrap_sets_state_and_flag():
    ukf = UKFFusion()
    assert not ukf.is_bootstrapped
    ukf.bootstrap(100.0, 200.0, 1.0, -1.0)
    assert ukf.is_bootstrapped
    np.testing.assert_array_almost_equal(ukf.ukf.x, [100.0, 200.0, 1.0, -1.0])


def test_soft_correction_is_gradual_not_instant():
    """After a VPS fix far from the current estimate, the very next step
    should move only partway there, not snap instantly - that's the whole
    point of soft correction."""
    ukf = UKFFusion(FusionConfig(vps_soft_frames=10, vps_chi2_threshold=None))
    ukf.bootstrap(0.0, 0.0, 5.0, 0.0)

    target = np.array([100.0, 0.0])
    vio_obs = np.array([5.0, 0.0, 5.0, 0.0])  # consistent small VIO motion
    fused_xy = ukf.step(vio_obs, target)

    dist_to_target = np.hypot(*(target - fused_xy))
    # Should have moved meaningfully toward the target, but not be there yet.
    assert dist_to_target > 5.0
    assert dist_to_target < 100.0


def test_soft_correction_converges_over_n_steps():
    ukf = UKFFusion(FusionConfig(vps_soft_frames=10, vps_chi2_threshold=None))
    ukf.bootstrap(0.0, 0.0, 0.0, 0.0)

    target = np.array([50.0, 50.0])
    # Anchor velocity at zero each step (consistent with this test's
    # premise: not moving on its own, only being corrected toward a
    # target) without fighting position - feed the filter's own current
    # position back so the position component is neutral. Without any
    # velocity anchoring at all, repeated position-only corrections make
    # the filter infer a runaway "phantom" velocity via position/velocity
    # cross-covariance and overshoot well past the target - a real
    # dynamic, but one that can't happen in the actual pipeline since
    # run_fusion always feeds a genuine VIO observation every frame.
    def vio_obs_anchored_velocity():
        cur = ukf.ukf.x[:2]
        return np.array([cur[0], cur[1], 0.0, 0.0])

    fused_xy = ukf.step(vio_obs_anchored_velocity(), target)
    for _ in range(9):
        fused_xy = ukf.step(vio_obs_anchored_velocity(), None)

    initial_gap = np.hypot(*target)  # distance from bootstrap origin (0,0) to target
    remaining_gap = np.hypot(*(target - fused_xy))
    # Kalman updates never fully snap to a measurement (gain < 1), so
    # exact convergence after exactly vps_soft_frames partial-trust
    # updates isn't expected - meaningful progress is the right bar here.
    assert remaining_gap < 0.5 * initial_gap


def test_mahalanobis_gate_rejects_anomalous_fix():
    ukf = UKFFusion(FusionConfig(vps_soft_frames=10, vps_chi2_threshold=9.21, vps_noise_std=4.0))
    ukf.bootstrap(0.0, 0.0, 1.0, 0.0)

    # Run a few normal steps to tighten covariance somewhat.
    for _ in range(5):
        ukf.step(np.array([1.0, 0.0, 1.0, 0.0]), None)

    pre_xy = ukf.ukf.x[:2].copy()
    wild_fix = pre_xy + np.array([5000.0, 5000.0])  # absurd, should be gated out
    fused_xy = ukf.step(np.array([1.0, 0.0, 1.0, 0.0]), wild_fix)

    # State should NOT have jumped toward the absurd fix.
    assert np.hypot(*(fused_xy - wild_fix)) > 1000.0


def test_filter_does_not_crash_under_aggressive_process_noise():
    """Regression test for a real Cholesky-failure crash found during
    tuning at higher process_noise_std values - matters because Phase 5's
    parameter sandbox will let users pick arbitrary slider values."""
    world, frames = _scenario()
    for q_std in [0.5, 2.0, 5.0, 10.0, 25.0, 50.0]:
        cfg = FusionConfig(process_noise_std=q_std)
        result = run_fusion(frames, world, vps_rate=10, config=cfg)
        assert not np.any(np.isnan(result["fused_path"][result["vps_fix_frames"][0]:]))


def test_run_fusion_no_nan_after_bootstrap():
    world, frames = _scenario()
    result = run_fusion(frames, world, vps_rate=10)
    fused = result["fused_path"]
    first_fix = result["vps_fix_frames"][0]
    assert not np.any(np.isnan(fused[first_fix:]))


def test_fusion_meaningfully_improves_on_canonical_scenario():
    """The headline result: on the standard loop/±2deg-yaw scenario (seed=7,
    used throughout this project), UKF fusion should substantially
    outperform raw odometry. This is a regression test pinned to the
    specific scenario - fusion's benefit is seed/scenario-dependent (see
    README 'When fusion helps' section), so this is deliberately not a
    claim that it always wins."""
    world, frames = _scenario(seed=7)
    result = run_fusion(frames, world, vps_rate=10)
    fused, vio, gt = result["fused_path"], result["vio_global_path"], result["gt_path"]
    valid = ~np.isnan(fused[:, 0])

    err_fused = np.hypot(fused[valid, 0] - gt[valid, 0], fused[valid, 1] - gt[valid, 1])
    err_vio = np.hypot(vio[valid, 0] - gt[valid, 0], vio[valid, 1] - gt[valid, 1])

    assert err_fused.mean() < 0.7 * err_vio.mean()
    assert err_fused[-1] < 0.5 * err_vio[-1]


def test_fusion_does_not_blow_up_when_baseline_drift_is_already_small():
    """Characterization test for the 'ceiling effect' found during
    validation: when raw VIO is already near the localizer's own noise
    floor, fusion shouldn't catastrophically diverge, even if it doesn't
    meaningfully improve on it."""
    world, frames = _scenario(seed=13)
    result = run_fusion(frames, world, vps_rate=10)
    fused, gt = result["fused_path"], result["gt_path"]
    valid = ~np.isnan(fused[:, 0])
    err_fused = np.hypot(fused[valid, 0] - gt[valid, 0], fused[valid, 1] - gt[valid, 1])
    assert err_fused.mean() < 15.0  # bounded, not diverging
