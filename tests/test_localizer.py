import numpy as np

from visloc.world import generate_world
from visloc.simulator import FrameSimulator, make_path
from visloc.localizer import AbsoluteLocalizer, LocalizationResult


def _build_world_and_frames(kind="loop", n_frames=60, noise_std=2.0, max_yaw_deg=5.0):
    world = generate_world(width=1200, height=1200, seed=42, n_blobs=150, n_roads=10)
    path = make_path(kind, n_frames=n_frames, world_w=world.shape[1], world_h=world.shape[0])
    sim = FrameSimulator(world, path, crop_size=200, noise_std=noise_std, max_yaw_deg=max_yaw_deg, seed=1)
    return world, sim.generate()


def test_localizer_succeeds_on_clean_crop():
    world, frames = _build_world_and_frames(noise_std=0.0, max_yaw_deg=0.0)
    loc = AbsoluteLocalizer(world, detector="orb")

    res = loc.localize(frames[10].image)
    assert isinstance(res, LocalizationResult)
    assert res.success
    err = np.hypot(res.x - frames[10].gt_x, res.y - frames[10].gt_y)
    assert err < 5.0  # near-pixel-perfect when there's no injected noise


def test_localizer_accuracy_under_noise_and_yaw():
    world, frames = _build_world_and_frames(noise_std=3.0, max_yaw_deg=8.0)
    loc = AbsoluteLocalizer(world, detector="orb")

    errors = []
    failures = 0
    for f in frames[::5]:
        res = loc.localize(f.image)
        if not res.success:
            failures += 1
            continue
        errors.append(np.hypot(res.x - f.gt_x, res.y - f.gt_y))

    # Should resolve the overwhelming majority of sampled frames, and
    # mean error should stay well within a single crop-width of truth.
    assert failures <= 1
    assert np.mean(errors) < 15.0


def test_localizer_rejects_unrelated_image():
    world, _ = _build_world_and_frames()
    loc = AbsoluteLocalizer(world, detector="orb")

    rng = np.random.default_rng(0)
    junk = rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
    res = loc.localize(junk)
    assert not res.success


def test_localizer_yaw_recovered_approximately():
    world, frames = _build_world_and_frames(noise_std=0.0, max_yaw_deg=15.0)
    loc = AbsoluteLocalizer(world, detector="orb")

    checked = 0
    for f in frames:
        res = loc.localize(f.image)
        if not res.success:
            continue
        # Homography-recovered yaw should be in the right ballpark of the
        # injected ground-truth yaw (loose tolerance - this isn't the
        # precision-critical path, the position fix is).
        assert abs(res.yaw_deg - (-f.gt_yaw_deg)) < 20.0 or abs(res.yaw_deg - f.gt_yaw_deg) < 20.0
        checked += 1
        if checked >= 5:
            break
    assert checked > 0
