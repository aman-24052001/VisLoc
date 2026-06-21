import numpy as np
import pytest

from visloc.world import generate_world
from visloc.simulator import FrameSimulator, make_path, Frame


def test_generate_world_shape_and_dtype():
    world = generate_world(width=400, height=300, seed=1, n_blobs=20, n_roads=3)
    assert world.shape == (300, 400, 3)
    assert world.dtype == np.uint8


def test_generate_world_deterministic():
    a = generate_world(width=200, height=200, seed=7)
    b = generate_world(width=200, height=200, seed=7)
    assert np.array_equal(a, b)


def test_generate_world_seed_changes_output():
    a = generate_world(width=200, height=200, seed=1)
    b = generate_world(width=200, height=200, seed=2)
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("kind", ["loop", "zigzag", "straight"])
def test_make_path_shape(kind):
    path = make_path(kind, n_frames=50, world_w=1000, world_h=800)
    assert path.shape == (50, 2)
    # All waypoints should stay within world bounds.
    assert np.all(path[:, 0] >= 0) and np.all(path[:, 0] <= 1000)
    assert np.all(path[:, 1] >= 0) and np.all(path[:, 1] <= 800)


def test_make_path_unknown_kind_raises():
    with pytest.raises(ValueError):
        make_path("not_a_real_path", n_frames=10, world_w=100, world_h=100)


def test_frame_simulator_generates_correct_count_and_size():
    world = generate_world(width=500, height=500, seed=3, n_blobs=30)
    path = make_path("straight", n_frames=20, world_w=500, world_h=500)
    sim = FrameSimulator(world, path, crop_size=64, noise_std=0.0)
    frames = sim.generate()

    assert len(frames) == 20
    assert all(isinstance(f, Frame) for f in frames)
    assert all(f.image.shape == (64, 64, 3) for f in frames)


def test_frame_simulator_ground_truth_matches_path():
    world = generate_world(width=500, height=500, seed=3, n_blobs=30)
    path = make_path("straight", n_frames=20, world_w=500, world_h=500)
    sim = FrameSimulator(world, path, crop_size=64, noise_std=0.0)
    frames = sim.generate()

    for f, (px, py) in zip(frames, path):
        assert f.gt_x == px
        assert f.gt_y == py


def test_frame_simulator_crop_stays_in_bounds():
    world = generate_world(width=300, height=300, seed=5, n_blobs=10)
    # Path that goes right up to the edges to stress-test clipping.
    path = make_path("straight", n_frames=10, world_w=300, world_h=300, margin=5)
    sim = FrameSimulator(world, path, crop_size=64, noise_std=0.0)
    frames = sim.generate()
    assert all(f.image.shape == (64, 64, 3) for f in frames)
