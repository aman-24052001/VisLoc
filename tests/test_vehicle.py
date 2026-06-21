import numpy as np
import pytest

from visloc3d.vehicle import Drone, DroneSpec
from visloc3d.dynamics import GRAVITY, quat_to_euler


def test_hover_holds_position_indefinitely():
    drone = Drone()
    drone.reset(0, 0, 5)
    target = np.array([0, 0, 5])
    dt = 0.005
    for _ in range(2000):  # 10s
        state = drone.step(target, 0.0, dt)
    assert np.allclose(state.p, target, atol=1e-6)
    assert np.allclose(state.v, 0.0, atol=1e-6)


def test_hover_motor_speeds_match_momentum_theory():
    spec = DroneSpec()
    expected_omega_sq = (spec.mass * GRAVITY) / (4 * spec.c_thrust)
    drone = Drone(spec)
    drone.reset(0, 0, 5)
    target = np.array([0, 0, 5])
    for _ in range(500):
        drone.step(target, 0.0, 0.005)
    assert np.allclose(drone.last_omega_sq, expected_omega_sq, rtol=1e-3)


def test_step_climb_converges_without_excessive_overshoot():
    drone = Drone()
    drone.reset(0, 0, 0)
    target = np.array([0, 0, 10])
    dt = 0.005
    zs = []
    for _ in range(4000):  # 20s
        state = drone.step(target, 0.0, dt)
        zs.append(state.p[2])
    zs = np.array(zs)
    assert np.isclose(zs[-1], 10.0, atol=1e-2)
    overshoot = max(0.0, zs.max() - 10.0)
    assert overshoot < 1.0  # less than 10% overshoot on a 10m step


def test_waypoint_tracking_converges_exactly():
    drone = Drone()
    drone.reset(0, 0, 5)
    target = np.array([10, 5, 5])
    dt = 0.005
    for _ in range(6000):  # 30s
        state = drone.step(target, 0.0, dt)
    assert np.allclose(state.p, target, atol=1e-2)


def test_waypoint_tracking_tilts_then_levels_out():
    """Confirms the controller achieves translation by genuinely tilting
    (the only way a quadrotor can move horizontally - it has no direct
    horizontal actuator), and returns to level once settled, rather than
    some degenerate non-physical solution."""
    drone = Drone()
    drone.reset(0, 0, 5)
    target = np.array([10, 5, 5])
    dt = 0.005
    pitches = []
    for i in range(2000):  # 10s
        state = drone.step(target, 0.0, dt)
        _, pitch, _ = quat_to_euler(state.q)
        pitches.append(pitch)
    pitches = np.array(pitches)
    assert pitches.max() > np.radians(5)  # meaningfully pitched at some point
    assert abs(pitches[-1]) < np.radians(1)  # but settled level


def test_reset_clears_controller_integral_state():
    """A stale integral term from a previous run leaking into a fresh
    reset would cause an unexplained initial transient - reset() must
    actually clear it, not just reset position."""
    drone = Drone()
    drone.reset(0, 0, 0)
    target = np.array([5, 0, 5])
    for _ in range(1000):
        drone.step(target, 0.0, 0.005)
    drone.reset(0, 0, 5)
    assert all(pid._integral == 0.0 for pid in drone.controller.rate_pid)
