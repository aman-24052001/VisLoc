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


def test_hover_electrical_power_matches_validated_battery_model():
    """Cross-model consistency check: the real motor-torque-based power
    draw at hover (Q*Omega from actual commanded motor speeds) must match
    the independently-validated momentum-theory hover power from
    battery.py - found during development that these disagreed by
    ~2x when c_drag was left as an arbitrary independent default;
    Drone.__init__ now calibrates c_drag specifically to close this gap."""
    from visloc3d.battery import hover_mechanical_power, motor_electrical_power
    drone = Drone()
    drone.reset(0, 0, 5)
    target = np.array([0, 0, 5])
    for _ in range(500):
        drone.step(target, 0.0, 0.005)
    p_theory = motor_electrical_power(hover_mechanical_power(drone.battery_spec))
    assert np.isclose(drone.last_electrical_power_w, p_theory, rtol=1e-6)


def test_battery_depletes_during_extended_hover_and_matches_estimate():
    """Full integration: actually fly (hover) until the battery reports
    depleted, and check the elapsed time matches the independently
    computed hover-endurance estimate - not just that each piece works
    in isolation, but that wiring them together preserves the validated
    numbers."""
    from visloc3d.battery import (
        hover_mechanical_power, motor_electrical_power, per_cell_power,
        effective_capacity_ratio, NOMINAL_CELL_VOLTAGE,
    )
    drone = Drone()
    drone.reset(0, 0, 5)
    target = np.array([0, 0, 5])
    dt = 0.05
    t = 0.0
    while not drone.battery_depleted and t < 4000:
        drone.step(target, 0.0, dt)
        t += dt

    spec = drone.battery_spec
    p_mech_h = hover_mechanical_power(spec)
    p_mot_h = motor_electrical_power(p_mech_h)
    p_cell_h = per_cell_power(p_mot_h, spec)
    kappa_h = effective_capacity_ratio(p_cell_h)
    expected_s = kappa_h * spec.battery_capacity_ah * NOMINAL_CELL_VOLTAGE * spec.n_cells * 3600 / p_mot_h

    assert abs(t - expected_s) / expected_s < 0.02
    # and position must not have drifted even as the battery depleted -
    # this sim doesn't (yet) model voltage-sag-induced thrust loss, an
    # explicit, documented scope limitation rather than a silent gap.
    assert np.allclose(drone.state.p, target, atol=1e-3)
