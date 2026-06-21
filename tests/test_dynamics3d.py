import numpy as np
import pytest

from visloc3d.dynamics import (
    RigidBodyState, RigidBodyParams, rk4_step, GRAVITY,
    quat_normalize, quat_to_rotmat, quat_from_euler, quat_to_euler,
)


def test_quaternion_roundtrip_euler():
    for r, p, y in [(0.1, -0.2, 0.3), (0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (-1.0, 0.3, 2.0)]:
        q = quat_from_euler(r, p, y)
        rpy = quat_to_euler(q)
        assert np.allclose(rpy, [r, p, y], atol=1e-9)


def test_quaternion_normalize():
    q = np.array([2.0, 0.0, 0.0, 0.0])
    qn = quat_normalize(q)
    assert np.isclose(np.linalg.norm(qn), 1.0)
    # zero-quaternion edge case shouldn't divide by zero
    qz = quat_normalize(np.zeros(4))
    assert np.allclose(qz, [1, 0, 0, 0])


def test_identity_quaternion_rotation_is_identity():
    R = quat_to_rotmat(np.array([1.0, 0, 0, 0]))
    assert np.allclose(R, np.eye(3))


def test_free_fall_matches_kinematics():
    params = RigidBodyParams(mass=1.0, inertia=np.diag([0.01, 0.01, 0.02]))
    state = RigidBodyState.hover_at(0, 0, 10)
    dt = 0.001
    for _ in range(1000):  # 1 second
        state = rk4_step(state, params, f_body=np.zeros(3), tau_body=np.zeros(3), dt=dt)
    expected_z = 10 - 0.5 * GRAVITY * 1.0 ** 2
    assert np.isclose(state.p[2], expected_z, atol=1e-3)
    assert np.isclose(state.v[2], -GRAVITY, atol=1e-3)


def test_hover_thrust_holds_position_with_no_drift():
    params = RigidBodyParams(mass=1.0, inertia=np.diag([0.01, 0.01, 0.02]))
    state = RigidBodyState.hover_at(0, 0, 10)
    thrust = np.array([0, 0, params.mass * GRAVITY])
    dt = 0.005
    for _ in range(2000):  # 10 seconds
        state = rk4_step(state, params, f_body=thrust, tau_body=np.zeros(3), dt=dt)
    assert np.isclose(state.p[2], 10.0, atol=1e-9)
    assert np.isclose(state.v[2], 0.0, atol=1e-9)


def test_torque_free_spherical_body_conserves_angular_velocity():
    """For a spherical inertia tensor, -w x Jw = 0 exactly (Euler's
    equations degenerate), so w must stay exactly constant with no
    applied torque - a hard physics invariant, not a tuning outcome."""
    params = RigidBodyParams(mass=1.0, inertia=np.diag([0.02, 0.02, 0.02]))
    state = RigidBodyState.hover_at(0, 0, 10)
    state.w = np.array([1.0, 0.5, -0.3])
    w0 = state.w.copy()
    dt = 0.005
    for _ in range(1000):  # 5 seconds
        state = rk4_step(state, params, f_body=np.zeros(3), tau_body=np.zeros(3), dt=dt)
    assert np.allclose(state.w, w0, atol=1e-9)


def test_quaternion_stays_normalized_under_rotation():
    params = RigidBodyParams(mass=1.0, inertia=np.diag([0.01, 0.01, 0.02]))
    state = RigidBodyState.hover_at(0, 0, 10)
    dt = 0.005
    for _ in range(500):
        state = rk4_step(state, params, f_body=np.array([0, 0, params.mass * GRAVITY]),
                          tau_body=np.array([0.01, 0.0, 0.0]), dt=dt)
        assert np.isclose(np.linalg.norm(state.q), 1.0, atol=1e-9)


def test_applied_torque_changes_angular_velocity_in_expected_direction():
    params = RigidBodyParams(mass=1.0, inertia=np.diag([0.01, 0.01, 0.02]))
    state = RigidBodyState.hover_at(0, 0, 10)
    dt = 0.001
    state = rk4_step(state, params, f_body=np.zeros(3), tau_body=np.array([1.0, 0, 0]), dt=dt)
    # tau_x positive with positive Ixx should produce positive w_x
    assert state.w[0] > 0
    assert np.isclose(state.w[1], 0.0, atol=1e-9)
    assert np.isclose(state.w[2], 0.0, atol=1e-9)
