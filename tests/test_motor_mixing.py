import numpy as np
import pytest

from visloc3d.motor_mixing import QuadrotorMixer, QuadrotorParams


def _mixer():
    return QuadrotorMixer(QuadrotorParams(
        arm_length=0.25, c_thrust=3e-5, c_drag=5e-7, max_motor_speed=900,
    ))


def test_mix_allocate_roundtrip_exact_for_feasible_wrench():
    mixer = _mixer()
    desired_thrust = 12.0
    desired_tau = np.array([0.3, -0.2, 0.1])
    omega_sq = mixer.allocate(desired_thrust, desired_tau)
    assert np.all(omega_sq >= 0)
    f_body, tau_body = mixer.mix(omega_sq)
    assert np.isclose(f_body[2], desired_thrust)
    assert np.allclose(tau_body, desired_tau)


def test_pure_roll_does_not_leak_into_pitch_or_yaw():
    mixer = _mixer()
    hover_thrust = 9.80665
    omega_sq = mixer.allocate(hover_thrust, np.array([0.05, 0, 0]))
    _, tau = mixer.mix(omega_sq)
    assert np.isclose(tau[0], 0.05, atol=1e-9)
    assert np.isclose(tau[1], 0.0, atol=1e-9)
    assert np.isclose(tau[2], 0.0, atol=1e-9)


def test_pure_pitch_does_not_leak_into_roll_or_yaw():
    mixer = _mixer()
    hover_thrust = 9.80665
    omega_sq = mixer.allocate(hover_thrust, np.array([0, 0.05, 0]))
    _, tau = mixer.mix(omega_sq)
    assert np.isclose(tau[0], 0.0, atol=1e-9)
    assert np.isclose(tau[1], 0.05, atol=1e-9)
    assert np.isclose(tau[2], 0.0, atol=1e-9)


def test_pure_yaw_within_feasible_range_does_not_leak():
    mixer = _mixer()
    hover_thrust = 9.80665
    omega_sq = mixer.allocate(hover_thrust, np.array([0, 0, 0.005]))
    assert np.all(omega_sq > 0)  # confirm this request is within the feasible region
    _, tau = mixer.mix(omega_sq)
    assert np.isclose(tau[0], 0.0, atol=1e-9)
    assert np.isclose(tau[1], 0.0, atol=1e-9)
    assert np.isclose(tau[2], 0.005, atol=1e-9)


def test_large_yaw_request_saturates_rather_than_lying():
    """Yaw authority on a quadrotor is fundamentally limited by the
    thrust/drag-torque ratio - a large enough yaw request at full hover
    thrust requires negative motor speed on some motor, which is
    physically impossible and must saturate, not silently produce the
    full requested torque."""
    mixer = _mixer()
    hover_thrust = 9.80665
    omega_sq = mixer.allocate(hover_thrust, np.array([0, 0, 0.5]))
    _, tau = mixer.mix(omega_sq)
    assert tau[2] < 0.5  # cannot achieve the full request
    assert np.any(omega_sq <= 0) or np.any(omega_sq >= mixer.p.max_motor_speed ** 2)


def test_symmetric_hover_gives_equal_motor_speeds():
    mixer = _mixer()
    omega_sq = mixer.allocate(9.80665, np.zeros(3))
    assert np.allclose(omega_sq, omega_sq[0], atol=1e-6)
