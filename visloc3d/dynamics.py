"""
6-DOF rigid body dynamics (Newton-Euler formalism).

State: x = (p, q, v, w)
  p - position, world frame, ENU (X=East, Y=North, Z=Up)
  q - attitude unit quaternion [w, x, y, z], body frame FLU (front-left-up),
      chosen specifically to avoid gimbal lock - a pure Euler-angle
      integration breaks down at pitch = +-90 deg, which a tumbling or
      aggressively-controlled drone can absolutely reach during testing.
  v - velocity, world frame
  w - body angular rate (the quantity an onboard gyroscope actually
      measures, and the natural state for the inner control loop)

This module is deliberately generic: it propagates a rigid body forward
given a body-frame force and torque, however those were produced. Motor-
specific force/torque generation lives in motor_mixing.py - this keeps
the physics propagator testable independent of any particular vehicle
configuration, mirroring how the UKF in the main VisLoc branch was kept
separate from the VIO/VPS data that feeds it.

Formalism follows the standard Newton-Euler UAV model used across nearly
all major simulators (AirSim, Flightmare, gym-pybullet-drones, etc.) and
summarized in Dimmig et al., "Survey of Simulators for Aerial Robots,"
2023 (arXiv:2311.02296), Eq. 1:

    p_dot = v
    m*v_dot = m*g + R(q) @ f_body + f_aero
    q_dot = 0.5 * q (x) [0, w]
    J*w_dot = -w x (J@w) + tau_body + tau_aero

Integrated with fixed-step RK4 (not an adaptive solver) - matches the
fixed-rate control loop a real flight controller runs, and matches the
integration scheme used in the energy-consumption validation literature
(Wikariak et al. 2022) this project's battery model is checked against.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --- Quaternion helpers (scalar-first: [w, x, y, z]) -----------------------

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Body-to-world rotation matrix from a unit quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_derivative(q: np.ndarray, w_body: np.ndarray) -> np.ndarray:
    """q_dot = 0.5 * q (x) [0, w] (quaternion product, w as a pure quaternion)."""
    qw, qx, qy, qz = q
    wx, wy, wz = w_body
    return 0.5 * np.array([
        -qx * wx - qy * wy - qz * wz,
        qw * wx + qy * wz - qz * wy,
        qw * wy - qx * wz + qz * wx,
        qw * wz + qx * wy - qy * wx,
    ])


def quat_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX intrinsic Euler -> quaternion, for setting up initial/desired attitudes."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Quaternion -> (roll, pitch, yaw), ZYX intrinsic. For logging/control only -
    the state itself always stays in quaternion form to avoid gimbal lock."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


GRAVITY = 9.80665  # m/s^2, standard gravity


@dataclass
class RigidBodyState:
    p: np.ndarray  # (3,) position, world ENU
    q: np.ndarray  # (4,) attitude quaternion [w,x,y,z]
    v: np.ndarray  # (3,) velocity, world frame
    w: np.ndarray  # (3,) body angular rate

    @staticmethod
    def hover_at(x: float, y: float, z: float) -> "RigidBodyState":
        return RigidBodyState(
            p=np.array([x, y, z], dtype=float),
            q=np.array([1.0, 0.0, 0.0, 0.0]),
            v=np.zeros(3),
            w=np.zeros(3),
        )

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.p, self.q, self.v, self.w])

    @staticmethod
    def from_vector(vec: np.ndarray) -> "RigidBodyState":
        return RigidBodyState(p=vec[0:3], q=vec[3:7], v=vec[7:10], w=vec[10:13])


@dataclass
class RigidBodyParams:
    mass: float                       # kg
    inertia: np.ndarray                # (3,3) body-frame inertia tensor, kg*m^2
    inertia_inv: np.ndarray = field(init=False)
    linear_drag_coeff: float = 0.0     # f_aero = -drag * v (world frame), simple linear model
    angular_drag_coeff: float = 0.0    # tau_aero = -drag * w

    def __post_init__(self):
        self.inertia_inv = np.linalg.inv(self.inertia)


def state_derivative(state: RigidBodyState, params: RigidBodyParams,
                      f_body: np.ndarray, tau_body: np.ndarray) -> np.ndarray:
    """Right-hand side of the Newton-Euler ODE, returned as a flat (13,) vector
    matching RigidBodyState.as_vector()'s layout."""
    R = quat_to_rotmat(state.q)

    f_aero_world = -params.linear_drag_coeff * state.v
    p_dot = state.v
    v_dot = (np.array([0, 0, -GRAVITY]) * params.mass
             + R @ f_body + f_aero_world) / params.mass

    q_dot = quat_derivative(state.q, state.w)

    tau_aero_body = -params.angular_drag_coeff * state.w
    w_dot = params.inertia_inv @ (
        -np.cross(state.w, params.inertia @ state.w) + tau_body + tau_aero_body
    )

    return np.concatenate([p_dot, q_dot, v_dot, w_dot])


def rk4_step(state: RigidBodyState, params: RigidBodyParams,
             f_body: np.ndarray, tau_body: np.ndarray, dt: float) -> RigidBodyState:
    """Single fixed-step RK4 integration step. f_body/tau_body are held
    constant over the step (standard zero-order-hold assumption matching
    a fixed-rate flight controller's control update)."""
    x0 = state.as_vector()

    def f(vec):
        s = RigidBodyState.from_vector(vec)
        return state_derivative(s, params, f_body, tau_body)

    k1 = f(x0)
    k2 = f(x0 + 0.5 * dt * k1)
    k3 = f(x0 + 0.5 * dt * k2)
    k4 = f(x0 + dt * k3)
    x1 = x0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    new_state = RigidBodyState.from_vector(x1)
    new_state.q = quat_normalize(new_state.q)  # correct numerical drift each step
    return new_state
