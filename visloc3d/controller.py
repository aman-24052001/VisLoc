"""
Cascaded PID flight controller - the same architecture PX4 and ArduPilot
use (referenced throughout the simulator survey's "Part of Flight
Stacks" category): an outer position loop sets a desired tilt angle and
total thrust, an attitude loop converts desired tilt into a desired body
rate, and an inner rate loop converts that into the torque commands that
motor_mixing.py turns into actual motor speeds. Each loop runs on the
error from the loop above it; the rate loop is the innermost/fastest in
a real flight controller (attitude dynamics are much faster than
translational dynamics, so this ordering isn't a stylistic choice - an
inverted cascade is unstable).

Position -> tilt angle conversion uses the standard small-angle /
differential-flatness approach (the same idea behind Mellinger & Kumar's
minimum-snap trajectory tracking, cited in the simulator survey): a
desired horizontal acceleration is achieved by tilting the thrust vector,
not by any direct horizontal force actuator (quadrotors don't have one).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from visloc3d.dynamics import GRAVITY, quat_to_rotmat, quat_to_euler


@dataclass
class PID:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    integral_limit: float = float("inf")  # anti-windup: clamp the integral term itself
    _integral: float = field(default=0.0, init=False)
    _prev_error: float | None = field(default=None, init=False)

    def reset(self):
        self._integral = 0.0
        self._prev_error = None

    def step(self, error: float, dt: float) -> float:
        self._integral = np.clip(self._integral + error * dt,
                                  -self.integral_limit, self.integral_limit)
        derivative = 0.0 if self._prev_error is None else (error - self._prev_error) / dt
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative


@dataclass
class ControllerGains:
    # Position loop (outer) - error in meters -> desired acceleration in m/s^2
    pos_kp: float = 2.0
    pos_kd: float = 2.5
    # Altitude loop - error in meters -> desired vertical acceleration
    alt_kp: float = 4.0
    alt_kd: float = 3.0
    # Attitude loop - tilt error in radians -> desired body rate in rad/s
    att_kp: float = 6.0
    # Rate loop (inner, fastest) - rate error in rad/s -> torque in N*m
    rate_kp: float = 0.15
    rate_ki: float = 0.05
    rate_kd: float = 0.003
    max_tilt_rad: float = 0.5  # ~28 deg, a real safety/control limit, not just a clamp


class FlightController:
    def __init__(self, mass: float, gains: ControllerGains | None = None):
        self.mass = mass
        self.g = gains or ControllerGains()
        self.rate_pid = [PID(self.g.rate_kp, self.g.rate_ki, self.g.rate_kd,
                              integral_limit=1.0) for _ in range(3)]

    def reset(self):
        for pid in self.rate_pid:
            pid.reset()

    def compute(self, state, target_pos: np.ndarray, target_yaw: float,
                dt: float) -> tuple[float, np.ndarray]:
        """state: RigidBodyState. target_pos: (3,) desired world position.
        target_yaw: desired heading, radians.
        Returns (total_thrust, tau_body) for motor_mixing.allocate()."""
        roll, pitch, yaw = quat_to_euler(state.q)

        # --- Outer position loop: position+velocity error -> desired
        #     horizontal acceleration (world frame) ---
        pos_err = target_pos - state.p
        # Velocity target is implicitly zero (station-keeping / waypoint
        # hold) - a velocity feedforward would be added here for trajectory
        # tracking, out of scope for this validation pass.
        vel_err = -state.v
        acc_des_world = self.g.pos_kp * pos_err + self.g.pos_kd * vel_err
        acc_des_world[2] = self.g.alt_kp * pos_err[2] + self.g.alt_kd * vel_err[2]

        # --- Convert desired horizontal acceleration into a desired tilt.
        #     Standard small-angle inversion: to accelerate forward, pitch
        #     forward; to accelerate sideways (body +Y = left in FLU), roll
        #     into it. Expressed in the *current* yaw frame since tilt is a
        #     body-frame concept. ---
        ax, ay = acc_des_world[0], acc_des_world[1]
        ax_body = ax * np.cos(yaw) + ay * np.sin(yaw)
        ay_body = -ax * np.sin(yaw) + ay * np.cos(yaw)
        pitch_des = np.clip(ax_body / GRAVITY, -self.g.max_tilt_rad, self.g.max_tilt_rad)
        roll_des = np.clip(-ay_body / GRAVITY, -self.g.max_tilt_rad, self.g.max_tilt_rad)

        total_thrust = self.mass * (GRAVITY + acc_des_world[2])
        total_thrust = max(total_thrust, 0.0)  # can't pull down with a top-mounted prop

        # --- Attitude loop: tilt error -> desired body rate ---
        roll_rate_des = self.g.att_kp * (roll_des - roll)
        pitch_rate_des = self.g.att_kp * (pitch_des - pitch)
        yaw_err = _wrap_angle(target_yaw - yaw)
        yaw_rate_des = self.g.att_kp * yaw_err

        rate_des = np.array([roll_rate_des, pitch_rate_des, yaw_rate_des])

        # --- Inner rate loop: body rate error -> torque ---
        rate_err = rate_des - state.w
        tau = np.array([self.rate_pid[i].step(rate_err[i], dt) for i in range(3)])

        return total_thrust, tau


def _wrap_angle(a: float) -> float:
    """Wrap to [-pi, pi] - yaw error must take the short way around."""
    return (a + np.pi) % (2 * np.pi) - np.pi
