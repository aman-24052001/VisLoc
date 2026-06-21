"""
Quadrotor motor mixing (X-configuration, the layout used by nearly all
consumer quadrotors including the DJI models this project's battery model
is validated against).

Body frame: FLU (X=front, Y=left, Z=up). Motor layout:

         front (+X)
            |
    M2 (CCW)   M0 (CW)
        \\     /
         \\   /
          \\ /
           X  ----> this is the CoM, arms at 45 deg
          / \\
         /   \\
        /     \\
    M1 (CW)    M3 (CCW)
            |
         back (-X)

Each motor produces:
  - a thrust force c_f * Omega^2 along body +Z
  - a reaction drag torque c_tau * Omega^2 about its own spin axis (+Z if
    CCW, -Z if CW) - this is what couples motor speed differences to yaw

Each motor's thrust, applied at its arm position, also produces a
roll/pitch torque about the center of mass: tau = p_i x (0,0,f_i).

This module provides both directions:
  - mix(): motor speeds -> total thrust + 3-axis torque (used by the
    dynamics propagator)
  - allocate(): desired total thrust + 3-axis torque -> 4 motor speeds
    (used by the controller, the actual "control allocation" problem
    every real flight controller solves every control cycle)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QuadrotorParams:
    arm_length: float       # meters, CoM to motor
    c_thrust: float         # thrust coefficient: f = c_thrust * Omega^2 (N per (rad/s)^2)
    c_drag: float           # drag torque coefficient: tau = c_drag * Omega^2
    max_motor_speed: float  # rad/s, for saturation


class QuadrotorMixer:
    """X-configuration motor mixer. Motors indexed 0-3 as in the module
    docstring: 0=front-right(CW), 1=back-left(CW), 2=front-left(CCW),
    3=back-right(CCW)."""

    def __init__(self, params: QuadrotorParams):
        self.p = params
        L = params.arm_length
        s = 1.0 / np.sqrt(2.0)  # 45-degree arm projection onto X/Y

        # Motor positions in body frame (x, y), z=0 (rotor plane through CoM)
        self.positions = np.array([
            [s * L, -s * L],   # M0 front-right
            [-s * L, s * L],   # M1 back-left
            [s * L, s * L],    # M2 front-left
            [-s * L, -s * L],  # M3 back-right
        ])
        # Spin direction: +1 = CCW (reaction torque +Z), -1 = CW (reaction torque -Z)
        self.spin = np.array([-1.0, -1.0, 1.0, 1.0])  # M0,M1 CW; M2,M3 CCW

        # Build the 4x4 allocation matrix A such that
        # [total_thrust, tau_x, tau_y, tau_z]^T = A @ [Omega0^2, .., Omega3^2]^T
        cf, cd = params.c_thrust, params.c_drag
        A = np.zeros((4, 4))
        for i in range(4):
            px, py = self.positions[i]
            A[0, i] = cf                      # total thrust contribution
            A[1, i] = py * cf                 # roll torque (tau_x) from thrust at arm
            A[2, i] = -px * cf                # pitch torque (tau_y) from thrust at arm
            A[3, i] = self.spin[i] * cd        # yaw torque (tau_z) from drag reaction
        self.A = A
        self.A_inv = np.linalg.inv(A)

    def mix(self, omega_sq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Motor squared-speeds (4,) -> (f_body (3,), tau_body (3,))."""
        wrench = self.A @ omega_sq
        f_body = np.array([0.0, 0.0, wrench[0]])
        tau_body = wrench[1:4]
        return f_body, tau_body

    def allocate(self, total_thrust: float, tau_desired: np.ndarray) -> np.ndarray:
        """Desired total thrust + 3-axis torque -> 4 motor squared-speeds.
        This is the control allocation step every real flight controller
        runs every cycle. Negative results (physically impossible - a
        propeller can't produce negative thrust by spinning backwards in
        this configuration) are clipped to zero, and speeds are saturated
        at max_motor_speed^2; both are real actuator limits, not numerical
        convenience."""
        wrench = np.concatenate([[total_thrust], tau_desired])
        omega_sq = self.A_inv @ wrench
        omega_sq = np.clip(omega_sq, 0.0, self.p.max_motor_speed ** 2)
        return omega_sq
