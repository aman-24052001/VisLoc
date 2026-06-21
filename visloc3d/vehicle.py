"""
Top-level Drone: wires dynamics.py + motor_mixing.py + controller.py into
a steppable simulation, matching the same "build the pieces, validate
each independently, then wire them together last" approach used for the
2D VisLoc pipeline (world -> simulator -> localizer -> odometry -> fusion,
each validated before being connected to the next).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visloc3d.dynamics import RigidBodyState, RigidBodyParams, rk4_step
from visloc3d.motor_mixing import QuadrotorMixer, QuadrotorParams
from visloc3d.controller import FlightController, ControllerGains


@dataclass
class DroneSpec:
    """Physical specification of the vehicle - deliberately named/shaped
    like a real drone's spec sheet, since this is exactly what feeds the
    battery/range model later."""
    mass: float = 0.9                 # kg (DJI Mavic-3-ish order of magnitude)
    arm_length: float = 0.22          # m
    inertia_xx: float = 0.011
    inertia_yy: float = 0.011
    inertia_zz: float = 0.021
    c_thrust: float = 2.5e-5
    c_drag: float = 4.0e-7
    max_motor_speed: float = 950.0    # rad/s
    linear_drag_coeff: float = 0.25
    angular_drag_coeff: float = 0.01


class Drone:
    def __init__(self, spec: DroneSpec | None = None, gains: ControllerGains | None = None):
        self.spec = spec or DroneSpec()
        self.body_params = RigidBodyParams(
            mass=self.spec.mass,
            inertia=np.diag([self.spec.inertia_xx, self.spec.inertia_yy, self.spec.inertia_zz]),
            linear_drag_coeff=self.spec.linear_drag_coeff,
            angular_drag_coeff=self.spec.angular_drag_coeff,
        )
        self.mixer = QuadrotorMixer(QuadrotorParams(
            arm_length=self.spec.arm_length, c_thrust=self.spec.c_thrust,
            c_drag=self.spec.c_drag, max_motor_speed=self.spec.max_motor_speed,
        ))
        self.controller = FlightController(self.spec.mass, gains)
        self.state = RigidBodyState.hover_at(0, 0, 0)
        self.last_omega_sq = np.zeros(4)

    def reset(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.state = RigidBodyState.hover_at(x, y, z)
        self.controller.reset()
        self.last_omega_sq = np.zeros(4)

    def step(self, target_pos: np.ndarray, target_yaw: float, dt: float) -> RigidBodyState:
        thrust, tau = self.controller.compute(self.state, target_pos, target_yaw, dt)
        omega_sq = self.mixer.allocate(thrust, tau)
        self.last_omega_sq = omega_sq
        f_body, tau_body = self.mixer.mix(omega_sq)
        self.state = rk4_step(self.state, self.body_params, f_body, tau_body, dt)
        return self.state
