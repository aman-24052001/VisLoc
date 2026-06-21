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
from visloc3d.battery import MulticopterSpec, Battery, MOTOR_EFFICIENCY


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
    # Battery spec (feeds visloc3d.battery.MulticopterSpec directly)
    n_cells: int = 4
    n_parallel: int = 1
    battery_capacity_ah: float = 5.0
    frontal_area: float = 0.0215


class Drone:
    def __init__(self, spec: DroneSpec | None = None, gains: ControllerGains | None = None,
                 calibrate_drag_to_battery_model: bool = True):
        self.spec = spec or DroneSpec()
        self.body_params = RigidBodyParams(
            mass=self.spec.mass,
            inertia=np.diag([self.spec.inertia_xx, self.spec.inertia_yy, self.spec.inertia_zz]),
            linear_drag_coeff=self.spec.linear_drag_coeff,
            angular_drag_coeff=self.spec.angular_drag_coeff,
        )

        self.battery_spec = MulticopterSpec(
            name="drone", mass=self.spec.mass, n_rotors=4,
            prop_radius=self.spec.arm_length * 0.5,  # rough default link to airframe size
            n_cells=self.spec.n_cells, n_parallel=self.spec.n_parallel,
            battery_capacity_ah=self.spec.battery_capacity_ah,
            frontal_area=self.spec.frontal_area,
        )
        self.battery = Battery(self.battery_spec)

        c_drag = self.spec.c_drag
        if calibrate_drag_to_battery_model:
            # Found during validation: c_thrust/c_drag as independent
            # arbitrary defaults gave a hover electrical power of ~56W via
            # the real motor-torque model (Q*Omega = c_drag*Omega^3) vs.
            # ~104W via the validated momentum-theory model
            # (hover_mechanical_power) for the *same* mass/size drone -
            # nearly 2x disagreement between two supposedly-physical
            # models of the same quantity. Deriving c_drag from the
            # validated power model instead of leaving it as a free
            # constant ties the two together: at hover, the real
            # motor-mixing power now matches what the manufacturer-spec-
            # validated battery model expects, so flight-time simulations
            # are actually consistent with the validated range numbers.
            from visloc3d.battery import hover_mechanical_power
            omega_hover = np.sqrt((self.spec.mass * 9.80665) / (4 * self.spec.c_thrust))
            p_mech_hover = hover_mechanical_power(self.battery_spec)
            c_drag = p_mech_hover / (4 * omega_hover ** 3)

        self.mixer = QuadrotorMixer(QuadrotorParams(
            arm_length=self.spec.arm_length, c_thrust=self.spec.c_thrust,
            c_drag=c_drag, max_motor_speed=self.spec.max_motor_speed,
        ))
        self.controller = FlightController(self.spec.mass, gains)
        self.state = RigidBodyState.hover_at(0, 0, 0)
        self.last_omega_sq = np.zeros(4)
        self.last_electrical_power_w = 0.0

    def reset(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, full_battery: bool = True):
        self.state = RigidBodyState.hover_at(x, y, z)
        self.controller.reset()
        self.last_omega_sq = np.zeros(4)
        self.last_electrical_power_w = 0.0
        if full_battery:
            self.battery = Battery(self.battery_spec)

    def _electrical_power_draw(self, omega_sq: np.ndarray) -> float:
        """Real instantaneous electrical power from the *actual* commanded
        motor speeds this step (not the hover-power formula, which only
        applies at equilibrium) - mechanical power per motor is drag
        torque times angular speed (Q*Omega = c_drag*Omega^3), matching
        the motor model in battery.py's source paper (their Eq. 6 with a
        constant-efficiency simplification, same eta_M used throughout
        this module rather than re-deriving a second motor model)."""
        omega = np.sqrt(np.maximum(omega_sq, 0.0))
        p_mech_total = float(np.sum(self.mixer.p.c_drag * omega_sq * omega))
        return p_mech_total / MOTOR_EFFICIENCY

    def step(self, target_pos: np.ndarray, target_yaw: float, dt: float) -> RigidBodyState:
        thrust, tau = self.controller.compute(self.state, target_pos, target_yaw, dt)
        omega_sq = self.mixer.allocate(thrust, tau)
        self.last_omega_sq = omega_sq
        f_body, tau_body = self.mixer.mix(omega_sq)
        self.state = rk4_step(self.state, self.body_params, f_body, tau_body, dt)

        self.last_electrical_power_w = self._electrical_power_draw(omega_sq)
        self.battery.update(self.last_electrical_power_w, dt)
        return self.state

    @property
    def battery_depleted(self) -> bool:
        return self.battery.is_depleted(self.last_electrical_power_w)

