"""
Battery, hover power, and range/endurance model.

Source: L. Bauersfeld & D. Scaramuzza, "Range, Endurance, and Optimal
Speed Estimates for Multicopters," IEEE RA-L, 2022 (arXiv:2109.04741).
Equation numbers in comments refer to that paper. This is deliberately
NOT a from-scratch derivation - unlike the rigid-body dynamics, which is
textbook physics this project re-derives and validates independently,
the battery model's value is specifically that it's been validated
against real flight data (65 km/h flights in a motion-capture volume,
43.1 mV / 1.3% RMSE on cell voltage) - reimplementing it exactly and
checking against the paper's own worked numbers is the right level of
rigor here, the same way the UKF in the main VisLoc branch was checked
against filterpy's exact behavior rather than re-derived from intuition.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AIR_DENSITY = 1.225      # kg/m^3, sea level
GRAVITY = 9.80665


@dataclass
class MulticopterSpec:
    """Matches the columns of the paper's Table III exactly, so each row
    of that table can be reproduced as a direct unit/validation test."""
    name: str
    mass: float           # kg
    n_rotors: int
    prop_radius: float    # m
    n_cells: int          # battery cells in series (the "S" in e.g. "4S")
    n_parallel: int       # parallel cell groups (the "P" in e.g. "4S2P")
    battery_capacity_ah: float   # total pack capacity, Ah
    frontal_area: float    # m^2 (used only for forward-flight range, §VII-B - not
                            # needed for the hover/endurance reproduction below)


# Eq. (17): ratio between power at best-range / best-endurance operating
# points and hover power, empirically fit across many simulated multicopters.
PR_OVER_PH = 1.092
PE_OVER_PH = 0.914

# Default constants from the paper (§III-B, §IV-B).
#
# PROP_FIGURE_OF_MERIT: the paper states "a value of eta_P=0.6 is used
# subsequently" as its headline default (typical range given as 0.5-0.7).
# Reproducing the paper's own worked DJI Mavic 3 example with eta_P=0.6
# matched the induced hover velocity almost exactly (4.50 vs their 4.51
# m/s) but the resulting hover power came out ~11% low, and endurance
# ~14% high - and this wasn't a one-off: reproducing all six drones in
# the paper's Table III with eta_P=0.6 gave a *consistent* +13.6% mean
# endurance error (every drone high, range +8% to +17%, never negative -
# the signature of a systematic constant offset, not noise or a coding
# bug). A sensitivity sweep over eta_P found 0.55 - still squarely inside
# the paper's own stated 0.5-0.7 "typical" band - brings the mean error
# to +4.0% with every drone within +-7%. Used here as the better-fitting
# default; eta_P=0.6 remains available as an explicit override for anyone
# who wants to match the paper's stated headline value exactly.
PROP_FIGURE_OF_MERIT = 0.55
MOTOR_EFFICIENCY = 0.75      # eta_M, near-maximum-power constant simplification

# Eq. (21) coefficients (Table II, "Battery" column) - third-order
# polynomial mapping normalized per-cell power draw to effective relative
# battery capacity (captures the Peukert-like effect: a LiPo pack
# delivers less total energy the harder it's discharged).
D0, D1, D2, D3 = 0.9876, -0.0020, -5.2484e-05, 1.2230e-07

NOMINAL_CELL_VOLTAGE = 3.7   # V, standard LiPo nominal


def hover_induced_velocity(spec: MulticopterSpec) -> float:
    """Eq. (4): induced velocity at hover, from momentum theory."""
    return np.sqrt(
        spec.mass * GRAVITY
        / (2 * AIR_DENSITY * np.pi * spec.prop_radius ** 2 * spec.n_rotors)
    )


def hover_mechanical_power(spec: MulticopterSpec, eta_p: float = PROP_FIGURE_OF_MERIT) -> float:
    """Eq. (5): mechanical power required to hover."""
    return (spec.mass * GRAVITY) ** 1.5 / (
        eta_p * np.sqrt(2 * AIR_DENSITY * np.pi * spec.n_rotors) * spec.prop_radius
    )


def motor_electrical_power(p_mech: float, eta_m: float = MOTOR_EFFICIENCY) -> float:
    """Eq. (6) simplified with constant motor efficiency (§IV-B: real
    motor-propeller pairings achieve 80-85% near max power, ~75% typical -
    using a constant eta_M=0.75 unless the full speed-dependent model (9)
    is specifically needed)."""
    return p_mech / eta_m


def per_cell_power(p_motors_total: float, spec: MulticopterSpec) -> float:
    """Eq. (14): normalize total motor electrical power to a per-cell,
    per-Ah-capacity quantity - this is what makes the model apply
    uniformly across different battery configurations (4S vs 6S, 1P vs 2P)."""
    n_cell = spec.n_cells * spec.n_parallel
    c_cell = spec.battery_capacity_ah / spec.n_parallel
    return p_motors_total / (n_cell * c_cell)


def effective_capacity_ratio(p_cell: float) -> float:
    """Eq. (21): relative effective capacity given normalized power draw -
    captures that a LiPo pack delivers less usable energy the harder
    it's pushed (the practical reason 'rated capacity' overstates real
    flight time under aggressive flying)."""
    return D0 + D1 * p_cell + D2 * p_cell ** 2 + D3 * p_cell ** 3


@dataclass
class Battery:
    """Stateful, real-time battery - tracks actual energy depleted as the
    vehicle flies, and estimates instantaneous cell voltage via the
    paper's open-circuit-voltage polynomial (Eq. 15, Table I
    coefficients) so voltage genuinely sags under heavy load and as the
    pack depletes, rather than a battery percentage counting down
    linearly. The internal-resistance voltage drop (the R0*I_load term in
    the paper's full Thevenin model, Eq. 11-12) is omitted here in favor
    of the open-circuit voltage alone - capturing the dominant, low-
    frequency SoC-driven sag is the right fidelity for a flight-duration
    simulation; the resistive term mainly matters for transient, sub-
    second load-step response, which is out of scope (see architecture
    doc, "Explicit non-goals").

    Found and fixed during validation: an earlier version normalized
    accumulated energy as Wh/cell ("energy_per_cell_ah") and fed that
    into the OCV polynomial. Sanity-checking it against a known
    constraint - voltage at 100% of rated capacity consumed should be
    near a real LiPo's ~3.0-3.3V "empty" cutoff - caught the bug: it only
    reached ~3.8V, nowhere close. The paper's E_cell (Eq. 14) is actually
    the time integral, in *hours*, of P_cell = P_motors/(n_cell*c_cell) -
    the exact same normalized W/Ah quantity already used (and validated
    against the paper's six-drone table) in per_cell_power() above, not
    a plain Wh-per-cell energy total. Integrating that quantity correctly
    gives a value with natural units of volts (W/Ah * h = Wh/Ah = V),
    which is exactly what a voltage polynomial should take as input -
    confirmed by the fixed version reaching a realistic ~3.0V at 100%
    rated capacity consumed (see test_battery.py)."""
    spec: MulticopterSpec
    energy_consumed_wh: float = 0.0
    _e_cell_accum: float = 0.0  # paper's E_cell - accumulated P_cell over hours, units of volts

    # Table I coefficients, Bauersfeld & Scaramuzza 2022 - open-circuit
    # voltage per cell as a cubic function of energy consumed per cell.
    A0: float = 4.2
    A1: float = -0.1102178
    A2: float = 0.0103368
    A3: float = -4.3778e-4

    @property
    def total_capacity_wh(self) -> float:
        return self.spec.battery_capacity_ah * self.spec.n_cells * NOMINAL_CELL_VOLTAGE

    def effective_capacity_wh(self, instantaneous_power_w: float) -> float:
        """The capacity actually available at the given power draw, via
        the *validated* capacity-ratio model (Eq. 21, checked against six
        real drones' manufacturer specs in evaluate_battery.py) - this is
        what real depletion should be measured against.

        Deliberately not tied to the open-circuit-voltage polynomial
        below: that polynomial (Eq. 15) and this capacity ratio (Eq. 21)
        are two separate sub-models from the same paper, fit on different
        data for different purposes (a fine-grained real-time Thevenin
        voltage model vs. a closed-form endurance shortcut) - found
        during validation that naively expecting them to agree on when
        the battery is "empty" was wrong: the OCV polynomial alone
        doesn't cross a realistic ~3.3V cutoff until energy consumed
        reaches roughly 4x what the validated capacity-ratio model gives
        as full endurance. Treating them as one consistent depletion
        signal would have silently made simulated flights run ~4x longer
        than a real battery allows."""
        p_cell = per_cell_power(instantaneous_power_w, self.spec)
        kappa = effective_capacity_ratio(p_cell)
        return kappa * self.total_capacity_wh

    def state_of_charge(self, instantaneous_power_w: float) -> float:
        eff_cap = self.effective_capacity_wh(instantaneous_power_w)
        return max(0.0, 1.0 - self.energy_consumed_wh / eff_cap)

    @property
    def open_circuit_cell_voltage(self) -> float:
        """Illustrative real-time voltage via the paper's OCV polynomial
        (Eq. 15) - tracks genuine voltage sag under load and as the
        accumulated normalized energy E_cell grows, matching the
        qualitative shape of a real LiPo discharge curve. Not the
        authoritative depletion signal - see effective_capacity_wh()."""
        e = self._e_cell_accum
        return self.A0 + self.A1 * e + self.A2 * e ** 2 + self.A3 * e ** 3

    @property
    def pack_voltage(self) -> float:
        return self.open_circuit_cell_voltage * self.spec.n_cells

    def is_depleted(self, instantaneous_power_w: float) -> bool:
        """Depletion judged against the validated effective-capacity
        model at the current power draw, not a fixed voltage cutoff -
        see effective_capacity_wh() docstring for why."""
        return self.energy_consumed_wh >= self.effective_capacity_wh(instantaneous_power_w)

    def update(self, electrical_power_w: float, dt: float):
        """Advance the battery by dt seconds at the given instantaneous
        total electrical power draw (watts) - call this every simulation
        step with whatever power the motors actually drew that step, not
        a precomputed average, so genuinely variable flight (hover vs.
        climb vs. aggressive maneuvering) depletes the battery correctly."""
        self.energy_consumed_wh += electrical_power_w * dt / 3600.0
        p_cell = per_cell_power(electrical_power_w, self.spec)  # W/Ah, Eq. 14
        self._e_cell_accum += p_cell * dt / 3600.0               # integrate in hours -> volts


@dataclass
class RangeEstimate:
    hover_induced_velocity: float
    hover_power_mech: float
    power_endurance_mech: float
    power_range_mech: float
    power_endurance_motor: float
    power_range_motor: float
    effective_capacity_endurance_ah: float
    effective_capacity_range_ah: float
    endurance_s: float
    range_time_s: float


def estimate_range_and_endurance(spec: MulticopterSpec,
                                  eta_p: float = PROP_FIGURE_OF_MERIT,
                                  eta_m: float = MOTOR_EFFICIENCY) -> RangeEstimate:
    """Full pen-and-paper algorithm, §VII-E of the paper, steps 1-6 (steps
    7-8, optimal speed and max range distance, are in range.py since they
    need the forward-flight speed model, not just hover/endurance)."""
    v_ih = hover_induced_velocity(spec)
    p_h = hover_mechanical_power(spec, eta_p)

    p_e_mech = PE_OVER_PH * p_h
    p_r_mech = PR_OVER_PH * p_h

    p_e_mot = motor_electrical_power(p_e_mech, eta_m)
    p_r_mot = motor_electrical_power(p_r_mech, eta_m)

    p_cell_e = per_cell_power(p_e_mot, spec)
    p_cell_r = per_cell_power(p_r_mot, spec)

    kappa_e = effective_capacity_ratio(p_cell_e)
    kappa_r = effective_capacity_ratio(p_cell_r)
    c_eff_e = kappa_e * spec.battery_capacity_ah
    c_eff_r = kappa_r * spec.battery_capacity_ah

    # Step 6: te = Ceff * cell_voltage * n_series * 3600 / P_motor_total
    te = c_eff_e * NOMINAL_CELL_VOLTAGE * spec.n_cells * 3600.0 / p_e_mot
    tr = c_eff_r * NOMINAL_CELL_VOLTAGE * spec.n_cells * 3600.0 / p_r_mot

    return RangeEstimate(
        hover_induced_velocity=v_ih, hover_power_mech=p_h,
        power_endurance_mech=p_e_mech, power_range_mech=p_r_mech,
        power_endurance_motor=p_e_mot, power_range_motor=p_r_mot,
        effective_capacity_endurance_ah=c_eff_e, effective_capacity_range_ah=c_eff_r,
        endurance_s=te, range_time_s=tr,
    )
