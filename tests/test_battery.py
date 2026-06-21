import numpy as np
import pytest

from visloc3d.battery import (
    MulticopterSpec, Battery, estimate_range_and_endurance,
    hover_induced_velocity, hover_mechanical_power, motor_electrical_power,
    per_cell_power, effective_capacity_ratio,
)

MAVIC3 = MulticopterSpec("DJI Mavic 3", 0.90, 4, 0.119, 4, 1, 5.0, 0.0215)

# All six drones from Bauersfeld & Scaramuzza (2022), Table III, with the
# paper's own "pen-and-paper" reproduction (minutes) and manufacturer
# spec (minutes) transcribed directly for regression testing.
TABLE_III = [
    (MulticopterSpec("DJI Mavic 2", 0.91, 4, 0.110, 4, 1, 3.9, 0.0200), 33, 31),
    (MAVIC3, 48, 46),
    (MulticopterSpec("DJI Matrice 200", 6.14, 4, 0.216, 6, 2, 15.3, 0.1700), 23, 24),
    (MulticopterSpec("DJI Matrice 600 Pro", 15.5, 6, 0.267, 6, 6, 34.2, 0.1760), 18, 18),
    (MulticopterSpec("Parrot Anafi AI", 0.90, 4, 0.057, 4, 1, 6.8, 0.0400), 31, 32),
    (MulticopterSpec("Skydio 2", 0.78, 4, 0.085, 3, 1, 4.3, 0.0268), 26, 23),
]


def test_mavic3_hover_induced_velocity_matches_paper():
    """Paper's own worked example states v_i,h = 4.51 m/s for this exact
    drone - validates the propeller-radius-column convention (radius in
    meters directly, not diameter) and the momentum-theory formula."""
    v = hover_induced_velocity(MAVIC3)
    assert abs(v - 4.51) < 0.02


@pytest.mark.parametrize("spec,paper_min,mfr_min", TABLE_III)
def test_reproduction_within_tolerance_of_manufacturer_spec(spec, paper_min, mfr_min):
    """Regression test pinned to Table III: our reproduction (using the
    empirically-calibrated eta_P=0.55, see battery.py for the derivation)
    should land within a generous tolerance of the manufacturer spec -
    matching the same order of accuracy the paper itself claims for its
    own pen-and-paper method."""
    est = estimate_range_and_endurance(spec)
    ours_min = est.endurance_s / 60
    rel_err = abs(ours_min - mfr_min) / mfr_min
    assert rel_err < 0.25  # generous - Skydio 2 alone is ~21% off, a known outlier


def test_mean_error_across_all_six_drones_is_reasonable():
    errs = []
    for spec, paper_min, mfr_min in TABLE_III:
        est = estimate_range_and_endurance(spec)
        ours_min = est.endurance_s / 60
        errs.append(abs(ours_min - mfr_min) / mfr_min)
    mean_err = np.mean(errs)
    assert mean_err < 0.15  # matches the ~8.7% mean found during calibration


def test_per_cell_power_scales_inversely_with_capacity():
    """Same motor power draw on a bigger battery should give a smaller
    normalized per-cell power - this is the entire point of the
    normalization (Eq. 14), letting one polynomial apply across pack sizes."""
    small = MulticopterSpec("small", 1.0, 4, 0.1, 4, 1, 2.0, 0.02)
    big = MulticopterSpec("big", 1.0, 4, 0.1, 4, 1, 10.0, 0.02)
    assert per_cell_power(100.0, big) < per_cell_power(100.0, small)


def test_effective_capacity_ratio_decreases_with_higher_power():
    """The Peukert-like effect: harder discharge -> less usable capacity."""
    low = effective_capacity_ratio(2.0)
    high = effective_capacity_ratio(8.0)
    assert high < low


def test_battery_update_accumulates_energy_correctly():
    battery = Battery(MAVIC3)
    battery.update(electrical_power_w=100.0, dt=3600.0)  # 1 hour at 100W
    assert np.isclose(battery.energy_consumed_wh, 100.0)


def test_battery_voltage_starts_at_full_charge_and_decreases():
    battery = Battery(MAVIC3)
    v0 = battery.open_circuit_cell_voltage
    assert np.isclose(v0, 4.2)  # A0, fully charged, zero energy consumed
    battery.update(electrical_power_w=90.0, dt=600.0)
    v1 = battery.open_circuit_cell_voltage
    assert v1 < v0


def test_battery_depletion_matches_static_endurance_estimate():
    """The real-time depletion model (tied to the validated capacity-
    ratio model, not the separate OCV polynomial - see battery.py for why)
    should reach 'depleted' at very close to the static pen-and-paper
    endurance estimate for the same constant power draw."""
    est = estimate_range_and_endurance(MAVIC3)
    p_mot = motor_electrical_power(est.power_endurance_mech)
    battery = Battery(MAVIC3)
    dt = 1.0
    t = 0.0
    while not battery.is_depleted(p_mot) and t < est.endurance_s * 2:
        battery.update(p_mot, dt)
        t += dt
    assert abs(t - est.endurance_s) / est.endurance_s < 0.02  # within 2%


def test_battery_lasts_longer_at_lower_power_draw():
    """Sanity check on direction: flying gentler (lower power) must
    extend, not shorten, time to depletion."""
    def time_to_depletion(power_w):
        battery = Battery(MAVIC3)
        t, dt = 0.0, 1.0
        while not battery.is_depleted(power_w) and t < 20000:
            battery.update(power_w, dt)
            t += dt
        return t

    t_low = time_to_depletion(60.0)
    t_high = time_to_depletion(120.0)
    assert t_low > t_high
