# VisLoc3D — 6-DOF Flight Dynamics Extension

**Branch:** `3d-flight-dynamics` (isolated from `main` — the 2D VisLoc pipeline is untouched)

## 1. What this is

The original VisLoc project simulates a camera moving along a *scripted,
flat, 2D path* at a fixed implicit altitude — the "flight" was geometry,
not physics. This extension replaces that scripted path with a real
**6-degree-of-freedom rigid-body quadrotor simulator**: actual mass,
inertia, motor thrust, a cascaded PID flight controller, and a
physically-grounded battery model that depletes in real time based on
how the drone is actually flown. The camera that feeds into VisLoc's
existing localization pipeline becomes a *consequence* of the drone's
real position and attitude, not a scripted waypoint.

## 2. Survey of existing 3D/N-D drone simulation environments

Before building anything, the landscape of already-available simulators
was surveyed (primary source: Dimmig et al., *"Survey of Simulators for
Aerial Robots: An Overview and In-Depth Systematic Comparisons,"* 2023,
arXiv:2311.02296 — compares 44 UAV simulators).

| Simulator | Physics backend | Rendering | Fit for this sandbox |
|---|---|---|---|
| AirSim (Microsoft) | NVIDIA PhysX | Unreal/Unity | No — discontinued, needs a game engine + GPU |
| Gazebo + PX4/ArduPilot SITL | ODE/Bullet/DART | OGRE | No — needs ROS + a full middleware stack; this is what the `ngps_flight` project studied earlier in this conversation uses |
| Isaac Sim / Isaac Gym (NVIDIA) | PhysX/Flex | Vulkan | No — proprietary, GPU-mandatory |
| Flightmare / FlightGoggles | Ad hoc + Unity | Unity | No — Unity dependency, no longer actively maintained |
| MuJoCo | MuJoCo | OpenGL | Partial — installable, but a general physics engine, not quadrotor-specific |
| **gym-pybullet-drones** | PyBullet | OpenGL | Yes, technically — pip-installable, MIT-licensed, real aerodynamic effects (drag, ground effect, downwash), Gymnasium API |
| **PyFlyt** | PyBullet | OpenGL | Yes, technically — modular UAV construction, actively maintained, Gymnasium + PettingZoo |
| **RotorPy** | Pure Python | Minimal | Yes — built explicitly as a teaching tool (UPenn), full 6-DOF + aerodynamics + sensors, closest in spirit to this project |

**Decision: build the dynamics from scratch rather than depend on any of
these.** Two reasons:

1. This project's entire identity (every phase of VisLoc so far) has been
   "implement it yourself, validate every number against ground truth" —
   the UKF was hand-built and validated bit-for-bit against `filterpy`
   rather than imported as a black box. The same standard applies here.
2. Practically: PyBullet (the backend for the two most-cited options
   above) is a ~90MB compiled wheel; it would work in this sandbox (PyPI
   is on the allowed egress list) but adds a heavy dependency for what is,
   underneath, a textbook set of ODEs we can integrate ourselves with
   `numpy` + `scipy.integrate`, exactly as the Quadrotor Model for Energy
   Consumption Analysis paper (Wikariak et al., 2022) does — they
   integrate the same equations with fixed-step RK4, no physics engine.

The actual **physics formalism** (state vector, force/torque allocation
matrices) is taken directly from the survey's UAV Dynamics Background
section, which is itself the standard Newton-Euler formulation used
across nearly all the simulators above (Mahony, Kumar & Corke 2012;
Hoffmann et al. 2007) — so the equations are the same ones AirSim,
Flightmare, and gym-pybullet-drones all implement; we're just not
depending on their engines to run them.

## 3. Rigid-body dynamics (6-DOF, Newton-Euler)

State vector:
```
x = (p, q, v, ω)
```
- `p` ∈ ℝ³ — position (world frame)
- `q` ∈ ℝ⁴ — attitude as a unit quaternion (avoids gimbal lock that a
  pure Euler-angle integration would hit at pitch = ±90°)
- `v` ∈ ℝ³ — velocity (world frame)
- `ω` ∈ ℝ³ — body angular rate (the direct measurement an onboard
  gyroscope provides, and the natural variable for the inner control loop)

Control input: four squared motor speeds `Ω₁..Ω₄`.

Forces and torques (per motor `i`, with thrust/torque coefficients
`c_f`, `c_τ`, rotor position `p_Ωi`, spin-axis unit vector `z_Ωi`):
```
f = Σ c_f,i Ω_i z_Ωi = F u_Ω        (total thrust vector, body frame)
τ = Σ (c_f,i p_Ωi × z_Ωi + c_τ,i z_Ωi) Ω_i = M u_Ω   (total torque, body frame)
```
`F` and `M` are the **force/torque allocation matrices** — for a
standard "+"-configuration quadrotor these reduce to the familiar motor
mixing equations (front/back motors control pitch, left/right control
roll, and CW/CCW motor-pair speed difference controls yaw via reaction
torque).

Equations of motion:
```
ṅ = v
mv̇ = mg + R(q) F u_Ω + f_a
q̇ = ½ q ∘ [0, ω]              (quaternion derivative)
J ω̇ = -ω × Jω + M u_Ω + τ_a
```
where `f_a`, `τ_a` are aerodynamic disturbances (drag ∝ v, ∝ ω — included
as a simple linear term; full blade-element-momentum aerodynamics is
explicitly out of scope, see §6).

Integration: fixed-step RK4 (same choice as the Wikariak et al. energy
paper, dt = 1ms–5ms), not an adaptive solver — matches how real flight
controllers run their state estimator/control loop at a fixed rate, and
keeps the implementation directly comparable to the discrete-time UKF
work already validated in the main VisLoc branch.

## 4. Control: cascaded PID (the standard architecture)

This mirrors the actual control architecture used by PX4 and ArduPilot
(referenced throughout the survey's "Part of Flight Stacks" simulators),
not a simplified toy controller:

```
Position/altitude error
        │
        ▼
  Position PID  ──► desired acceleration ──► desired attitude + total thrust
        │
        ▼
  Attitude PID  ──► desired body rate
        │
        ▼
  Rate PID  ──► motor torque commands
        │
        ▼
  Motor mixing  ──► four individual motor speed commands
```

Each loop runs faster than the one above it (rate loop fastest, position
loop slowest) — standard practice, and necessary for stability since the
attitude dynamics are much faster than the translational dynamics.

**Validation plan** (matching the rigor used for the UKF in the main
branch): hover equilibrium check (thrust must exactly equal weight at
zero attitude error — this is a hard physical constraint, not a tuning
choice), step-response checks on each axis (pitch/roll/yaw/altitude),
and a closed-loop waypoint-tracking test before connecting anything to
the camera/localization pipeline.

## 5. Battery and range model

Source: Bauersfeld & Scaramuzza, *"Range, Endurance, and Optimal Speed
Estimates for Multicopters,"* IEEE RA-L, 2022 (arXiv:2109.04741). This
paper is unusually well-suited to reuse directly: it's validated against
real flight data (65 km/h in a motion-capture volume), provides a
**fully worked numeric example** for a DJI Mavic 3 that this project can
replicate exactly as a validation target, and gives closed-form
"pen-and-paper" equations rather than requiring the full blade-element
simulator.

**Hover power (momentum theory):**
```
v_i,h = sqrt(mg / (2ρπr²_prop N_r))         induced velocity at hover
P_h = (mg)^1.5 / (η_P sqrt(2ρπN_r) r_prop)   mechanical hover power
```
(`η_P` = propeller figure of merit, 0.5–0.7 typical, 0.6 used as default)

**Motor electrical power:**
```
P_mot = P_mech / η_M     (η_M ≈ 0.75 constant-efficiency simplification,
                           or the full speed-dependent model if needed)
```

**Battery — two model fidelities, matching the paper's own two-tier approach:**
1. *Simplified ("pen-and-paper") range/endurance estimate* — closed-form,
   uses only mass, battery capacity, propeller size, and frontal area.
   This is the first implementation target since it's directly
   checkable against the paper's worked DJI Mavic 3 example
   (predicted: 48 min endurance / 32.1 km range, vs. manufacturer spec
   46 min / 30 km — within the paper's own quoted <10% error band).
2. *Full Thevenin-equivalent-circuit (OTC) voltage model* — tracks actual
   cell voltage sag under load in real time, not just a depleting
   percentage. The paper provides fitted numeric coefficients (their
   Table I) for the one-time-constant model:
   ```
   U_bat = ½(U₀ - U_cap - sqrt((U₀-U_cap)² - 4R₀P_cell))
   U₀(E_cell) = a₀ + a₁E_cell + a₂E²_cell + a₃E³_cell
   R₀ = max(b₀ + b₁P̄_cell + b₂C_cell, R_min)
   ```
   This is what makes "real-life battery" actually real: voltage sags
   under heavy throttle exactly like a real LiPo pack, not a linear
   percentage countdown.

**Validation target:** reproduce the paper's Table III row-for-row
(DJI Mavic 2, Mavic 3, Matrice 200, Matrice 600 Pro, Parrot Anafi AI,
Skydio 2) — all six are within 10% of manufacturer-spec endurance/range
in the source paper, and our reimplementation should land in the same
band, not just "some plausible number."

## 6. Explicit non-goals (scope discipline, matching how VisLoc itself was scoped)

- **No blade-element-momentum (BEM) aerodynamics.** The full BEM model
  (used by the source paper for its highest-fidelity validation) requires
  per-blade-element integration and oblique-inflow corrections — disk-level
  momentum theory (used for the hover/range equations above) is the right
  fidelity level for this project, the same way Phase 1 of VisLoc used
  classical ORB rather than a learned feature matcher.
- **No wind/turbulence model in v1** — the paper's wind-correction factors
  (§VII-C) are a clearly-scoped v2 addition, not required for the core
  validation target.
- **No photorealistic rendering.** Camera frames for the localization
  pipeline remain procedurally generated (consistent with the main
  branch's synthetic world), now sampled via a real 3D projection from
  the drone's actual altitude/attitude instead of a fixed-size crop.

## 7. Integration point with the existing VisLoc pipeline

The existing `visloc/simulator.py` (`FrameSimulator`) currently extracts
a fixed-size, axis-aligned crop from the world map at each scripted
waypoint. In this extension, the crop becomes a real perspective
projection: crop size and position depend on the drone's actual altitude
(higher = larger ground footprint, matching a real camera's field of
view) and attitude (roll/pitch tilt the projected footprint into a
trapezoid, not a square — corrected or left as a modeling limitation to
be explicit about, matching how earlier phases documented known
limitations rather than hiding them).

This is deliberately the *last* integration step, after the dynamics,
controller, and battery model are independently validated — same
incremental, validate-before-connecting approach used for every prior
phase of VisLoc.

## 8. Interactive 3D viewer (`docs3d/`)

A live, interactive viewer - not a precomputed/played-back animation.
The dynamics, motor mixing, controller, and battery model are ported to
JavaScript (`docs3d/assets/sim3d.js`) and run for real in the browser at
each render frame, the same way the 2D VisLoc project's UKF was ported
to JS for its parameter sandbox. Validated the same way: ran an identical
scenario (waypoint (10,5,5), yaw 0.3 rad, 15s) through both the Python
and JS implementations and compared every state variable - position,
velocity, attitude quaternion, angular rate, individual motor speeds,
electrical power, and accumulated battery energy all matched to
floating-point precision. Separately confirmed the longer-horizon
hover-to-battery-depletion time also matches exactly (2496.05s both).

Rendering uses Three.js (vendored locally via `assets/three.module.min.js`
+ `assets/OrbitControls.js` rather than a CDN - this sandbox's egress
restrictions block CDN fetches even from a launched browser, confirmed by
testing directly, so vendoring was the only way to actually verify the
viewer renders correctly before describing it as working). The physics
state (ENU, Z-up) is mapped to Three.js's Y-up convention via a proper
(determinant +1) change of basis, applied consistently to both position
and attitude quaternion - not just position - so the model's tilt during
maneuvers is geometrically correct, not just its location.

Interaction: drag-to-orbit / scroll-to-zoom / right-drag-to-pan camera,
live target sliders (X/Y/Z/yaw) that feed directly into the real
controller every frame, maneuver presets, play/pause, and a reset that
restores both the vehicle state and the target controls. Telemetry
(position, speed, attitude, motor power, battery %, pack voltage) reads
directly from the same simulation object driving the render.

## 9. Planned module layout

```
visloc3d/
  dynamics.py       6-DOF Newton-Euler rigid body + RK4 integrator
  motor_mixing.py    Force/torque allocation matrices, "+"-config quadrotor
  controller.py       Cascaded PID (position -> attitude -> rate -> mixing)
  battery.py           Hover power, motor efficiency, OTC battery voltage model
  vehicle.py            Top-level Drone class wiring dynamics+controller+battery
  evaluate_dynamics.py   Hover/step-response validation, charts
  evaluate_battery.py     Reproduces the paper's Table III, charts
tests/
  test_dynamics3d.py, test_motor_mixing.py, test_vehicle.py, test_battery.py
docs3d/
  index.html              Interactive Three.js viewer
  assets/sim3d.js           JS port of dynamics+controller+battery, validated bit-for-bit
  assets/three.module.min.js, assets/OrbitControls.js   Vendored locally (see Section 9)
```

## 10. Validation findings (what actually broke, and how it was found)

Consistent with how every phase of the main VisLoc pipeline was built,
nothing here was trusted just because the math looked right - each
piece was checked against an external reference, and several real bugs
were found and fixed in the process:

1. **`η_P` calibration.** The paper's stated default (0.6) reproduced
   the worked DJI Mavic 3 example's induced velocity almost exactly
   (4.50 vs. 4.51 m/s) but came out ~11% low on hover power - and this
   wasn't a one-off: reproducing all six drones in the paper's Table III
   showed a *consistent* +13.6% mean endurance error, every drone high.
   A sensitivity sweep found η_P=0.55 (still inside the paper's own
   stated 0.5-0.7 "typical" range) brings mean error to +4.0%, every
   drone within ±7%. See `battery.py`'s `PROP_FIGURE_OF_MERIT` for the
   full derivation.

2. **Propeller-radius column misread.** First reproduction attempt across
   all six Table III drones gave wildly inconsistent errors (some 2x too
   low, one too high) - traced to having halved some drones' propeller
   values and not others, instead of consistently using each table value
   directly as the radius (the convention confirmed correct by the Mavic
   3 worked example). Fixing the units consistently turned scattered,
   unexplainable errors into a clean, uniform +4-21% pattern - the
   signature of an actual calibration question rather than a coding bug.

3. **Two incompatible "battery empty" signals.** An early version of the
   real-time `Battery` class fed accumulated energy into the paper's
   open-circuit-voltage polynomial (Eq. 15) and expected it to cross a
   realistic ~3.3V cutoff at roughly the validated endurance time. It
   didn't - the polynomial needed ~4x that much accumulated energy
   before nearing 3.3V. Root cause: Eq. 15 (voltage) and Eq. 21
   (capacity ratio, the one validated against manufacturer specs) are
   two separate sub-models in the source paper, fit on different data for
   different purposes, not designed to agree on a depletion threshold.
   Fixed by tying actual depletion to the validated capacity-ratio model
   and keeping the OCV polynomial as an illustrative voltage readout only
   - documented explicitly in `Battery.effective_capacity_wh()`.

4. **Motor-mixing/battery-model power mismatch.** Wiring the validated
   battery model to the validated rigid-body dynamics for the first time
   immediately surfaced a cross-model inconsistency: real instantaneous
   electrical power computed from actual motor torque (`Q·Ω` from the
   commanded motor speeds) came out ~56W at hover, while the
   independently-validated momentum-theory model said hover should cost
   ~104W for the same drone - nearly 2x apart, because `c_thrust`/`c_drag`
   had been chosen as plausible-looking but otherwise arbitrary
   constants, never checked against the battery model. Fixed by deriving
   `c_drag` from the validated hover-power model instead of leaving it
   free (see `Drone.__init__`, `calibrate_drag_to_battery_model`) - after
   the fix, the two independently-built models agree on hover power
   exactly (ratio = 1.0000000000000002, i.e. to floating-point precision).

5. **A documented, not hidden, scope limitation.** Running a full
   hover-to-depletion simulation shows the drone's position holding
   *exactly* at the target the entire time, even as the battery
   approaches empty - because this model doesn't (yet) couple battery
   voltage sag to available thrust. A real drone's motors can't
   maintain commanded RPM as pack voltage drops near end-of-discharge,
   and would eventually start to sink. This is an explicit, scoped-out
   v2 item (the same way wind/turbulence is), not a silent gap - noted
   here so it isn't mistaken for a validated capability.

## 11. References

- C. A. Dimmig et al., "Survey of Simulators for Aerial Robots," 2023, arXiv:2311.02296
- L. Bauersfeld & D. Scaramuzza, "Range, Endurance, and Optimal Speed Estimates for Multicopters," IEEE RA-L, 2022, arXiv:2109.04741
- R. Mahony, V. Kumar, P. Corke, "Multirotor Aerial Vehicles," IEEE Robotics & Automation Magazine, 2012
- G. Hoffmann et al., "Quadrotor Helicopter Flight Dynamics and Control," AIAA GNC, 2007
- M. Wikariak et al., "Quadrotor Model for Energy Consumption Analysis," Energies, 2022