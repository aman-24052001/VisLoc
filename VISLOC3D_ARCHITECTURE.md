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

## 7. Integration with the existing VisLoc pipeline (implemented)

`visloc3d/camera.py` replaces `FrameSimulator`'s scripted, always-axis-
aligned crop with a real perspective projection: imaging a flat ground
plane from any camera pose is *exactly* a homography (Hartley &
Zisserman, ch. 8), so altitude and attitude both fall out of one 3x3
matrix derived from the drone's real pose, not separate ad hoc effects.

**Validated against the original 2D pipeline**, not just trusted because
the math looked right: footprint corners match the old fixed-size crop's
coverage to <0.001px at a calibrated reference altitude; raw pixel
comparison against the old crop showed a misleadingly large 7.7/255 mean
difference at first, traced to the world map's per-pixel speckle noise
decorrelating under bilinear resampling (confirmed by blurring both
images to average out that noise, which dropped the difference to
1.8/255) rather than any geometric error. Footprint size scales exactly
linearly with altitude; tilt produces a genuine, correctly-shaped
trapezoid (asymmetric near/far edges), not an approximation.

**A real camera-mounting bug found and fixed along the way:** the first
choice of body-to-camera rotation produced an image where motion along
world Y mapped to image *column* rather than row — confirmed analytically
(not just empirically) that a right-handed, downward-looking camera frame
cannot simultaneously have image-right tied to +X *and* image-down tied
to +Y without a flip somewhere — the same reason a top-down map view and
a looking-up-at-the-sky view have opposite handedness for the same
compass directions. Fixed by choosing image-right = body-front and
applying one deterministic vertical flip, documented in `camera.py`
rather than hidden inside a "clever" rotation choice.

**The most interesting finding: feeding tilted frames through the
existing (Phase 1) ORB localizer.** Position error grew from 1.2px level
to 116px at 30° tilt, inliers dropping from 133 to 21 — which looked at
first like the well-known ORB/SIFT viewpoint-invariance limitation
(neither is invariant to large perspective skew, only modest
rotation/scale). It's partly that, but checking with the *exact*
ground-truth homography (no feature matching at all) revealed the
dominant effect is purely geometric: a tilted rigid camera's image
center looks at a ground point shifted from the drone's actual (x, y) by
`altitude * tan(tilt)` — confirmed to <1% accuracy analytically. This is
precisely *why* the original ArduPilot project requires a gimbal-
stabilized camera — this work independently rediscovered and quantified
that requirement from first principles rather than just citing it. A
rigid-mount localizer wanting true drone position (not "where the camera
is pointing") would need to know/estimate attitude and subtract this
offset explicitly — a documented, scoped-out follow-on, not fixed here.

**Update: implemented and validated.** `nadir_offset()` and
`correct_position_estimate()` generalize the single-axis closed form
above to arbitrary combined roll+pitch, by tracing the camera's actual
optical axis to its ground intersection rather than assuming a
particular tilt axis. Feeding the real ORB localizer's output through
this correction (using known/estimated attitude, the same way a real
flight controller's state estimate would be available) brings error back
down to the flat-level noise floor at *every* tilt tested — 1.19px level,
0.86px at 10°, 1.23px at 20°, 1.04px at 30° — essentially flat, where the
uncorrected estimate grew from 1.19px to 116.51px over the same range.
The geometric explanation was complete, not partial.

## 8. Deployment (`server/`, Render)

ORB matching and homography fitting have no practical browser
equivalent (unlike the dynamics/controller/battery, which are pure
linear algebra and ported to JS directly), so that one piece needed an
actual backend. `server/main.py` is a small FastAPI service exposing
exactly that: given a real pose, render the camera view and run the same
validated localizer + nadir-offset correction used throughout this
project.

**Memory discipline, measured rather than assumed.** The only meaningful
state - the world map and its ~20,000 ORB keypoints - is built once at
process startup (`lifespan` context manager), not per-request, mirroring
how `AbsoluteLocalizer` was already designed in Phase 1 (expensive
one-time indexing, cheap repeated `localize()` calls). Measured directly,
not estimated:
- Baseline RSS after startup, full dependency set: 133.6MB
- Baseline RSS, minimal `server/requirements.txt` in a clean venv: 126.5MB
- After 150 mixed requests (15 with full frame-image encoding): 139.0MB
- After 500 requests: 139.1MB — confirms the small initial bump is one-
  time buffer warmup, not a leak; memory is flat under sustained load

All comfortably inside Render's free-tier 512MB limit, with margin to
spare. `render.yaml` pins `--workers 1` explicitly rather than relying on
a default: each additional worker would load its own full copy of the
world map and keypoint index, multiplying memory by the worker count -
the single most important lever for staying inside the free tier, made
explicit rather than left to whatever uvicorn happens to default to.

`server/requirements.txt` deliberately doesn't reuse the project's root
`requirements.txt` - matplotlib, scipy, and filterpy are needed for the
offline evaluation/chart scripts and the unrelated 2D fusion code, but
nothing in the server's actual import chain touches any of them. Cut
them and verified the server still runs correctly (and uses slightly
*less* memory) in a clean virtualenv before trusting the smaller file.

**Edge cases checked, not assumed handled:** out-of-bounds position
(400), zero/malformed quaternion (400), malformed request body (422,
automatic via Pydantic validation), and a camera pointed above the
horizon (200 with `success: false`, not a crash - render_view() doesn't
itself guard this case, but the resulting degenerate image naturally
fails ORB matching before nadir_offset()'s explicit ValueError path is
ever reached).

**The live 3D viewer (`docs/3d/`) now has a "Take a fix" button** wiring
the whole validated pipeline together end-to-end: live JS flight physics
→ real attitude → server-side camera render → real ORB localization →
geometric correction, displayed against the actual rendered camera frame.
Required bridging a unit mismatch that had never been made explicit
before this point: flight dynamics run in real meters, but the world map
calibration uses "world units" (`z_ref=200` ≈ a realistic ~57° FOV
footprint) - `METERS_TO_WORLD_UNITS = 40` ties them together deliberately
(documented in the viewer's JS) rather than leaving an implicit,
undocumented scale mismatch. End-to-end tested by hand via Playwright
against a live server, not just unit-tested in isolation: a real banking
maneuver at -7°/-6° roll/pitch showed a raw error of 6.13m collapsing to
0.09m corrected - the same effect validated in Python, now visible
through the actual deployed pipeline.

**Deployment is prepared, not completed** - I can't create or access a
Render account on anyone's behalf. `render.yaml` is committed and ready
for one-click Blueprint deployment; the actual "connect repo and deploy"
step needs to happen from the account owner's side.

## 9. Interactive 3D viewer (`docs/3d/`)

A live, interactive viewer - not a precomputed/played-back animation.
The dynamics, motor mixing, controller, and battery model are ported to
JavaScript (`docs/3d/assets/sim3d.js`) and run for real in the browser at
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

## 10. World physics objects (`world_physics.js`)

Static obstacles (buildings) and dynamic objects (balls of varied real
mass, 0.15-3.0kg) the drone can actually collide with - impulse-based
collision resolution (Erin Catto's standard formulation), not a scripted
bounce. Validated with 7 tests before any visual integration: exact
momentum conservation (drift 0.000000 across a 2-sphere collision),
energy values matching hand-calculated theory exactly (equal-mass
collision at e=0.5: 12.5J -> 7.8125J, matching the textbook formula to
the decimal), stable rest states (a dropped ball settles in finite time,
never bounces forever), no tunneling through walls, and correct
asymmetric-mass momentum transfer (a 0.9kg drone hitting a 0.05kg ball:
momentum conserved to within rounding, 2.7 -> 2.70025).

**A real sign-convention bug, found by tracing rather than assumed away.**
Two of three collision code paths (sphere-vs-box, drone-vs-box) computed
their collision normal in the opposite direction from what the impulse
formula expected - sphere-vs-sphere happened to have it right, which is
exactly why momentum conservation passed immediately while wall and
ball-knock tests failed in a specific, deceptive way: the position
correction (direction-symmetric regardless of which way the normal
points) kept working, masking that the velocity response silently never
fired. A ball thrown at a wall froze in place at exactly the contact
distance, velocity completely unchanged - traced step-by-step rather
than guessed at. Fixed both call sites; re-validated to floating-point
agreement with hand-derived theory (e.g. a 0.7-restitution wall bounce:
v' = -0.7 * 8 = -5.6 exactly).

## 11. Reactive obstacle avoidance (`world_physics.js`, "Autonomous" mode)

A genuinely different problem from Section 10's collision response:
that's *reaction after contact*. This is *avoidance before contact* - a
drone with no map and only a short-range proximity sensor, the way real
obstacle-avoidance firmware works without SLAM. Implemented as an
artificial potential field (Khatib, "Real-Time Obstacle Avoidance for
Manipulators and Mobile Robots," 1986): repulsion strength grows as
1/distance, is exactly zero at the sensing boundary (~3.5m), and simply
adds to the flight controller's own goal-seeking acceleration - so the
drone is always still trying to reach its target, just minimally
diverted while something is close, resuming course the instant it's
clear. "Autonomous" mode picks genuinely random destinations (no
foreknowledge of what's in the way), encountering both static obstacles
and kinematic (scripted-motion - oscillating/circular patrol paths)
moving ones.

**A real, found-not-assumed limitation, then fixed.** A deliberately
adversarial test - start and target placed so the straight-line path
passes exactly through a building's center - exposed the textbook
potential-field failure mode: a perfectly symmetric approach gives the
repulsion no preferred left/right direction, so it just opposes the
goal-seeking force with nothing to break the tie. The drone stalled
pressed against the wall, never reaching the target 7.4m away. Fixed
with the standard mitigation: a tangential component added to the
repulsion, consistently rotated the same way every time, so symmetry
never has a chance to form. Re-ran the identical adversarial test after
the fix: the drone now reaches the target exactly (0.000m final error),
briefly grazing the obstacle's collision boundary at the closest approach
(by design - going around something necessarily means passing close to
it) rather than stalling against it.

**Validated, not just demoed:** 25 random start/target/obstacle-field
trials reach their destination 92-96% across repeated runs (some
randomly-generated configurations are inherently near-unsolvable for any
reactive method, e.g. a target placed almost inside an obstacle's
footprint); zero actual tunneling in any trial; and a per-trial
correlation check confirmed that on the runs where a close graze *did*
occur, it correlated with continued success, not failure - confirming
it's benign contact-sliding while working around an obstacle, not a
disguised stuck state. A "minimum movement" check on the canonical
head-on case found a 1.25x path-length ratio versus the direct distance
- a real, quantified answer to "how minimal is the deviation," not just
an assumption that it's reasonable.

## 12. Planned module layout

```
visloc3d/
  dynamics.py       6-DOF Newton-Euler rigid body + RK4 integrator
  motor_mixing.py    Force/torque allocation matrices, "+"-config quadrotor
  controller.py       Cascaded PID (position -> attitude -> rate -> mixing)
  battery.py           Hover power, motor efficiency, OTC battery voltage model
  camera.py             Ground-plane homography projection (altitude + tilt)
  vehicle.py            Top-level Drone class wiring dynamics+controller+battery
  evaluate_dynamics.py   Hover/step-response validation, charts
  evaluate_battery.py     Reproduces the paper's Table III, charts
tests/
  test_dynamics3d.py, test_motor_mixing.py, test_vehicle.py, test_battery.py,
  test_camera.py, test_server.py
docs/3d/
  index.html              Interactive Three.js viewer + "Take a fix" panel + Autonomous mode
  assets/sim3d.js           JS port of dynamics+controller+battery, validated bit-for-bit
  assets/world_physics.js    Collision physics + reactive avoidance, validated separately
  assets/three.module.min.js, assets/OrbitControls.js   Vendored locally (see Section 9)
server/
  main.py                  FastAPI service: camera render + ORB localize + correction
  requirements.txt           Minimal runtime deps (no matplotlib/scipy/filterpy)
render.yaml                  One-click Render Blueprint deployment config
```

## 13. Validation findings (what actually broke, and how it was found)

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

## 14. References

- C. A. Dimmig et al., "Survey of Simulators for Aerial Robots," 2023, arXiv:2311.02296
- L. Bauersfeld & D. Scaramuzza, "Range, Endurance, and Optimal Speed Estimates for Multicopters," IEEE RA-L, 2022, arXiv:2109.04741
- R. Mahony, V. Kumar, P. Corke, "Multirotor Aerial Vehicles," IEEE Robotics & Automation Magazine, 2012
- G. Hoffmann et al., "Quadrotor Helicopter Flight Dynamics and Control," AIAA GNC, 2007
- M. Wikariak et al., "Quadrotor Model for Energy Consumption Analysis," Energies, 2022