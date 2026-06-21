# VisLoc — Product Requirements Document

**Vision-based GPS-Denied Localization Simulator**

Status: Draft v1 · Inspired by: ArduPilot GSoC 2024 / `ngps_flight` (Sanket Sharma)

---

## 1. Overview

GPS-based positioning fails in jammed, spoofed, or signal-denied environments. UAVs flying at high altitude or in contested airspace need an alternative way to know "where am I" using only onboard sensors — primarily a camera.

VisLoc is a scaled-down, fully simulated reproduction of this problem: it demonstrates how a moving camera can be localized against a known reference map (visual absolute positioning) and how that signal can be fused with noisy frame-to-frame motion estimates (visual odometry) to produce a stable, drift-corrected position estimate — without needing a real drone, GPU, or paid satellite imagery.

This is an educational reproduction, not a research contribution. The goal is a self-contained, visually compelling demonstration of the core concepts involved in GPS-denied visual navigation — feature-based absolute localization, optical-flow-based relative odometry, and Kalman/UKF sensor fusion.

---

## 2. Motivation

| Reason | Detail |
|---|---|
| **Real-world relevance** | GPS denial (jamming, spoofing, signal blackout) is a genuine operational problem for UAVs and other autonomous platforms; visual localization is an active, well-studied mitigation. |
| **Educational value** | Each component — feature matching, optical flow, Kalman/UKF fusion — is independently well-understood and explainable, making the combined system a good vehicle for learning estimation theory and classical computer vision together, rather than relying on an opaque end-to-end model. |
| **Zero infrastructure cost** | No GPU, drone, or paid satellite imagery is required (see Section 10). The entire pipeline runs in simulation on commodity hardware. |
| **Quantifiable outcome** | The project produces a concrete, measurable result — % position-drift reduction from fusion vs. raw odometry — rather than a purely qualitative demo. |

---

## 3. Goals

1. Simulate a moving camera traversing a large static reference image along a synthetic flight path.
2. Estimate absolute position periodically via classical feature matching (no real GPS used, by design).
3. Estimate continuous relative motion via frame-to-frame tracking (optical flow).
4. Fuse both signals via a Kalman/Unscented Kalman Filter with soft-correction (no instantaneous position snapping).
5. Visualize and quantify: ground truth path vs. raw odometry-only path (drift) vs. fused path (corrected).
6. Package as a clean, deployable, interactive web demo with a coherent visual design.

## 4. Non-Goals

- No real drone hardware, no ArduPilot/Gazebo SITL integration.
- No real-time onboard deployment or embedded optimization.
- No transformer-based matcher (LightGlue/SuperPoint) in v1 — explicitly deferred to a v2 stretch goal.
- No multi-reference-image stitching or wide-area mapping.
- Not a claim of novel research — explicitly framed as an educational reproduction.

---

## 5. Target Audience / Use Case

- **Primary:** Engineers, students, and hobbyists learning visual odometry and sensor fusion concepts who want a hands-on, runnable reference implementation.
- **Secondary:** Anyone studying visual-inertial odometry or sensor fusion basics — the deployed demo should be self-explanatory enough to function as a teaching tool.

---

## 6. System Architecture

```
                         ┌─────────────────────────┐
   Static reference      │   Frame Simulator         │
   image (the "world") ─▶│   - crops moving window   │
                         │   - injects noise/rotation │
                         └────────────┬──────────────┘
                                      │ synthetic camera frames
                ┌─────────────────────┼─────────────────────┐
                ▼                                           ▼
   ┌─────────────────────────┐               ┌─────────────────────────┐
   │  Absolute Localizer       │               │  Relative Odometry        │
   │  (ORB/AKAZE match vs       │               │  (optical flow /          │
   │   full reference image)    │               │   frame-to-frame ORB)     │
   │  → runs at 1-2 Hz (sim)     │               │  → runs every frame       │
   └────────────┬───────────────┘               └────────────┬──────────────┘
                │  absolute (x,y), confidence                │ relative Δ(x,y)
                └─────────────────────┬───────────────────────┘
                                      ▼
                         ┌─────────────────────────┐
                         │   UKF Fusion Engine        │
                         │   - bootstrap on 1st fix    │
                         │   - soft correction (N      │
                         │     frames, no snapping)    │
                         │   - Mahalanobis gating       │
                         └────────────┬──────────────┘
                                      ▼
                         ┌─────────────────────────┐
                         │   Dashboard / Visualizer   │
                         │   - ground truth vs raw    │
                         │     vs fused path           │
                         │   - live drift metrics      │
                         │   - tunable parameters      │
                         └─────────────────────────┘
```

---

## 7. Functional Requirements

### 7.1 Frame Simulator
- Accepts one large static image as "world map" and a list of waypoints (or parametric curve) as ground-truth flight path.
- Generates a sequence of cropped frames simulating a downward-facing camera moving along that path.
- Configurable: frame rate (sim Hz), crop size (simulated altitude), injected Gaussian noise, injected rotation/yaw, optional motion blur.
- Outputs: frame sequence + ground-truth (x, y, yaw) per frame for later evaluation.

### 7.2 Absolute Localizer
- Takes a single cropped frame, matches it against the full reference image using ORB or AKAZE keypoints + descriptor matching (RANSAC-filtered homography).
- Returns estimated (x, y) in world-image coordinates + a confidence score (number of inlier matches).
- Runs at a reduced rate relative to the simulator (e.g. every 10th frame) to mimic the real system's 1-2Hz constraint vs. higher-frequency odometry.
- Rejects low-confidence matches below a configurable inlier threshold (mirrors the real system's match-quality gating).

### 7.3 Relative Odometry
- Computes frame-to-frame displacement using either:
  - Dense/sparse optical flow (Lucas-Kanade), or
  - Matched-keypoint centroid shift between consecutive frames
- Outputs a continuous relative (Δx, Δy) per frame, which will drift over time by design (no correction at this stage) — this drift is the "problem" the fusion stage solves.

### 7.4 UKF Fusion Engine
- 4D state: (x, y, vx, vy) — deliberately scoped down, mirroring the real `ap_ukf` simplification.
- Bootstraps on the first absolute fix (no arbitrary initial guess).
- Applies **soft correction**: each absolute fix is blended in over N intermediate steps rather than snapping the position instantly (configurable `soft_frames` parameter).
- Applies a Mahalanobis-distance gate to reject anomalous absolute fixes (configurable threshold, can be disabled).
- Exposes tunable process/measurement noise covariances.

### 7.5 Evaluation & Metrics
- Computes position error (Euclidean distance from ground truth) over time for two conditions:
  1. Raw odometry only (no fusion) — establishes the drift baseline
  2. Fused output — shows the correction
- Reports: cumulative drift (final-frame error), mean error, and % error reduction (fused vs raw) — the project's headline benchmark number.
- Exports a results table/CSV for reproducibility.

### 7.6 Dashboard / Visualizer
- See Section 8 (UI/UX) for full detail.

---

## 8. UI / UX Design

### 8.1 Design Language
Dark-theme, technical/data-product aesthetic: a monospace font for UI and data elements paired with a serif display font for headers, dark background, a single accent color used sparingly for emphasis and active states, minimal chrome, section navigation via anchor dots, and step-by-step animated reveals with log-style status output for pipeline stages. Avoid generic gradient-heavy "AI demo" styling.

### 8.2 Pages / Views

**1. Landing / Hero**
- One-line pitch: "GPS fails. Cameras don't have to." or similar
- Animated preview: a small looping clip of the drift-vs-fused path comparison
- Links: GitHub repo, "Run the demo" CTA

**2. Live Simulation View (main feature)**
- Split layout:
  - **Left:** the reference world-image with three overlaid path traces — ground truth (solid white), raw odometry (dashed red, visibly drifting), fused estimate (solid gold) — animated as playback progresses
  - **Right:** live-updating panel showing:
    - Current frame crop (what the "camera" currently sees)
    - Current absolute-fix confidence (only flashes when a fix occurs, mirroring the 1-2Hz rate)
    - Running error chart (line chart, raw vs fused error over time)
- **Playback controls:** play/pause, scrub bar, speed control

**3. Parameter Sandbox**
- Sliders/inputs in the same interactive-demo style as the rest of the UI:
  - Noise injection level
  - Soft-correction frame count (`N`)
  - Absolute-fix rate (Hz)
  - Mahalanobis gate threshold (on/off + value)
- Re-running with new parameters re-renders the path comparison and error chart live — a "show, don't tell" interactive centerpiece.

**4. Results / Methodology**
- Static write-up: architecture diagram (from Section 6), explanation of each module, final benchmark numbers, link back to the original ArduPilot project for attribution/inspiration.

### 8.3 Key UX Principle
The entire point of the UI is to make drift *visible* and correction *visible* — the red dashed line wandering away from the white ground-truth line, then the gold fused line staying close, is the single most important visual in the whole project. Everything else supports that one moment of clarity.

---

## 9. Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| **Core CV** | OpenCV (Python) — ORB/AKAZE, optical flow, homography/RANSAC | Mature, no GPU required, directly comparable to the real system's matching stage |
| **Fusion math** | Custom UKF implementation (NumPy), or `filterpy` as a base to extend | Writing the core filter by hand keeps the estimation logic transparent rather than hidden behind a library |
| **Simulation/backend** | Python (frame simulator, evaluation harness) | Single language across the whole pipeline, minimal context-switching |
| **API layer (if needed)** | FastAPI | Lightweight, enables a backend if the dashboard needs live recomputation rather than precomputed playback |
| **Frontend** | Vanilla JS/HTML/CSS or React | Vanilla keeps the project dependency-free; React is preferable if richer chart libraries are wanted |
| **Data** | One free aerial/satellite tile (Sentinel-2, ESRI World Imagery free tier, or any public high-res aerial photo) | No paid imagery required |
| **Deployment** | GitHub Pages (static, precomputed playback) or Render free tier (if FastAPI backend needed for live parameter sandbox) | Free-tier deployable end to end |
| **Testing** | pytest for the CV/fusion modules | Ensures core matching and fusion logic is regression-tested as the system grows |

**Recommended decision:** start with **precomputed playback + static GitHub Pages** for the Live Simulation View (cheapest, fastest to ship), and only stand up a FastAPI backend if the Parameter Sandbox needs true live recomputation rather than a few precomputed parameter presets.

---

## 10. Data Requirements

- One static reference image (the "world"), free source (Sentinel-2 / ESRI / any public aerial photo)
- Synthetic flight path — author 1-2 interesting paths yourself (e.g. a loop, a zigzag) to make the demo visually engaging
- No dataset licensing concerns if sourced from free/public-domain imagery — verify license on whichever tile source is chosen

---

## 11. Success Metrics

| Metric | Target |
|---|---|
| % drift reduction (fused vs. raw odometry) | Headline benchmark number — aim for a clearly demonstrable, non-trivial reduction (exact target depends on injected noise level chosen) |
| Demo load/playback smoothness | No visible UI lag during path animation |
| Code quality | Test coverage on core modules (matcher, fusion) |
| Deployment | Live deployed link + clean, reproducible README |

---

## 12. Milestones

| Phase | Deliverable |
|---|---|
| 1 | Frame simulator + ORB/AKAZE absolute localizer working standalone |
| 2 | Optical flow odometry + drift-only baseline (no fusion) — first "before" chart |
| 3 | UKF fusion implemented — first "after" chart, headline metric computed |
| 4 | Dashboard (Live Simulation View) built and wired to precomputed results |
| 5 | Parameter Sandbox + Results/Methodology page |
| 6 | README, deploy, final polish |

---

## 13. Risks & Open Questions

| Risk | Mitigation |
|---|---|
| ORB/AKAZE matching too unreliable on chosen reference image (low texture, repetitive patterns) | Pick a reference image with strong, varied visual features (urban/mixed terrain over uniform farmland) |
| Drift baseline too small to show a compelling correction | Tune injected noise to guarantee visible drift before fusion is applied |
| Scope creep toward full real-system parity | Explicitly time-box to the 6 milestones above; transformer-matcher (LightGlue) is v2, not v1 |

---

## 14. Out of Scope / Future Work (v2+)

- Swap ORB/AKAZE for LightGlue/SuperPoint (pretrained, no training needed) — a transformer-based feature matching upgrade path
- Multi-reference-image support (tile stitching across a larger area)
- Real video input (e.g. publicly available UAV footage) instead of purely synthetic frames

---

## 15. Attribution

Explicitly inspired by and modeled on the architecture of `ngps_flight` / `ap_nongps` (Sanket Sharma, ArduPilot GSoC 2024 and follow-on work). VisLoc is an independent, simplified, simulation-only reproduction for learning purposes — not a fork or derivative of that codebase. Credit to be stated clearly in the README.
