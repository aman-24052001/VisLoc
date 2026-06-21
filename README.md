# VisLoc

**Vision-based GPS-denied localization, simulated end-to-end.**

Drones lose GPS — jammed, spoofed, or just unavailable. VisLoc demonstrates the
core idea behind GPS-denied visual navigation: match a downward-facing camera
against a known reference map for an absolute (but slow) position fix, track
frame-to-frame motion for a fast (but drifting) relative estimate, and fuse
the two with a Kalman/UKF filter to get a stable position with no GPS involved.

Fully simulated — no drone, no GPU, no paid satellite imagery required.

Inspired by [`ngps_flight`](https://github.com/snktshrma/ngps_flight) /
[`ap_nongps`](https://github.com/snktshrma/ap_nongps) (Sanket Sharma, ArduPilot
GSoC 2024 and follow-on work). This is an independent, simulation-only
reproduction for learning/portfolio purposes — not a fork of that codebase.

## Status: Phase 1 of 6

- [x] **Phase 1** — Synthetic world generator + frame simulator + ORB-based absolute localizer
- [ ] Phase 2 — Optical flow odometry + drift-only baseline
- [ ] Phase 3 — UKF fusion
- [ ] Phase 4 — Dashboard (live simulation view)
- [ ] Phase 5 — Parameter sandbox
- [ ] Phase 6 — Deploy + docs

See [`PRD.md`](PRD.md) for the full design doc.

## What's working right now

- `visloc/world.py` — generates a deterministic, feature-rich synthetic
  aerial-style reference map (stands in for a real satellite tile; swap in
  a real aerial photo later with no code changes elsewhere)
- `visloc/simulator.py` — simulates a moving downward-facing camera along a
  configurable flight path (`loop`, `zigzag`, `straight`), with injectable
  position noise and yaw
- `visloc/localizer.py` — ORB/AKAZE feature matching + RANSAC homography to
  recover an absolute (x, y, yaw) fix for a single camera crop against the
  full reference map

Tested under injected noise (σ=3px) and yaw (±8°): **0 failures, ~4px mean
error** across sampled frames at a simulated 1-in-10 absolute-fix rate.

## Try it

```bash
pip install -r requirements.txt
python -m visloc.world        # generates assets/world.png
python -m visloc.simulator    # generates sample camera-crop frames
python -m visloc.localizer    # runs the localizer against simulated frames
pytest tests/ -v
```

## Engineering note: why a synthetic world map in v1

ORB/AKAZE matching needs real keypoint density to work — the build process
surfaced this directly: default OpenCV ORB thresholds (tuned for full-size
photographs) starved a 220×220px crop down to **5 keypoints**, and the
reference map needed roughly **20,000 features over a 2000×2000px area**
(not the default ~1,500) before matching stopped failing outright. Both are
documented inline in `localizer.py`. A real aerial photo would hit the same
density requirement — the synthetic map just makes that tunable and
reproducible while the matching pipeline is being built out.
