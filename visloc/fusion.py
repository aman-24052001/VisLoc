"""
UKF Fusion Engine.

Fuses two signals:
  - VIO (visual odometry, from visloc.odometry.RelativeOdometry): a full
    (x, y, vx, vy) observation, available every frame, but expressed in
    VIO's own local frame (starts at an arbitrary local origin) and
    drifts over time since it has no absolute reference.
  - VPS (absolute fix, from visloc.localizer.AbsoluteLocalizer): an
    absolute (x, y) observation in world coordinates, available
    infrequently (simulating a real 1-2Hz constraint), accurate but
    sparse.

State: [x, y, vx, vy] in world pixel coordinates / pixels-per-frame.
Process model: constant velocity.

Frame alignment: VIO's local frame is aligned to world coordinates once,
at bootstrap, using the offset between the first VPS fix and VIO's local
position at that same frame. This mirrors how real VIO/VPS fusion systems
reconcile an unanchored local odometry frame with absolute fixes, without
needing any rotation/scale alignment (both frames share orientation and
scale here - only a constant translational offset is unknown).

VPS fixes are applied via "soft correction" - spread over N intermediate
steps rather than snapped in instantly - and gated by a Mahalanobis
distance threshold to reject anomalous fixes outright. Mirrors the design
of the real `ap_ukf` module this project is modeled on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

from visloc.odometry import RelativeOdometry
from visloc.localizer import AbsoluteLocalizer


def _fx(state, dt):
    x, y, vx, vy = state
    return np.array([x + vx * dt, y + vy * dt, vx, vy])


def _hx_full(state):
    return state  # VIO observes the full state directly.


def _hx_pos(state):
    return state[:2]  # VPS observes position only.


@dataclass
class FusionConfig:
    dt: float = 1.0
    # Tuned via sweep on the standard ±2deg-yaw / loop-path scenario (see
    # evaluate_fusion.py). process_noise_std=2.5 was the key lever: the
    # path is a loop (continuously turning), and the original near-zero
    # value (0.5) assumed velocity barely changes between frames, causing
    # the filter's velocity estimate to structurally lag the true,
    # continuously-rotating velocity vector - a classic constant-velocity-
    # model-vs-curved-path mismatch. 2.5 lets the filter actually track
    # the turn instead of fighting it every frame.
    process_noise_std: float = 2.5
    vio_pos_noise_std: float = 40.0     # LOOSE trust in VIO's absolute position - it drifts by
                                         # design, so it shouldn't anchor the fused estimate; this
                                         # leaves room for VPS soft-correction to actually take effect
    vio_vel_noise_std: float = 1.5      # TIGHT trust in VIO's instantaneous velocity (per-frame
                                         # optical flow displacement is accurate; only its long-run
                                         # integral drifts)
    vps_noise_std: float = 4.0          # trust in a VPS fix (matches Phase 1's measured ~4px accuracy)
    vps_soft_frames: int = 10           # spread each VPS correction over N frames
    vps_chi2_threshold: float | None = 9.21  # ~99% conf., 2 DoF; None/0 disables gating


class UKFFusion:
    """4D constant-velocity UKF fusing VIO (every frame) with VPS (sparse)."""

    def __init__(self, config: FusionConfig | None = None):
        self.cfg = config or FusionConfig()
        points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2.0, kappa=0.0)
        self.ukf = UnscentedKalmanFilter(
            dim_x=4, dim_z=4, dt=self.cfg.dt, fx=_fx, hx=_hx_full, points=points
        )
        q = self.cfg.process_noise_std ** 2
        self.ukf.Q = np.diag([q, q, q, q])
        pos_var = self.cfg.vio_pos_noise_std ** 2
        vel_var = self.cfg.vio_vel_noise_std ** 2
        self.ukf.R = np.diag([pos_var, pos_var, vel_var, vel_var])

        self._bootstrapped = False
        self._soft_step_delta: np.ndarray | None = None
        self._soft_steps_left: int = 0

    @property
    def is_bootstrapped(self) -> bool:
        return self._bootstrapped

    def bootstrap(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        self.ukf.x = np.array([x, y, vx, vy], dtype=float)
        # Velocity at bootstrap is a guess (frame 0 has no prior frame to
        # derive it from) - giving it tight uncertainty here was a real
        # bug: it made the filter under-trust the first genuine VIO
        # velocity observation, so it crawled toward the true velocity
        # over many frames instead of snapping to it immediately, causing
        # several frames of position lag (predicting forward with a
        # near-zero velocity while the camera was actually moving fast).
        self.ukf.P = np.diag([
            self.cfg.vps_noise_std ** 2, self.cfg.vps_noise_std ** 2, 500.0, 500.0
        ])
        self._bootstrapped = True
        self._soft_step_delta = None
        self._soft_steps_left = 0

    def _stabilize(self):
        """Guard against P losing symmetry/positive-definiteness after many
        sequential ad-hoc updates. A flat epsilon-jitter fix was tried
        first and wasn't enough - the filter still crashed via a Cholesky
        failure at higher process_noise_std values, because accumulated
        floating-point error had pushed an eigenvalue meaningfully
        negative, not just borderline. Clipping eigenvalues directly
        guarantees positive-definiteness regardless of how aggressive the
        chosen parameters are - this matters once Phase 5 lets users pick
        arbitrary values via sliders, where this kind of instability would
        otherwise be very easy to trigger."""
        P = (self.ukf.P + self.ukf.P.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals = np.clip(eigvals, 1e-6, None)
        self.ukf.P = eigvecs @ np.diag(eigvals) @ eigvecs.T

    def step(self, vio_obs: np.ndarray | None, vps_obs: np.ndarray | None) -> np.ndarray:
        """Advance the filter by one frame; returns the fused (x, y)."""
        if not self._bootstrapped:
            raise RuntimeError("UKFFusion.bootstrap() must be called before step()")

        self.ukf.predict()
        self._stabilize()

        if vio_obs is not None:
            self.ukf.update(np.asarray(vio_obs, dtype=float), R=self.ukf.R, hx=_hx_full)
            self._stabilize()

        if vps_obs is not None:
            self._handle_vps_fix(np.asarray(vps_obs, dtype=float))

        if self._soft_steps_left > 0:
            # Target is defined relative to *wherever the filter currently
            # is* (after this frame's predict+VIO update), not a stale
            # absolute waypoint computed back when the fix arrived. The
            # earlier (buggy) version anchored waypoints to the drone's
            # position at fix-time and applied them several frames later,
            # by which point the true position had moved well past that
            # point along the path - which pulled the filter backward and
            # caused oscillation instead of convergence. Re-deriving the
            # target from the live state each step closes the original
            # gap gradually without fighting genuine motion in between.
            target = self.ukf.x[:2] + self._soft_step_delta
            self.ukf.update(target, R=np.eye(2) * self.cfg.vps_noise_std ** 2, hx=_hx_pos)
            self._stabilize()
            self._soft_steps_left -= 1

        return self.ukf.x[:2].copy()

    def _handle_vps_fix(self, vps_xy: np.ndarray):
        cur_xy = self.ukf.x[:2]

        if self.cfg.vps_chi2_threshold:
            P_xy = self.ukf.P[:2, :2]
            R = np.eye(2) * self.cfg.vps_noise_std ** 2
            S = P_xy + R
            innovation = vps_xy - cur_xy
            d2 = float(innovation @ np.linalg.inv(S) @ innovation)
            if d2 > self.cfg.vps_chi2_threshold:
                return  # Reject anomalous fix outright - no correction queued.

        n = max(1, self.cfg.vps_soft_frames)
        # New fix supersedes any in-progress correction (simplest, avoids
        # compounding overlapping corrections if fixes arrive faster than
        # the soft-correction window drains).
        self._soft_step_delta = (vps_xy - cur_xy) / n
        self._soft_steps_left = n


def run_fusion(frames, world, vps_rate: int = 10, config: FusionConfig | None = None):
    """Run the full VIO+VPS+UKF pipeline over a sequence of simulated frames.

    Returns a dict with: fused_path, vio_global_path, gt_path, vps_fix_frames.
    """
    odo = RelativeOdometry()
    loc = AbsoluteLocalizer(world, detector="orb")

    # VIO: pure relative integration from an arbitrary local origin (0, 0) -
    # this never sees the absolute world frame, matching real VIO.
    vio_local_path = odo.track_path(frames, start_xy=(0.0, 0.0))
    vio_vel = np.zeros_like(vio_local_path)
    vio_vel[1:] = vio_local_path[1:] - vio_local_path[:-1]

    gt_path = np.array([[f.gt_x, f.gt_y] for f in frames])

    ukf = UKFFusion(config)
    fused_path = np.full((len(frames), 2), np.nan)
    vps_fix_frames = []
    offset = None

    for i, f in enumerate(frames):
        vps_obs = None
        if i % vps_rate == 0:
            res = loc.localize(f.image)
            if res.success:
                vps_obs = np.array([res.x, res.y])
                vps_fix_frames.append(i)

        if not ukf.is_bootstrapped:
            if vps_obs is None:
                continue  # can't start fusing until the first VPS fix arrives
            offset = vps_obs - vio_local_path[i]
            ukf.bootstrap(vps_obs[0], vps_obs[1], vio_vel[i][0], vio_vel[i][1])
            fused_path[i] = vps_obs
            continue

        vio_obs = np.array([
            vio_local_path[i][0] + offset[0],
            vio_local_path[i][1] + offset[1],
            vio_vel[i][0],
            vio_vel[i][1],
        ])
        fused_path[i] = ukf.step(vio_obs, vps_obs)

    vio_global_path = vio_local_path + offset if offset is not None else vio_local_path
    return {
        "fused_path": fused_path,
        "vio_global_path": vio_global_path,
        "gt_path": gt_path,
        "vps_fix_frames": vps_fix_frames,
    }
