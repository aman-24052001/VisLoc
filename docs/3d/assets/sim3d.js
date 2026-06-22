/*
 * VisLoc3D - JS port of visloc3d/{dynamics,motor_mixing,controller,battery}.py
 *
 * Faithful port for live, interactive simulation in the browser - same
 * approach as docs/assets/sandbox.js (the 2D UKF port): implement the
 * exact same equations, then validate against the Python reference
 * before trusting it for anything visual. See validate_sim3d.html for
 * the validation harness.
 *
 * Unlike the 2D ORB/optical-flow pipeline (which genuinely can't be
 * ported without bringing OpenCV to JS), this dynamics/control/battery
 * stack is pure linear algebra and ODE integration - fully portable,
 * so this *is* the real validated physics running live, not a
 * simplified stand-in.
 */
const VisLoc3D = (function () {
  const GRAVITY = 9.80665;

  // ---------- quaternion helpers (scalar-first [w,x,y,z]) ----------
  function quatNormalize(q) {
    const n = Math.hypot(q[0], q[1], q[2], q[3]);
    return n < 1e-12 ? [1, 0, 0, 0] : q.map(v => v / n);
  }
  function quatToRotmat(q) {
    const [w, x, y, z] = q;
    return [
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ];
  }
  function quatDerivative(q, w) {
    const [qw, qx, qy, qz] = q;
    const [wx, wy, wz] = w;
    return [
      -qx * wx - qy * wy - qz * wz,
      qw * wx + qy * wz - qz * wy,
      qw * wy - qx * wz + qz * wx,
      qw * wz + qx * wy - qy * wx,
    ].map(v => 0.5 * v);
  }
  function quatFromEuler(roll, pitch, yaw) {
    const cr = Math.cos(roll / 2), sr = Math.sin(roll / 2);
    const cp = Math.cos(pitch / 2), sp = Math.sin(pitch / 2);
    const cy = Math.cos(yaw / 2), sy = Math.sin(yaw / 2);
    return [
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
    ];
  }
  function quatToEuler(q) {
    const [w, x, y, z] = q;
    const roll = Math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
    const sinp = Math.clamp ? Math.clamp(2 * (w * y - z * x), -1, 1) : Math.max(-1, Math.min(1, 2 * (w * y - z * x)));
    const pitch = Math.asin(sinp);
    const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
    return [roll, pitch, yaw];
  }
  function vecAdd(a, b) { return a.map((v, i) => v + b[i]); }
  function vecSub(a, b) { return a.map((v, i) => v - b[i]); }
  function vecScale(a, s) { return a.map(v => v * s); }
  function cross(a, b) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  }
  function matVec(M, v) { return M.map(row => row.reduce((s, m, j) => s + m * v[j], 0)); }

  // ---------- rigid body state + RK4 ----------
  function makeState(x, y, z) {
    return { p: [x, y, z], q: [1, 0, 0, 0], v: [0, 0, 0], w: [0, 0, 0] };
  }
  function stateToVec(s) { return [...s.p, ...s.q, ...s.v, ...s.w]; }
  function vecToState(v) {
    return { p: v.slice(0, 3), q: v.slice(3, 7), v: v.slice(7, 10), w: v.slice(10, 13) };
  }

  function stateDerivative(s, params, fBody, tauBody) {
    const R = quatToRotmat(s.q);
    const fAeroWorld = vecScale(s.v, -params.linearDrag);
    const pDot = s.v;
    const Rf = matVec(R, fBody);
    const vDot = vecScale(
      vecAdd(vecAdd([0, 0, -GRAVITY * params.mass], Rf), fAeroWorld), 1 / params.mass
    );
    const qDot = quatDerivative(s.q, s.w);
    const tauAero = vecScale(s.w, -params.angularDrag);
    const Jw = [params.inertia[0] * s.w[0], params.inertia[1] * s.w[1], params.inertia[2] * s.w[2]];
    const gyroTerm = vecScale(cross(s.w, Jw), -1);
    const torqueSum = vecAdd(vecAdd(gyroTerm, tauBody), tauAero);
    const wDot = [torqueSum[0] / params.inertia[0], torqueSum[1] / params.inertia[1], torqueSum[2] / params.inertia[2]];
    return [...pDot, ...qDot, ...vDot, ...wDot];
  }

  function rk4Step(s, params, fBody, tauBody, dt) {
    const x0 = stateToVec(s);
    const f = (vec) => stateDerivative(vecToState(vec), params, fBody, tauBody);
    const k1 = f(x0);
    const k2 = f(x0.map((v, i) => v + 0.5 * dt * k1[i]));
    const k3 = f(x0.map((v, i) => v + 0.5 * dt * k2[i]));
    const k4 = f(x0.map((v, i) => v + dt * k3[i]));
    const x1 = x0.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
    const ns = vecToState(x1);
    ns.q = quatNormalize(ns.q);
    return ns;
  }

  // ---------- motor mixing (X-configuration) ----------
  class QuadrotorMixer {
    constructor(params) {
      this.p = params;
      const L = params.armLength, s = 1 / Math.sqrt(2);
      this.positions = [[s * L, -s * L], [-s * L, s * L], [s * L, s * L], [-s * L, -s * L]];
      this.spin = [-1, -1, 1, 1];
      const cf = params.cThrust, cd = params.cDrag;
      const A = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
      for (let i = 0; i < 4; i++) {
        const [px, py] = this.positions[i];
        A[0][i] = cf;
        A[1][i] = py * cf;
        A[2][i] = -px * cf;
        A[3][i] = this.spin[i] * cd;
      }
      this.A = A;
      this.Ainv = invert4(A);
    }
    mix(omegaSq) {
      const wrench = matVec(this.A, omegaSq);
      return { fBody: [0, 0, wrench[0]], tauBody: [wrench[1], wrench[2], wrench[3]] };
    }
    allocate(totalThrust, tauDesired) {
      const wrench = [totalThrust, ...tauDesired];
      let omegaSq = matVec(this.Ainv, wrench);
      const maxSq = this.p.maxMotorSpeed ** 2;
      omegaSq = omegaSq.map(v => Math.max(0, Math.min(maxSq, v)));
      return omegaSq;
    }
  }
  function invert4(M) {
    const n = 4;
    const A = M.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
      [A[col], A[piv]] = [A[piv], A[col]];
      const pv = A[col][col] || 1e-12;
      A[col] = A[col].map(v => v / pv);
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const factor = A[r][col];
        A[r] = A[r].map((v, j) => v - factor * A[col][j]);
      }
    }
    return A.map(row => row.slice(n));
  }

  // ---------- cascaded PID controller ----------
  class PID {
    constructor(kp, ki = 0, kd = 0, iLimit = Infinity) {
      this.kp = kp; this.ki = ki; this.kd = kd; this.iLimit = iLimit;
      this.integral = 0; this.prevError = null;
    }
    reset() { this.integral = 0; this.prevError = null; }
    step(error, dt) {
      this.integral = Math.max(-this.iLimit, Math.min(this.iLimit, this.integral + error * dt));
      const deriv = this.prevError === null ? 0 : (error - this.prevError) / dt;
      this.prevError = error;
      return this.kp * error + this.ki * this.integral + this.kd * deriv;
    }
  }
  function wrapAngle(a) {
    return ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
  }
  class FlightController {
    constructor(mass, gains) {
      this.mass = mass;
      this.g = Object.assign({
        posKp: 2.0, posKd: 2.5, altKp: 4.0, altKd: 3.0, attKp: 6.0,
        rateKp: 0.15, rateKi: 0.05, rateKd: 0.003, maxTilt: 0.5,
      }, gains || {});
      this.ratePid = [0, 1, 2].map(() => new PID(this.g.rateKp, this.g.rateKi, this.g.rateKd, 1.0));
    }
    reset() { this.ratePid.forEach(p => p.reset()); }
    compute(state, targetPos, targetYaw, dt, avoidAccel) {
      const [roll, pitch, yaw] = quatToEuler(state.q);
      const posErr = vecSub(targetPos, state.p);
      const velErr = vecScale(state.v, -1);
      let accDes = vecAdd(vecScale(posErr, this.g.posKp), vecScale(velErr, this.g.posKd));
      accDes[2] = this.g.altKp * posErr[2] + this.g.altKd * velErr[2];

      // Reactive obstacle avoidance simply adds to the goal-seeking
      // acceleration, the same structure as a potential-field local
      // planner layered on a goal controller: the drone is always
      // "trying" to reach its target, and nearby obstacles add a
      // perturbation that's strong up close and zero once clear -
      // producing minimum-deviation dodges rather than a separate
      // "avoid mode" that overrides navigation entirely.
      if (avoidAccel) accDes = vecAdd(accDes, avoidAccel);

      const ax = accDes[0], ay = accDes[1];
      const axBody = ax * Math.cos(yaw) + ay * Math.sin(yaw);
      const ayBody = -ax * Math.sin(yaw) + ay * Math.cos(yaw);
      const clamp = (v, lim) => Math.max(-lim, Math.min(lim, v));
      const pitchDes = clamp(axBody / GRAVITY, this.g.maxTilt);
      const rollDes = clamp(-ayBody / GRAVITY, this.g.maxTilt);

      let totalThrust = this.mass * (GRAVITY + accDes[2]);
      totalThrust = Math.max(0, totalThrust);

      const rollRateDes = this.g.attKp * (rollDes - roll);
      const pitchRateDes = this.g.attKp * (pitchDes - pitch);
      const yawErr = wrapAngle(targetYaw - yaw);
      const yawRateDes = this.g.attKp * yawErr;
      const rateDes = [rollRateDes, pitchRateDes, yawRateDes];
      const rateErr = vecSub(rateDes, state.w);
      const tau = [0, 1, 2].map(i => this.ratePid[i].step(rateErr[i], dt));
      return { thrust: totalThrust, tau };
    }
  }

  // ---------- battery model (Bauersfeld & Scaramuzza 2022) ----------
  const AIR_DENSITY = 1.225;
  const PROP_FIGURE_OF_MERIT = 0.55;
  const MOTOR_EFFICIENCY = 0.75;
  const D = [0.9876, -0.0020, -5.2484e-5, 1.2230e-7];
  const NOMINAL_CELL_VOLTAGE = 3.7;
  const A_COEF = [4.2, -0.1102178, 0.0103368, -4.3778e-4];

  function hoverMechPower(spec, etaP = PROP_FIGURE_OF_MERIT) {
    return Math.pow(spec.mass * GRAVITY, 1.5) /
      (etaP * Math.sqrt(2 * AIR_DENSITY * Math.PI * spec.nRotors) * spec.propRadius);
  }
  function motorElecPower(pMech, etaM = MOTOR_EFFICIENCY) { return pMech / etaM; }
  function perCellPower(pMotTotal, spec) {
    const nCell = spec.nCells * spec.nParallel;
    const cCell = spec.batteryCapacityAh / spec.nParallel;
    return pMotTotal / (nCell * cCell);
  }
  function effCapRatio(pCell) {
    return D[0] + D[1] * pCell + D[2] * pCell ** 2 + D[3] * pCell ** 3;
  }
  class Battery {
    constructor(spec) {
      this.spec = spec;
      this.energyConsumedWh = 0;
      this.eCellAccum = 0;
    }
    get totalCapacityWh() { return this.spec.batteryCapacityAh * this.spec.nCells * NOMINAL_CELL_VOLTAGE; }
    effectiveCapacityWh(pInst) {
      const kappa = effCapRatio(perCellPower(pInst, this.spec));
      return kappa * this.totalCapacityWh;
    }
    get openCircuitCellVoltage() {
      const e = this.eCellAccum;
      return A_COEF[0] + A_COEF[1] * e + A_COEF[2] * e ** 2 + A_COEF[3] * e ** 3;
    }
    get packVoltage() { return this.openCircuitCellVoltage * this.spec.nCells; }
    isDepleted(pInst) { return this.energyConsumedWh >= this.effectiveCapacityWh(pInst); }
    socFraction(pInst) {
      const eff = this.effectiveCapacityWh(pInst);
      return Math.max(0, 1 - this.energyConsumedWh / eff);
    }
    update(pElecW, dt) {
      this.energyConsumedWh += pElecW * dt / 3600;
      const pCell = perCellPower(pElecW, this.spec);
      this.eCellAccum += pCell * dt / 3600;
    }
  }

  // ---------- top-level Drone ----------
  class Drone {
    constructor(spec) {
      this.spec = Object.assign({
        mass: 0.9, armLength: 0.22, inertia: [0.011, 0.011, 0.021],
        cThrust: 2.5e-5, cDrag: 4.0e-7, maxMotorSpeed: 950,
        linearDrag: 0.25, angularDrag: 0.01,
        nCells: 4, nParallel: 1, batteryCapacityAh: 5.0,
      }, spec || {});

      this.batterySpec = {
        mass: this.spec.mass, nRotors: 4, propRadius: this.spec.armLength * 0.5,
        nCells: this.spec.nCells, nParallel: this.spec.nParallel,
        batteryCapacityAh: this.spec.batteryCapacityAh,
      };
      this.battery = new Battery(this.batterySpec);

      // Calibrate cDrag to the validated momentum-theory hover power -
      // see vehicle.py for why this matters (a ~2x cross-model
      // inconsistency was found and fixed there; same fix applied here).
      const omegaHover = Math.sqrt((this.spec.mass * GRAVITY) / (4 * this.spec.cThrust));
      const pMechHover = hoverMechPower(this.batterySpec);
      const cDrag = pMechHover / (4 * Math.pow(omegaHover, 3));

      this.mixer = new QuadrotorMixer({
        armLength: this.spec.armLength, cThrust: this.spec.cThrust,
        cDrag, maxMotorSpeed: this.spec.maxMotorSpeed,
      });
      this.bodyParams = {
        mass: this.spec.mass, inertia: this.spec.inertia,
        linearDrag: this.spec.linearDrag, angularDrag: this.spec.angularDrag,
      };
      this.controller = new FlightController(this.spec.mass);
      this.state = makeState(0, 0, 0);
      this.lastOmegaSq = [0, 0, 0, 0];
      this.lastElecPower = 0;
    }
    reset(x = 0, y = 0, z = 0, fullBattery = true) {
      this.state = makeState(x, y, z);
      this.controller.reset();
      this.lastOmegaSq = [0, 0, 0, 0];
      this.lastElecPower = 0;
      if (fullBattery) this.battery = new Battery(this.batterySpec);
    }
    step(targetPos, targetYaw, dt, avoidAccel) {
      const { thrust, tau } = this.controller.compute(this.state, targetPos, targetYaw, dt, avoidAccel);
      const omegaSq = this.mixer.allocate(thrust, tau);
      this.lastOmegaSq = omegaSq;
      const { fBody, tauBody } = this.mixer.mix(omegaSq);
      this.state = rk4Step(this.state, this.bodyParams, fBody, tauBody, dt);

      const omega = omegaSq.map(v => Math.sqrt(Math.max(0, v)));
      const pMechTotal = omegaSq.reduce((s, osq, i) => s + this.mixer.p.cDrag * osq * omega[i], 0);
      this.lastElecPower = pMechTotal / MOTOR_EFFICIENCY;
      this.battery.update(this.lastElecPower, dt);
      return this.state;
    }
    get batteryDepleted() { return this.battery.isDepleted(this.lastElecPower); }
  }

  return {
    GRAVITY, quatToEuler, quatFromEuler, quatToRotmat, makeState, rk4Step, stateDerivative,
    QuadrotorMixer, FlightController, Battery, Drone, hoverMechPower, motorElecPower,
  };
})();
