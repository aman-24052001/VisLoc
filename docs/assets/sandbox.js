/*
 * VisLoc parameter sandbox - JS port of visloc/fusion.py's UKFFusion.
 *
 * This is a faithful, validated port (checked against the Python/filterpy
 * implementation to full numerical precision on the canonical scenario,
 * see commit history) - including a subtle filterpy behavior that isn't
 * obvious from the textbook UKF equations: when update() is called twice
 * in one step (once for the VIO observation, once for the VPS soft
 * correction), the SECOND call reuses the sigma points generated during
 * the most recent predict() rather than regenerating them from the
 * already-updated state. This is intentional (filterpy's own docs note
 * it's meant for "multiple simultaneous measurements"), and matters
 * enough numerically that this port reproduces it deliberately rather
 * than the more "obvious" approach of recomputing sigma points each time.
 *
 * Re-running the actual CV pipeline (ORB matching, optical flow) live in
 * the browser isn't feasible without porting OpenCV to JS, so VIO/VPS
 * data comes from precomputed presets (sandbox_data.json) - but the
 * fusion math itself, the part that's actually interesting to tune, runs
 * for real here.
 */
const VisLocSandbox = (function () {
  // ---- small matrix helpers (general size, used for n=4 and n=2) ----
  function zeros(n) { return Array.from({ length: n }, () => new Array(n).fill(0)); }
  function eye(n) { const M = zeros(n); for (let i = 0; i < n; i++) M[i][i] = 1; return M; }
  function matMul(A, B) {
    const n = A.length, m = B[0].length, k = B.length;
    const C = Array.from({ length: n }, () => new Array(m).fill(0));
    for (let i = 0; i < n; i++)
      for (let j = 0; j < m; j++) {
        let s = 0;
        for (let p = 0; p < k; p++) s += A[i][p] * B[p][j];
        C[i][j] = s;
      }
    return C;
  }
  function matVec(A, v) {
    return A.map(row => row.reduce((s, a, j) => s + a * v[j], 0));
  }
  function transpose(A) {
    const n = A.length, m = A[0].length;
    const T = Array.from({ length: m }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) T[j][i] = A[i][j];
    return T;
  }
  function matAdd(A, B) { return A.map((row, i) => row.map((v, j) => v + B[i][j])); }
  function matSub(A, B) { return A.map((row, i) => row.map((v, j) => v - B[i][j])); }
  function matScale(A, s) { return A.map(row => row.map(v => v * s)); }
  function outer(a, b) { return a.map(ai => b.map(bj => ai * bj)); }
  function vecAdd(a, b) { return a.map((v, i) => v + b[i]); }
  function vecSub(a, b) { return a.map((v, i) => v - b[i]); }
  function vecScale(a, s) { return a.map(v => v * s); }
  function vecDot(a, b) { return a.reduce((s, v, i) => s + v * b[i], 0); }

  // Lower-triangular Cholesky: returns L such that L @ L^T = M.
  // (scipy.linalg.cholesky used by the Python side returns upper U with
  // U^T U = M; U's row k equals L's column k, handled in sigmaPoints below.)
  function cholesky(M) {
    const n = M.length;
    const L = zeros(n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = M[i][j];
        for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
        if (i === j) L[i][j] = Math.sqrt(Math.max(sum, 1e-12));
        else L[i][j] = sum / L[j][j];
      }
    }
    return L;
  }

  // Cyclic Jacobi eigenvalue algorithm for symmetric matrices - used to
  // clip negative/near-zero eigenvalues after many sequential updates,
  // exactly mirroring the Python side's eigenvalue-clipping stabilizer
  // (a flat epsilon-jitter wasn't enough there either - see fusion.py).
  function jacobiEigenSym(Ain, maxIter = 60, tol = 1e-12) {
    const n = Ain.length;
    let A = Ain.map(row => row.slice());
    let V = eye(n);
    for (let iter = 0; iter < maxIter; iter++) {
      let p = 0, q = 1, maxVal = 0;
      for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++)
          if (Math.abs(A[i][j]) > maxVal) { maxVal = Math.abs(A[i][j]); p = i; q = j; }
      if (maxVal < tol) break;
      const app = A[p][p], aqq = A[q][q], apq = A[p][q];
      const theta = (aqq - app) / (2 * apq);
      const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let i = 0; i < n; i++) {
        const aip = A[i][p], aiq = A[i][q];
        A[i][p] = c * aip - s * aiq; A[i][q] = s * aip + c * aiq;
      }
      for (let i = 0; i < n; i++) {
        const api = A[p][i], aqi = A[q][i];
        A[p][i] = c * api - s * aqi; A[q][i] = s * api + c * aqi;
      }
      for (let i = 0; i < n; i++) {
        const vip = V[i][p], viq = V[i][q];
        V[i][p] = c * vip - s * viq; V[i][q] = s * vip + c * viq;
      }
    }
    return { values: A.map((row, i) => row[i]), vectors: V };
  }

  function stabilize(P) {
    const n = P.length;
    const sym = P.map((row, i) => row.map((v, j) => (v + P[j][i]) / 2));
    const { values, vectors } = jacobiEigenSym(sym);
    const clipped = values.map(v => Math.max(v, 1e-6));
    // Reconstruct V @ diag(clipped) @ V^T
    const VT = transpose(vectors);
    const scaledVT = VT.map((row, i) => row.map(v => v * clipped[i]));
    return matMul(vectors, scaledVT);
  }

  // ---- Unscented transform machinery (Van der Merwe scaled sigma points) ----
  class RefUKF {
    constructor(n = 4, alpha = 0.1, beta = 2.0, kappa = 0.0) {
      this.n = n;
      this.lam = alpha * alpha * (n + kappa) - n;
      const c = 0.5 / (n + this.lam);
      this.Wm = new Array(2 * n + 1).fill(c);
      this.Wc = new Array(2 * n + 1).fill(c);
      this.Wm[0] = this.lam / (n + this.lam);
      this.Wc[0] = this.lam / (n + this.lam) + (1 - alpha * alpha + beta);
    }

    sigmaPoints(x, P) {
      const n = this.n;
      const L = cholesky(matScale(P, this.lam + n));
      const sigmas = [x.slice()];
      for (let k = 0; k < n; k++) {
        const Uk = L.map(row => row[k]); // row k of U = column k of L
        sigmas.push(vecAdd(x, Uk));
      }
      for (let k = 0; k < n; k++) {
        const Uk = L.map(row => row[k]);
        sigmas.push(vecSub(x, Uk));
      }
      return sigmas;
    }

    predict(x, P, fx, dt, Q) {
      const sigmas = this.sigmaPoints(x, P);
      const sigmasF = sigmas.map(s => fx(s, dt));
      const xPred = new Array(this.n).fill(0);
      for (let i = 0; i < sigmasF.length; i++)
        for (let j = 0; j < this.n; j++) xPred[j] += this.Wm[i] * sigmasF[i][j];
      let Pacc = zeros(this.n);
      for (let i = 0; i < sigmasF.length; i++) {
        const d = vecSub(sigmasF[i], xPred);
        Pacc = matAdd(Pacc, matScale(outer(d, d), this.Wc[i]));
      }
      const Ppred = stabilize(matAdd(Pacc, Q));
      return { xPred, Ppred, sigmasF };
    }

    update(xPred, Ppred, sigmasF, hx, z, R) {
      const sigmasH = sigmasF.map(hx);
      const dimZ = sigmasH[0].length;
      const zp = new Array(dimZ).fill(0);
      for (let i = 0; i < sigmasH.length; i++)
        for (let j = 0; j < dimZ; j++) zp[j] += this.Wm[i] * sigmasH[i][j];
      let S = zeros(dimZ);
      for (let i = 0; i < sigmasH.length; i++) {
        const d = vecSub(sigmasH[i], zp);
        S = matAdd(S, matScale(outer(d, d), this.Wc[i]));
      }
      S = matAdd(S, R);
      let Pxz = Array.from({ length: this.n }, () => new Array(dimZ).fill(0));
      for (let i = 0; i < sigmasF.length; i++) {
        const dx = vecSub(sigmasF[i], xPred);
        const dz = vecSub(sigmasH[i], zp);
        Pxz = matAdd(Pxz, matScale(outer(dx, dz), this.Wc[i]));
      }
      const Sinv = invertSmall(S);
      const K = matMul(Pxz, Sinv);
      const y = vecSub(z, zp);
      const xNew = vecAdd(xPred, matVec(K, y));
      const Pnew = stabilize(matSub(Ppred, matMul(K, matMul(S, transpose(K)))));
      return { x: xNew, P: Pnew };
    }
  }

  // Small general matrix inverse via Gauss-Jordan - S is at most 4x4 here.
  function invertSmall(M) {
    const n = M.length;
    const A = M.map((row, i) => row.concat(eye(n)[i]));
    for (let col = 0; col < n; col++) {
      let pivot = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(A[r][col]) > Math.abs(A[pivot][col])) pivot = r;
      [A[col], A[pivot]] = [A[pivot], A[col]];
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

  function fx(state, dt) {
    const [x, y, vx, vy] = state;
    return [x + vx * dt, y + vy * dt, vx, vy];
  }
  function hxFull(state) { return state.slice(); }
  function hxPos(state) { return state.slice(0, 2); }

  class UKFFusion {
    constructor(cfg) {
      this.cfg = cfg;
      this.ref = new RefUKF();
      const q = cfg.processNoiseStd ** 2;
      this.Q = [[q, 0, 0, 0], [0, q, 0, 0], [0, 0, q, 0], [0, 0, 0, q]];
      const pv = cfg.vioPosNoiseStd ** 2, vv = cfg.vioVelNoiseStd ** 2;
      this.Rvio = [[pv, 0, 0, 0], [0, pv, 0, 0], [0, 0, vv, 0], [0, 0, 0, vv]];
      this.bootstrapped = false;
      this.softDelta = null;
      this.softLeft = 0;
    }
    bootstrap(x, y, vx, vy) {
      this.x = [x, y, vx, vy];
      const v0 = this.cfg.vpsNoiseStd ** 2;
      this.P = [[v0, 0, 0, 0], [0, v0, 0, 0], [0, 0, 500, 0], [0, 0, 0, 500]];
      this.bootstrapped = true;
      this.softDelta = null; this.softLeft = 0;
    }
    step(vioObs, vpsObs) {
      const { xPred, Ppred, sigmasF } = this.ref.predict(this.x, this.P, fx, 1.0, this.Q);
      this.x = xPred; this.P = Ppred;
      if (vioObs) {
        const r = this.ref.update(this.x, this.P, sigmasF, hxFull, vioObs, this.Rvio);
        this.x = r.x; this.P = r.P;
      }
      if (vpsObs) this._handleVps(vpsObs);
      if (this.softLeft > 0) {
        const target = vecAdd(this.x.slice(0, 2), this.softDelta);
        const Rpos = [[this.cfg.vpsNoiseStd ** 2, 0], [0, this.cfg.vpsNoiseStd ** 2]];
        const r = this.ref.update(this.x, this.P, sigmasF, hxPos, target, Rpos);
        this.x = r.x; this.P = r.P;
        this.softLeft -= 1;
      }
      return this.x.slice(0, 2);
    }
    _handleVps(vpsXy) {
      const cur = this.x.slice(0, 2);
      if (this.cfg.chi2Threshold) {
        const Pxy = [[this.P[0][0], this.P[0][1]], [this.P[1][0], this.P[1][1]]];
        const R = [[this.cfg.vpsNoiseStd ** 2, 0], [0, this.cfg.vpsNoiseStd ** 2]];
        const S = matAdd(Pxy, R);
        const innov = vecSub(vpsXy, cur);
        const Sinv = invertSmall(S);
        const d2 = vecDot(innov, matVec(Sinv, innov));
        if (d2 > this.cfg.chi2Threshold) return;
      }
      const n = Math.max(1, this.cfg.softFrames);
      this.softDelta = vecScale(vecSub(vpsXy, cur), 1 / n);
      this.softLeft = n;
    }
  }

  /**
   * Run the full fusion pipeline over a preset dataset.
   * dataset: { gt_path, vio_local_path, vps_candidates } (from sandbox_data.json)
   * cfg: { processNoiseStd, vioPosNoiseStd, vioVelNoiseStd, vpsNoiseStd,
   *        softFrames, chi2Threshold (0/null disables), vpsRate }
   */
  function runFusion(dataset, cfg) {
    const n = dataset.gt_path.length;
    const vioLocal = dataset.vio_local_path;
    const vioVel = vioLocal.map((p, i) => i === 0 ? [0, 0] : vecSub(p, vioLocal[i - 1]));

    // Subsample the dense candidate set at the chosen rate.
    const candidatesByFrame = {};
    dataset.vps_candidates.forEach(c => { candidatesByFrame[c.frame] = c; });
    const vpsFixFrames = [];

    const ukf = new UKFFusion(cfg);
    const fused = new Array(n).fill(null);
    let offset = null;
    let bootstrapped = false;

    for (let i = 0; i < n; i++) {
      let vpsObs = null;
      if (i % cfg.vpsRate === 0 && candidatesByFrame[i]) {
        vpsObs = [candidatesByFrame[i].x, candidatesByFrame[i].y];
        vpsFixFrames.push(i);
      }
      if (!bootstrapped) {
        if (!vpsObs) continue;
        offset = vecSub(vpsObs, vioLocal[i]);
        ukf.bootstrap(vpsObs[0], vpsObs[1], vioVel[i][0], vioVel[i][1]);
        fused[i] = vpsObs;
        bootstrapped = true;
        continue;
      }
      const vioObs = [vioLocal[i][0] + offset[0], vioLocal[i][1] + offset[1], vioVel[i][0], vioVel[i][1]];
      fused[i] = ukf.step(vioObs, vpsObs);
    }

    const vioGlobal = offset ? vioLocal.map(p => vecAdd(p, offset)) : vioLocal.slice();
    return { fusedPath: fused, vioGlobalPath: vioGlobal, vpsFixFrames };
  }

  function errorSeries(path, gt) {
    return path.map((p, i) => p ? Math.hypot(p[0] - gt[i][0], p[1] - gt[i][1]) : null);
  }

  return { runFusion, errorSeries };
})();
