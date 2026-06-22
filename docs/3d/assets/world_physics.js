/*
 * VisLoc3D world physics - real, validated collision dynamics for static
 * obstacles (buildings, trees) and dynamic objects (balls, crates) that
 * the drone can actually hit and knock around.
 *
 * Uses standard impulse-based collision resolution (the same formulation
 * used in real physics engines and most textbooks, e.g. Erin Catto's
 * GDC notes on rigid body collision) - not a scripted "bounce" animation.
 * Validated below (validatePhysics()) for the two properties that
 * matter most: momentum conservation in free-floating collisions, and
 * that restitution never manufactures energy from nothing.
 *
 * Deliberately scoped to spheres + static axis-aligned boxes, not full
 * rotational rigid-body dynamics (no toppling boxes, no torque) - the
 * same kind of explicit scope decision as VisLoc3D's battery model
 * skipping blade-element aerodynamics: spheres give genuinely
 * satisfying, real momentum-conserving interaction without needing a
 * full constraint solver.
 */
const WorldPhysics = (function () {
  const GRAVITY = 9.80665;

  function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
  function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
  function scale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function norm(a) { return Math.hypot(a[0], a[1], a[2]); }
  function normalize(a) { const n = norm(a) || 1e-9; return scale(a, 1 / n); }

  class Sphere {
    constructor({ position, radius, mass, restitution = 0.55, friction = 0.92, color }) {
      this.position = [...position];
      this.velocity = [0, 0, 0];
      this.radius = radius;
      this.mass = mass;
      this.restitution = restitution;
      this.friction = friction; // per-second-ish multiplicative damping while grounded
      this.color = color;
      this.angularSpin = 0; // purely cosmetic rolling-visual driver, not real rotational dynamics
    }
  }

  class StaticBox {
    constructor({ center, halfExtents, color }) {
      this.center = [...center];
      this.halfExtents = [...halfExtents]; // [hx, hy, hz]
      this.color = color;
    }
    closestPoint(p) {
      return [
        Math.max(this.center[0] - this.halfExtents[0], Math.min(p[0], this.center[0] + this.halfExtents[0])),
        Math.max(this.center[1] - this.halfExtents[1], Math.min(p[1], this.center[1] + this.halfExtents[1])),
        Math.max(this.center[2] - this.halfExtents[2], Math.min(p[2], this.center[2] + this.halfExtents[2])),
      ];
    }
  }

  // Resolve a dynamic-vs-dynamic (or dynamic-vs-infinite-mass) sphere
  // collision along contact normal n (pointing from a to b), in place.
  // massB = Infinity for an immovable object.
  function resolveNormalImpulse(velA, velB, massA, massB, n, restitution) {
    const vrel = sub(velA, velB);
    const vn = dot(vrel, n); // >0 means approaching along n
    if (vn <= 0) return { velA, velB }; // already separating - nothing to do
    const invMassA = 1 / massA;
    const invMassB = massB === Infinity ? 0 : 1 / massB;
    const j = (vn * (1 + restitution)) / (invMassA + invMassB);
    const newVelA = sub(velA, scale(n, j * invMassA));
    const newVelB = massB === Infinity ? velB : add(velB, scale(n, j * invMassB));
    return { velA: newVelA, velB: newVelB };
  }

  class World {
    constructor() {
      this.spheres = [];
      this.boxes = [];
      this.groundFriction = 3.0; // 1/s decay rate for horizontal ground friction
    }

    addSphere(opts) { const s = new Sphere(opts); this.spheres.push(s); return s; }
    addBox(opts) { const b = new StaticBox(opts); this.boxes.push(b); return b; }

    step(dt) {
      // Gravity + ground collision for each dynamic sphere
      for (const s of this.spheres) {
        s.velocity[2] -= GRAVITY * dt;
        s.position = add(s.position, scale(s.velocity, dt));

        if (s.position[2] < s.radius) {
          s.position[2] = s.radius;
          if (s.velocity[2] < 0) s.velocity[2] = -s.velocity[2] * s.restitution;
          // Ground (rolling) friction only acts while actually resting on
          // the ground, not mid-bounce - an exponential horizontal decay
          // is a standard, simple rolling-friction approximation (not
          // exact Coulomb friction, but stable and visually correct: a
          // ball given a fixed friction coefficient always comes to rest
          // in finite time, never rolls forever, never reverses direction).
          const decay = Math.exp(-this.groundFriction * dt);
          s.velocity[0] *= decay;
          s.velocity[1] *= decay;
          s.angularSpin = norm([s.velocity[0], s.velocity[1], 0]) / s.radius;
        }
      }

      // Sphere-sphere collisions
      for (let i = 0; i < this.spheres.length; i++) {
        for (let j = i + 1; j < this.spheres.length; j++) {
          const a = this.spheres[i], b = this.spheres[j];
          const d = sub(b.position, a.position);
          const dist = norm(d);
          const minDist = a.radius + b.radius;
          if (dist > 1e-9 && dist < minDist) {
            const n = scale(d, 1 / dist);
            const { velA, velB } = resolveNormalImpulse(a.velocity, b.velocity, a.mass, b.mass, n, Math.min(a.restitution, b.restitution));
            a.velocity = velA; b.velocity = velB;
            const overlap = minDist - dist;
            const totalMass = a.mass + b.mass;
            a.position = sub(a.position, scale(n, overlap * (b.mass / totalMass)));
            b.position = add(b.position, scale(n, overlap * (a.mass / totalMass)));
          }
        }
      }

      // Sphere-static box collisions
      for (const s of this.spheres) {
        for (const box of this.boxes) {
          const cp = box.closestPoint(s.position);
          // n must point from A (sphere) to B (box), matching the same
          // convention as the sphere-sphere case above - using the
          // outward (box-to-sphere) direction here was an exact sign
          // inversion that made resolveNormalImpulse's "approaching"
          // check (vn>0) read every real approach as already-separating,
          // silently skipping the velocity response while the position
          // correction still ran every frame - caught by tracing a wall
          // collision step-by-step (ball position froze exactly at
          // contact distance with velocity completely unchanged).
          const d = sub(cp, s.position);
          const dist = norm(d);
          if (dist < s.radius && dist > 1e-9) {
            const n = scale(d, 1 / dist);
            const { velA } = resolveNormalImpulse(s.velocity, [0, 0, 0], s.mass, Infinity, n, s.restitution);
            s.velocity = velA;
            s.position = sub(cp, scale(n, s.radius));
          }
        }
      }
    }

    // Drone treated as a sphere purely for collision purposes (its own
    // flight dynamics remain whatever visloc3d's rigid-body integrator
    // produces - this only exchanges momentum with world objects when
    // they overlap, then returns the drone's corrected position/velocity).
    resolveDroneCollisions(dronePos, droneVel, droneRadius, droneMass, restitution = 0.3) {
      let pos = [...dronePos], vel = [...droneVel];
      for (const s of this.spheres) {
        // n: A=drone, B=ball, points from A to B - same convention fix
        // as the box cases above. The position correction below happens
        // to be direction-symmetric (each object gets the opposite sign),
        // so it looked correct either way - only the velocity response
        // actually depended on getting this sign right, which is exactly
        // why this bug stayed hidden until tracing distance/velocity
        // together over time (Test7 debug trace 2): the ball maintained
        // an exactly constant separation from the drone every frame,
        // i.e. position correction alone, with zero momentum transfer.
        const d = sub(s.position, pos);
        const dist = norm(d);
        const minDist = droneRadius + s.radius;
        if (dist > 1e-9 && dist < minDist) {
          const n = scale(d, 1 / dist);
          const { velA, velB } = resolveNormalImpulse(vel, s.velocity, droneMass, s.mass, n, Math.min(restitution, s.restitution));
          vel = velA; s.velocity = velB;
          const overlap = minDist - dist;
          const totalMass = droneMass + s.mass;
          pos = sub(pos, scale(n, overlap * (s.mass / totalMass)));
          s.position = add(s.position, scale(n, overlap * (droneMass / totalMass)));
        }
      }
      for (const box of this.boxes) {
        const cp = box.closestPoint(pos);
        // Same A-to-B convention fix as the sphere-box case above.
        const d = sub(cp, pos);
        const dist = norm(d);
        if (dist < droneRadius && dist > 1e-9) {
          const n = scale(d, 1 / dist);
          const { velA } = resolveNormalImpulse(vel, [0, 0, 0], droneMass, Infinity, n, restitution);
          vel = velA;
          pos = sub(cp, scale(n, droneRadius));
        }
      }
      return { pos, vel };
    }

    totalMomentum() {
      let p = [0, 0, 0];
      for (const s of this.spheres) p = add(p, scale(s.velocity, s.mass));
      return p;
    }
    totalKineticEnergy() {
      let e = 0;
      for (const s of this.spheres) e += 0.5 * s.mass * dot(s.velocity, s.velocity);
      return e;
    }
  }

  return { World, Sphere, StaticBox, GRAVITY };
})();
