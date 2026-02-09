/**
 * CPU-side particle simulation.
 *
 * Manages N particles with position, velocity, and per-particle properties.
 * Each frame, forces are applied based on behavior motif and creature state,
 * and render data is built for GPU upload.
 */

import type { BehaviorMotif } from './behaviors';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface SimulationParams {
  /** Overall structural coherence (0-1). Scales center pull. */
  coherence: number;
  /** Speed multiplier from arousal. */
  speed: number;
  /** Active (possibly blended) behavior motif. */
  motif: BehaviorMotif;
  /** Drift angle in radians (direction of movement for walk/chase). */
  driftAngle: number;
  /** Transient emotion flash impulse. */
  emotionFlash: {
    valence: number;
    arousal: number;
    intensity: number;
  };
}

export interface RenderParams {
  /** Device pixel ratio for sizing. */
  dpr: number;
  /** Arousal value for size modulation. */
  arousal: number;
  /** Coherence for alpha modulation. */
  coherence: number;
  /** Active behavior motif (for size/alpha ranges). */
  motif: BehaviorMotif;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Simulation data layout: x, y, vx, vy, phase, baseAlpha
const SIM_STRIDE = 6;
// Render data layout: x, y, size, alpha, phase
const RENDER_STRIDE = 5;

// Force scales (tunable)
const PULL_SCALE = 3.0;
const ROTATION_SCALE = 0.8;
const NOISE_SCALE = 0.5;
const PULSE_SCALE = 0.15;
const SPREAD_RESTORE = 1.5;

// Smooth per-particle noise using sine products
function particleNoise(phase: number, time: number): [number, number] {
  const nx =
    Math.sin(time * 1.7 + phase * 127.1) * Math.cos(time * 0.9 + phase * 43.3);
  const ny =
    Math.cos(time * 1.3 + phase * 91.7) * Math.sin(time * 1.1 + phase * 67.1);
  return [nx, ny];
}

// ---------------------------------------------------------------------------
// Particle System
// ---------------------------------------------------------------------------

export class ParticleSystem {
  readonly count: number;

  /** Simulation state: [x, y, vx, vy, phase, baseAlpha] per particle. */
  private sim: Float32Array;

  /** GPU-ready render data: [x, y, size, alpha, phase] per particle. */
  private render: Float32Array;

  constructor(count: number) {
    this.count = count;
    this.sim = new Float32Array(count * SIM_STRIDE);
    this.render = new Float32Array(count * RENDER_STRIDE);
    this.initialize();
  }

  private initialize(): void {
    for (let i = 0; i < this.count; i++) {
      const o = i * SIM_STRIDE;
      const angle = Math.random() * Math.PI * 2;
      const radius = Math.random() * 0.3;

      this.sim[o + 0] = Math.cos(angle) * radius; // x
      this.sim[o + 1] = Math.sin(angle) * radius; // y
      this.sim[o + 2] = 0; // vx
      this.sim[o + 3] = 0; // vy
      this.sim[o + 4] = Math.random() * Math.PI * 2; // phase
      this.sim[o + 5] = 0.6 + Math.random() * 0.4; // baseAlpha
    }
  }

  /**
   * Advance particle positions by dt seconds.
   */
  update(dt: number, time: number, params: SimulationParams): void {
    const { coherence, speed, motif, driftAngle, emotionFlash } = params;

    const effectivePull =
      motif.centerPull * (0.5 + coherence * 0.5) * PULL_SCALE;
    const effectiveRotation = motif.rotationSpeed * speed * ROTATION_SCALE;
    const effectiveNoise =
      motif.noiseStrength * (1.0 + (1.0 - coherence) * 0.5) * speed * NOISE_SCALE;
    const targetRadius = motif.spread * 0.5;
    const dampPow = dt * 60; // frame-rate independent damping exponent

    // Elongation axes
    const elongCos = Math.cos(driftAngle);
    const elongSin = Math.sin(driftAngle);
    const elongFactor = motif.elongation;

    for (let i = 0; i < this.count; i++) {
      const o = i * SIM_STRIDE;
      let x = this.sim[o];
      let y = this.sim[o + 1];
      let vx = this.sim[o + 2];
      let vy = this.sim[o + 3];
      const phase = this.sim[o + 4];

      const dx = -x;
      const dy = -y;
      const dist = Math.sqrt(dx * dx + dy * dy) + 1e-4;
      const nx = dx / dist;
      const ny = dy / dist;

      // 1. Center attraction
      vx += dx * effectivePull * dt;
      vy += dy * effectivePull * dt;

      // 2. Spread restoration: push toward target radius
      const radiusDiff = dist - targetRadius;
      vx += nx * radiusDiff * SPREAD_RESTORE * dt;
      vy += ny * radiusDiff * SPREAD_RESTORE * dt;

      // 3. Orbital rotation (tangential force)
      const tx = -ny;
      const ty = nx;
      vx += tx * effectiveRotation * dt;
      vy += ty * effectiveRotation * dt;

      // 4. Noise
      const [noiseX, noiseY] = particleNoise(phase, time);
      vx += noiseX * effectiveNoise * dt;
      vy += noiseY * effectiveNoise * dt;

      // 5. Breathing pulse
      const pulse =
        Math.sin(time * motif.pulseRate * Math.PI * 2 + phase) *
        motif.pulseAmount *
        PULSE_SCALE;
      vx -= nx * pulse * dt;
      vy -= ny * pulse * dt;

      // 6. Elongation (stretch along drift direction)
      if (elongFactor > 0) {
        // Project position onto drift axis
        const projDrift = x * elongCos + y * elongSin;
        // Push away from center along drift axis
        const elongForce = elongFactor * 0.5;
        vx += elongCos * Math.sign(projDrift) * elongForce * speed * dt;
        vy += elongSin * Math.sign(projDrift) * elongForce * speed * dt;
      }

      // 7. Emotion flash impulse
      if (emotionFlash.intensity > 0) {
        // Positive valence: expand outward. Negative: contract inward.
        const radialPush = emotionFlash.valence * emotionFlash.intensity * 0.8;
        // High arousal: scatter briefly
        const scatterPush = emotionFlash.arousal * emotionFlash.intensity * 0.4;
        const [flashNoiseX, flashNoiseY] = particleNoise(phase + 100, time * 3);

        vx -= nx * radialPush * dt;
        vy -= ny * radialPush * dt;
        vx += flashNoiseX * scatterPush * dt;
        vy += flashNoiseY * scatterPush * dt;
      }

      // 8. Damping (frame-rate independent)
      const damp = Math.pow(motif.damping, dampPow);
      vx *= damp;
      vy *= damp;

      // 9. Integrate position
      x += vx;
      y += vy;

      // 10. Soft boundary (keep particles within visible area)
      const currentDist = Math.sqrt(x * x + y * y);
      if (currentDist > 0.9) {
        const scale = 0.9 / currentDist;
        x *= scale;
        y *= scale;
        vx *= 0.5;
        vy *= 0.5;
      }

      this.sim[o] = x;
      this.sim[o + 1] = y;
      this.sim[o + 2] = vx;
      this.sim[o + 3] = vy;
    }
  }

  /**
   * Build the interleaved render buffer: [x, y, size, alpha, phase] per particle.
   * Returns a Float32Array suitable for GPU upload.
   */
  buildRenderData(params: RenderParams): Float32Array {
    const { dpr, arousal, coherence, motif } = params;
    const sizeMin = motif.pointSize[0] * dpr;
    const sizeMax = motif.pointSize[1] * dpr;
    const sizeRange = sizeMax - sizeMin;
    const arousalSize = 1.0 + arousal * 0.4;
    const alphaScale = 0.3 + coherence * 0.7;

    for (let i = 0; i < this.count; i++) {
      const so = i * SIM_STRIDE;
      const ro = i * RENDER_STRIDE;

      const phase = this.sim[so + 4];
      const baseAlpha = this.sim[so + 5];

      // Per-particle size variation using phase
      const sizeT = (Math.sin(phase * 3.7) * 0.5 + 0.5);
      const size = (sizeMin + sizeRange * sizeT) * arousalSize;

      this.render[ro] = this.sim[so];      // x
      this.render[ro + 1] = this.sim[so + 1]; // y
      this.render[ro + 2] = size;
      this.render[ro + 3] = baseAlpha * alphaScale * motif.baseAlpha;
      this.render[ro + 4] = phase;
    }

    return this.render;
  }
}
