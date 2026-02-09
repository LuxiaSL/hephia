/**
 * Behavior motif definitions.
 *
 * Each behavior produces a distinct movement pattern and structural shape
 * in the particle system. These parameters control forces, formation shape,
 * and animation characteristics.
 */

export interface BehaviorMotif {
  /** Strength of attraction toward center (0-1). Higher = tighter formation. */
  centerPull: number;
  /** Target formation radius in normalized space (0-1). */
  spread: number;
  /** Orbital rotation speed around center. */
  rotationSpeed: number;
  /** Breathing/pulse oscillation frequency. */
  pulseRate: number;
  /** Breathing amplitude (radial push/pull strength). */
  pulseAmount: number;
  /** Per-frame damping factor at 60fps. Lower = more friction. */
  damping: number;
  /** Random wandering force strength. */
  noiseStrength: number;
  /** Stretch factor along drift direction (0 = none, 1 = full). */
  elongation: number;
  /** Point size range in logical pixels [min, max]. */
  pointSize: [number, number];
  /** Base alpha multiplier (0-1). */
  baseAlpha: number;
}

// ---------------------------------------------------------------------------
// Motif definitions per behavior
// ---------------------------------------------------------------------------

export const BEHAVIOR_MOTIFS: Record<string, BehaviorMotif> = {
  idle: {
    centerPull: 0.5,
    spread: 0.40,
    rotationSpeed: 0.30,
    pulseRate: 0.5,
    pulseAmount: 0.06,
    damping: 0.94,
    noiseStrength: 0.25,
    elongation: 0.0,
    pointSize: [4, 8],
    baseAlpha: 0.85,
  },
  walk: {
    centerPull: 0.55,
    spread: 0.35,
    rotationSpeed: 0.10,
    pulseRate: 0.3,
    pulseAmount: 0.03,
    damping: 0.92,
    noiseStrength: 0.18,
    elongation: 0.35,
    pointSize: [4, 7],
    baseAlpha: 0.80,
  },
  chase: {
    centerPull: 0.75,
    spread: 0.25,
    rotationSpeed: 0.20,
    pulseRate: 0.1,
    pulseAmount: 0.02,
    damping: 0.88,
    noiseStrength: 0.10,
    elongation: 0.50,
    pointSize: [3, 6],
    baseAlpha: 0.90,
  },
  sleep: {
    centerPull: 0.85,
    spread: 0.18,
    rotationSpeed: 0.04,
    pulseRate: 0.2,
    pulseAmount: 0.04,
    damping: 0.97,
    noiseStrength: 0.04,
    elongation: 0.0,
    pointSize: [3, 5],
    baseAlpha: 0.50,
  },
  relax: {
    centerPull: 0.30,
    spread: 0.55,
    rotationSpeed: 0.15,
    pulseRate: 0.3,
    pulseAmount: 0.05,
    damping: 0.96,
    noiseStrength: 0.30,
    elongation: 0.0,
    pointSize: [5, 9],
    baseAlpha: 0.75,
  },
  attentive: {
    centerPull: 0.60,
    spread: 0.30,
    rotationSpeed: 0.12,
    pulseRate: 0.4,
    pulseAmount: 0.03,
    damping: 0.94,
    noiseStrength: 0.12,
    elongation: 0.25,
    pointSize: [4, 7],
    baseAlpha: 0.90,
  },
};

/** Fallback motif used when behavior name is unrecognized. */
const FALLBACK_MOTIF = BEHAVIOR_MOTIFS.idle;

export function getMotif(behavior: string): BehaviorMotif {
  return BEHAVIOR_MOTIFS[behavior] ?? FALLBACK_MOTIF;
}

/** Linearly blend two motifs by factor t (0 = a, 1 = b). */
export function blendMotifs(a: BehaviorMotif, b: BehaviorMotif, t: number): BehaviorMotif {
  const inv = 1 - t;
  return {
    centerPull: a.centerPull * inv + b.centerPull * t,
    spread: a.spread * inv + b.spread * t,
    rotationSpeed: a.rotationSpeed * inv + b.rotationSpeed * t,
    pulseRate: a.pulseRate * inv + b.pulseRate * t,
    pulseAmount: a.pulseAmount * inv + b.pulseAmount * t,
    damping: a.damping * inv + b.damping * t,
    noiseStrength: a.noiseStrength * inv + b.noiseStrength * t,
    elongation: a.elongation * inv + b.elongation * t,
    pointSize: [
      a.pointSize[0] * inv + b.pointSize[0] * t,
      a.pointSize[1] * inv + b.pointSize[1] * t,
    ],
    baseAlpha: a.baseAlpha * inv + b.baseAlpha * t,
  };
}
