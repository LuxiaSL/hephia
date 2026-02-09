/**
 * Shared utility functions.
 */

/** Clamp a value between min and max. */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/** Linear interpolation. */
export function lerp(current: number, target: number, factor: number): number {
  return current + factor * (target - current);
}

/**
 * Frame-rate-independent lerp factor.
 * Returns a factor that achieves ~95% convergence in 1 second regardless of dt.
 */
export function smoothLerpFactor(dt: number, smoothness: number = 0.01): number {
  return 1 - Math.pow(smoothness, dt);
}

/** Map a value from one range to another. */
export function mapRange(
  value: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number,
): number {
  return outMin + ((value - inMin) / (inMax - inMin)) * (outMax - outMin);
}

/**
 * Map mood valence (-1 to 1) to a hue value (0 to 360).
 * Warm (positive) → orange/amber (30-50)
 * Neutral → white/light blue (180-200)
 * Cold (negative) → deep blue/purple (220-280)
 */
export function valenceToHue(valence: number): number {
  if (valence >= 0) {
    // Positive: 40 (warm amber) to 180 (neutral cyan)
    return mapRange(valence, 0, 1, 180, 40);
  } else {
    // Negative: 180 (neutral cyan) to 260 (deep purple)
    return mapRange(valence, -1, 0, 260, 180);
  }
}

/**
 * Convert HSL to RGB (all 0-1 range).
 */
export function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const hNorm = ((h % 360) + 360) % 360 / 360;
  let r: number, g: number, b: number;

  if (s === 0) {
    r = g = b = l;
  } else {
    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hueToRgb(p, q, hNorm + 1 / 3);
    g = hueToRgb(p, q, hNorm);
    b = hueToRgb(p, q, hNorm - 1 / 3);
  }

  return [r, g, b];
}

function hueToRgb(p: number, q: number, t: number): number {
  let tNorm = t;
  if (tNorm < 0) tNorm += 1;
  if (tNorm > 1) tNorm -= 1;
  if (tNorm < 1 / 6) return p + (q - p) * 6 * tNorm;
  if (tNorm < 1 / 2) return q;
  if (tNorm < 2 / 3) return p + (q - p) * (2 / 3 - tNorm) * 6;
  return p;
}
