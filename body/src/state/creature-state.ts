/**
 * Creature state management with interpolation.
 *
 * WebSocket updates set TARGET values. The render loop calls `tick(dt)` each
 * frame, which smoothly interpolates CURRENT values toward targets. The shader
 * reads from `current`.
 */

import { lerp, smoothLerpFactor, clamp } from '$lib/utils';
import type { SystemContext, EmotionVector, NeedsMap } from '$lib/types';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface NeedValues {
  hunger: number;
  thirst: number;
  boredom: number;
  loneliness: number;
  stamina: number;
}

export interface InterpolatableState {
  mood_valence: number;
  mood_arousal: number;
  coherence: number;
  needs: NeedValues;
}

export interface BehaviorTransition {
  current: string;
  previous: string | null;
  transition: number;  // 0 = just started, 1 = fully transitioned
}

export interface EmotionFlash {
  valence: number;
  arousal: number;
  intensity: number;
  remaining: number;   // seconds remaining
  duration: number;     // original duration
}

export interface Position {
  x: number;
  y: number;
}

export interface CreatureState {
  target: InterpolatableState;
  current: InterpolatableState;
  behavior: BehaviorTransition;
  emotion_flash: EmotionFlash;
  position: Position;
  velocity: Position;
  screen_bounds: { width: number; height: number };
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const TRANSITION_DURATION = 0.75;  // seconds for behavior crossfade
const EMOTION_FLASH_DURATION_MIN = 0.4;
const EMOTION_FLASH_DURATION_MAX = 1.0;
const LERP_SMOOTHNESS = 0.01;  // ~95% convergence in 1 second
const MAX_DT = 0.05;  // cap dt to handle tab-away

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export function createCreatureState(screenWidth: number, screenHeight: number): CreatureState {
  const defaultNeeds: NeedValues = {
    hunger: 0.5,
    thirst: 0.5,
    boredom: 0.5,
    loneliness: 0.5,
    stamina: 0.5,
  };

  const defaultInterp: InterpolatableState = {
    mood_valence: 0,
    mood_arousal: 0,
    coherence: 0.5,
    needs: { ...defaultNeeds },
  };

  return {
    target: { ...defaultInterp, needs: { ...defaultNeeds } },
    current: { ...defaultInterp, needs: { ...defaultNeeds } },
    behavior: {
      current: 'idle',
      previous: null,
      transition: 1.0,
    },
    emotion_flash: {
      valence: 0,
      arousal: 0,
      intensity: 0,
      remaining: 0,
      duration: 0,
    },
    position: {
      x: Math.random() * (screenWidth - 192) + 96,
      y: Math.random() * (screenHeight - 192) + 96,
    },
    velocity: { x: 0, y: 0 },
    screen_bounds: { width: screenWidth, height: screenHeight },
  };
}

// ---------------------------------------------------------------------------
// State update from backend
// ---------------------------------------------------------------------------

/**
 * Apply a state update from the soul server.
 * Sets interpolation targets — actual values will lerp toward these in tick().
 */
export function applyStateUpdate(state: CreatureState, context: SystemContext): void {
  // Mood targets
  state.target.mood_valence = context.mood.valence;
  state.target.mood_arousal = context.mood.arousal;

  // Need targets
  const needs = context.needs as unknown as Record<string, { satisfaction: number }>;
  for (const key of Object.keys(state.target.needs) as (keyof NeedValues)[]) {
    if (needs[key]) {
      state.target.needs[key] = needs[key].satisfaction;
    }
  }

  // Coherence derived from needs
  state.target.coherence = deriveCoherence(state.target.needs);

  // Behavior transition
  const newBehavior = context.behavior.name ?? 'idle';
  if (newBehavior !== state.behavior.current) {
    state.behavior.previous = state.behavior.current;
    state.behavior.current = newBehavior;
    state.behavior.transition = 0;
  }

  // Emotion flash from strongest new emotion
  if (context.emotional_state.length > 0) {
    const strongest = context.emotional_state.reduce<EmotionVector | null>(
      (best, e) => (!best || e.intensity > best.intensity ? e : best),
      null,
    );
    if (strongest && strongest.intensity > 0.2) {
      const duration = EMOTION_FLASH_DURATION_MIN +
        (EMOTION_FLASH_DURATION_MAX - EMOTION_FLASH_DURATION_MIN) * strongest.intensity;
      state.emotion_flash = {
        valence: strongest.valence,
        arousal: strongest.arousal,
        intensity: strongest.intensity,
        remaining: duration,
        duration,
      };
    }
  }
}

// ---------------------------------------------------------------------------
// Per-frame tick
// ---------------------------------------------------------------------------

/**
 * Advance the creature state by `rawDt` seconds.
 * Call this every frame from requestAnimationFrame.
 */
export function tickCreatureState(state: CreatureState, rawDt: number): void {
  const dt = Math.min(rawDt, MAX_DT);
  const factor = smoothLerpFactor(dt, LERP_SMOOTHNESS);

  // Interpolate continuous values
  state.current.mood_valence = lerp(state.current.mood_valence, state.target.mood_valence, factor);
  state.current.mood_arousal = lerp(state.current.mood_arousal, state.target.mood_arousal, factor);
  state.current.coherence = lerp(state.current.coherence, state.target.coherence, factor);

  for (const key of Object.keys(state.current.needs) as (keyof NeedValues)[]) {
    state.current.needs[key] = lerp(
      state.current.needs[key],
      state.target.needs[key],
      factor,
    );
  }

  // Behavior transition progress
  if (state.behavior.transition < 1.0) {
    state.behavior.transition = Math.min(1.0, state.behavior.transition + dt / TRANSITION_DURATION);
  }

  // Emotion flash decay
  if (state.emotion_flash.remaining > 0) {
    state.emotion_flash.remaining -= dt;
    if (state.emotion_flash.remaining <= 0) {
      state.emotion_flash.intensity = 0;
      state.emotion_flash.remaining = 0;
    } else {
      // Intensity decays with remaining time
      state.emotion_flash.intensity =
        (state.emotion_flash.remaining / state.emotion_flash.duration) *
        state.emotion_flash.intensity;
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Derive coherence (0-1) from need satisfactions.
 * Low satisfaction hurts coherence more than high satisfaction helps.
 */
function deriveCoherence(needs: NeedValues): number {
  const values = Object.values(needs);
  const avg = values.reduce((a, b) => a + b, 0) / values.length;
  const lowCount = values.filter((v) => v < 0.3).length;
  const penalty = (lowCount / values.length) * 0.3;
  return clamp(avg - penalty, 0, 1);
}
