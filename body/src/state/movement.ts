/**
 * Movement controller: drives creature screen position based on behavior.
 *
 * Pure state manipulation — no Tauri imports. The overlay component handles
 * window positioning and cursor polling separately.
 */

import { clamp } from '$lib/utils';
import type { CreatureState, Position } from './creature-state';

// ---------------------------------------------------------------------------
// Speed constants (logical pixels per second)
// ---------------------------------------------------------------------------

const IDLE_WANDER_SPEED = 30;     // lazy drift — visible but unhurried
const IDLE_PAUSE_MIN = 4.0;       // pause at target (seconds)
const IDLE_PAUSE_MAX = 12.0;      // long pauses — creature is just vibing

const WALK_SPEED = 65;            // purposeful movement
const WALK_PAUSE_MIN = 0.5;
const WALK_PAUSE_MAX = 2.0;
const WALK_ARRIVAL_THRESHOLD = 20;

const CHASE_SPEED = 120;
const CHASE_ARRIVAL_THRESHOLD = 30;
const CURSOR_POLL_INTERVAL = 0.1;

const RELAX_DRIFT_SPEED = 8;     // visible gentle sway

// Velocity smoothing per behavior (higher = more responsive).
const VELOCITY_SMOOTHING: Record<string, number> = {
  idle: 0.06,
  walk: 0.08,
  chase: 0.12,
  sleep: 0.02,
  relax: 0.04,
  attentive: 0.02,
};

const EDGE_BOUNCE = 0.3;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface MovementController {
  tick(state: CreatureState, dt: number): void;
  updateCursorPosition(x: number, y: number): void;
  shouldPollCursor(dt: number): boolean;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export function createMovementController(canvasSize: number): MovementController {
  const margin = canvasSize / 2;

  let noisePhase = Math.random() * 1000;
  let lastBehavior = '';

  // Idle wander state
  let idleTarget: Position | null = null;
  let idlePauseRemaining = 0;

  // Walk state
  let walkTarget: Position | null = null;
  let walkPauseRemaining = 0;

  // Cursor state (for chase)
  let cursorX = 0;
  let cursorY = 0;
  let cursorKnown = false;

  // Cursor poll throttle
  let cursorPollTimer = 0;

  // ------------------------------------------------------------------
  // Main tick
  // ------------------------------------------------------------------

  function tick(state: CreatureState, dt: number): void {
    const behavior = state.behavior.current;
    const bounds = state.screen_bounds;

    if (behavior !== lastBehavior) {
      onBehaviorChange(behavior, state);
      lastBehavior = behavior;
    }

    noisePhase += dt;

    let targetVx = 0;
    let targetVy = 0;

    switch (behavior) {
      case 'idle': {
        const idle = tickIdle(state, dt, bounds);
        targetVx = idle.vx;
        targetVy = idle.vy;
        break;
      }

      case 'walk': {
        const w = tickWalk(state, dt, bounds);
        targetVx = w.vx;
        targetVy = w.vy;
        break;
      }

      case 'chase': {
        const c = tickChase(state);
        targetVx = c.vx;
        targetVy = c.vy;
        break;
      }

      case 'sleep':
      case 'attentive':
        break;

      case 'relax': {
        // Gentle multi-harmonic sway — visible but lazy
        targetVx = (
          Math.sin(noisePhase * 0.25) +
          Math.sin(noisePhase * 0.13 + 1.7) * 0.6
        ) * RELAX_DRIFT_SPEED;
        targetVy = (
          Math.cos(noisePhase * 0.2 + 0.8) +
          Math.cos(noisePhase * 0.11 + 2.3) * 0.6
        ) * RELAX_DRIFT_SPEED;
        break;
      }

      default: {
        const fallback = tickIdle(state, dt, bounds);
        targetVx = fallback.vx;
        targetVy = fallback.vy;
      }
    }

    // Smooth velocity transition (frame-rate independent)
    const smoothing = VELOCITY_SMOOTHING[behavior] ?? 0.05;
    const factor = 1 - Math.pow(1 - smoothing, dt * 60);
    state.velocity.x += (targetVx - state.velocity.x) * factor;
    state.velocity.y += (targetVy - state.velocity.y) * factor;

    state.position.x += state.velocity.x * dt;
    state.position.y += state.velocity.y * dt;

    // Edge clamping with soft bounce
    const minX = margin;
    const maxX = bounds.width - margin;
    const minY = margin;
    const maxY = bounds.height - margin;

    if (state.position.x < minX) {
      state.position.x = minX;
      state.velocity.x = Math.abs(state.velocity.x) * EDGE_BOUNCE;
      walkTarget = null;
      idleTarget = null;
    } else if (state.position.x > maxX) {
      state.position.x = maxX;
      state.velocity.x = -Math.abs(state.velocity.x) * EDGE_BOUNCE;
      walkTarget = null;
      idleTarget = null;
    }

    if (state.position.y < minY) {
      state.position.y = minY;
      state.velocity.y = Math.abs(state.velocity.y) * EDGE_BOUNCE;
      walkTarget = null;
      idleTarget = null;
    } else if (state.position.y > maxY) {
      state.position.y = maxY;
      state.velocity.y = -Math.abs(state.velocity.y) * EDGE_BOUNCE;
      walkTarget = null;
      idleTarget = null;
    }
  }

  // ------------------------------------------------------------------
  // Idle: pick nearby random targets, drift toward them, pause, repeat
  // ------------------------------------------------------------------

  function tickIdle(
    state: CreatureState,
    dt: number,
    bounds: { width: number; height: number },
  ): { vx: number; vy: number } {
    if (idlePauseRemaining > 0) {
      idlePauseRemaining -= dt;
      return { vx: 0, vy: 0 };
    }

    if (!idleTarget) {
      // Pick a random target anywhere on screen — lazy screen-wide wander
      idleTarget = {
        x: margin + Math.random() * (bounds.width - 2 * margin),
        y: margin + Math.random() * (bounds.height - 2 * margin),
      };
    }

    const dx = idleTarget.x - state.position.x;
    const dy = idleTarget.y - state.position.y;
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist < 20) {
      idleTarget = null;
      idlePauseRemaining =
        IDLE_PAUSE_MIN + Math.random() * (IDLE_PAUSE_MAX - IDLE_PAUSE_MIN);
      return { vx: 0, vy: 0 };
    }

    const nx = dx / dist;
    const ny = dy / dist;

    // Subtle wobble while drifting
    const wobble = Math.sin(noisePhase * 1.8) * 2;

    return {
      vx: nx * IDLE_WANDER_SPEED + (-ny) * wobble,
      vy: ny * IDLE_WANDER_SPEED + nx * wobble,
    };
  }

  // ------------------------------------------------------------------
  // Walk: pick random screen-wide target, move purposefully, pause
  // ------------------------------------------------------------------

  function tickWalk(
    state: CreatureState,
    dt: number,
    bounds: { width: number; height: number },
  ): { vx: number; vy: number } {
    if (walkPauseRemaining > 0) {
      walkPauseRemaining -= dt;
      return { vx: 0, vy: 0 };
    }

    if (!walkTarget) {
      walkTarget = {
        x: margin + Math.random() * (bounds.width - 2 * margin),
        y: margin + Math.random() * (bounds.height - 2 * margin),
      };
    }

    const dx = walkTarget.x - state.position.x;
    const dy = walkTarget.y - state.position.y;
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist < WALK_ARRIVAL_THRESHOLD) {
      walkTarget = null;
      walkPauseRemaining =
        WALK_PAUSE_MIN + Math.random() * (WALK_PAUSE_MAX - WALK_PAUSE_MIN);
      return { vx: 0, vy: 0 };
    }

    const nx = dx / dist;
    const ny = dy / dist;
    const wobble = Math.sin(noisePhase * 2.5) * 3;

    return {
      vx: nx * WALK_SPEED + (-ny) * wobble,
      vy: ny * WALK_SPEED + nx * wobble,
    };
  }

  // ------------------------------------------------------------------
  // Chase: follow cursor at high speed
  // ------------------------------------------------------------------

  function tickChase(state: CreatureState): { vx: number; vy: number } {
    if (!cursorKnown) {
      return { vx: 0, vy: 0 };
    }

    const dx = cursorX - state.position.x;
    const dy = cursorY - state.position.y;
    const dist = Math.sqrt(dx * dx + dy * dy);

    if (dist < CHASE_ARRIVAL_THRESHOLD) {
      return { vx: -dy * 0.5, vy: dx * 0.5 };
    }

    const speedScale = clamp(dist / 200, 0.3, 1.0);
    const nx = dx / dist;
    const ny = dy / dist;

    return {
      vx: nx * CHASE_SPEED * speedScale,
      vy: ny * CHASE_SPEED * speedScale,
    };
  }

  // ------------------------------------------------------------------
  // Behavior transition
  // ------------------------------------------------------------------

  function onBehaviorChange(to: string, state: CreatureState): void {
    walkTarget = null;
    walkPauseRemaining = 0;
    idleTarget = null;
    idlePauseRemaining = 0;

    if (to === 'chase' && !cursorKnown) {
      cursorX = state.position.x;
      cursorY = state.position.y;
    }
  }

  // ------------------------------------------------------------------
  // Cursor management
  // ------------------------------------------------------------------

  function updateCursorPosition(x: number, y: number): void {
    cursorX = x;
    cursorY = y;
    cursorKnown = true;
  }

  function shouldPollCursor(dt: number): boolean {
    cursorPollTimer += dt;
    if (cursorPollTimer >= CURSOR_POLL_INTERVAL) {
      cursorPollTimer -= CURSOR_POLL_INTERVAL;
      return true;
    }
    return false;
  }

  return { tick, updateCursorPosition, shouldPollCursor };
}
