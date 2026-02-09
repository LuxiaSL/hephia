/**
 * Shader playground — interactive creature parameter tweaking.
 *
 * Imports the same creature renderer used by the real app.
 * Drives it with slider-controlled mock state instead of WebSocket data.
 */

import { createCreatureRenderer, BEHAVIOR_MOTIFS } from '../src/creature';
import type { CreatureRenderer } from '../src/creature';
import {
  createCreatureState,
  tickCreatureState,
  type CreatureState,
} from '../src/state/creature-state';

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

const canvas = document.getElementById('creature') as HTMLCanvasElement;
let renderer: CreatureRenderer;
let creatureState: CreatureState;

try {
  renderer = createCreatureRenderer(canvas, 400);
  creatureState = createCreatureState(256, 256);

  // Initial resize at device pixel ratio
  const dpr = window.devicePixelRatio || 1;
  renderer.resize(256, 256, dpr);
} catch (e) {
  console.error('Failed to initialize creature renderer:', e);
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.fillStyle = '#f44';
    ctx.font = '14px monospace';
    ctx.fillText('WebGL 2 init failed', 10, 128);
    ctx.fillText(String(e), 10, 148);
  }
  throw e;
}

// ---------------------------------------------------------------------------
// Slider wiring
// ---------------------------------------------------------------------------

function getSlider(id: string): HTMLInputElement {
  return document.getElementById(id) as HTMLInputElement;
}

function getSliderFloat(id: string): number {
  const el = getSlider(id);
  return parseInt(el.value, 10) / 100;
}

// Update value displays
const sliders = document.querySelectorAll<HTMLInputElement>('input[type="range"]');
for (const slider of sliders) {
  const valEl = document.getElementById(slider.id + '-val');
  if (valEl) {
    const updateVal = () => {
      const raw = parseInt(slider.value, 10);
      const mapped = raw / 100;
      valEl.textContent = mapped.toFixed(2);
    };
    slider.addEventListener('input', updateVal);
    updateVal();
  }
}

// ---------------------------------------------------------------------------
// State sync from sliders
// ---------------------------------------------------------------------------

function syncStateFromSliders(): void {
  // Mood
  creatureState.target.mood_valence = getSliderFloat('mood-valence');
  creatureState.target.mood_arousal = getSliderFloat('mood-arousal');

  // Needs
  creatureState.target.needs.hunger = getSliderFloat('need-hunger');
  creatureState.target.needs.thirst = getSliderFloat('need-thirst');
  creatureState.target.needs.boredom = getSliderFloat('need-boredom');
  creatureState.target.needs.loneliness = getSliderFloat('need-loneliness');
  creatureState.target.needs.stamina = getSliderFloat('need-stamina');

  // Coherence (derived from needs)
  const needs = creatureState.target.needs;
  const vals = [needs.hunger, needs.thirst, needs.boredom, needs.loneliness, needs.stamina];
  const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
  const lowCount = vals.filter(v => v < 0.3).length;
  const penalty = (lowCount / vals.length) * 0.3;
  creatureState.target.coherence = Math.max(0, avg - penalty);

  // Behavior
  const behaviorSelect = document.getElementById('behavior') as HTMLSelectElement;
  const newBehavior = behaviorSelect.value;
  if (newBehavior !== creatureState.behavior.current) {
    creatureState.behavior.previous = creatureState.behavior.current;
    creatureState.behavior.current = newBehavior;
    creatureState.behavior.transition = 0;
  }
}

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

interface Preset {
  valence: number;
  arousal: number;
  hunger: number;
  thirst: number;
  boredom: number;
  loneliness: number;
  stamina: number;
  behavior: string;
}

const PRESETS: Record<string, Preset> = {
  'preset-happy-idle': {
    valence: 70, arousal: 20,
    hunger: 80, thirst: 85, boredom: 75, loneliness: 80, stamina: 70,
    behavior: 'idle',
  },
  'preset-sad-sleep': {
    valence: -60, arousal: -50,
    hunger: 30, thirst: 35, boredom: 40, loneliness: 20, stamina: 15,
    behavior: 'sleep',
  },
  'preset-excited-chase': {
    valence: 50, arousal: 85,
    hunger: 60, thirst: 55, boredom: 90, loneliness: 70, stamina: 80,
    behavior: 'chase',
  },
  'preset-calm-relax': {
    valence: 30, arousal: -30,
    hunger: 65, thirst: 70, boredom: 55, loneliness: 60, stamina: 50,
    behavior: 'relax',
  },
  'preset-lonely-walk': {
    valence: -20, arousal: 10,
    hunger: 50, thirst: 50, boredom: 60, loneliness: 15, stamina: 45,
    behavior: 'walk',
  },
};

function applyPreset(presetId: string) {
  const p = PRESETS[presetId];
  if (!p) return;

  getSlider('mood-valence').value = String(p.valence);
  getSlider('mood-arousal').value = String(p.arousal);
  getSlider('need-hunger').value = String(p.hunger);
  getSlider('need-thirst').value = String(p.thirst);
  getSlider('need-boredom').value = String(p.boredom);
  getSlider('need-loneliness').value = String(p.loneliness);
  getSlider('need-stamina').value = String(p.stamina);
  (document.getElementById('behavior') as HTMLSelectElement).value = p.behavior;

  // Trigger value display updates
  sliders.forEach(s => s.dispatchEvent(new Event('input')));
}

for (const id of Object.keys(PRESETS)) {
  document.getElementById(id)?.addEventListener('click', () => applyPreset(id));
}

// ---------------------------------------------------------------------------
// Emotion flash buttons
// ---------------------------------------------------------------------------

function triggerFlash(valence: number, arousal: number) {
  creatureState.emotion_flash = {
    valence,
    arousal,
    intensity: 0.9,
    remaining: 0.8,
    duration: 0.8,
  };
}

document.getElementById('flash-positive')?.addEventListener('click', () => triggerFlash(0.8, 0.5));
document.getElementById('flash-negative')?.addEventListener('click', () => triggerFlash(-0.7, 0.3));
document.getElementById('flash-surprise')?.addEventListener('click', () => triggerFlash(0.2, 0.9));

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------

let lastTime = performance.now();

function frame(now: number) {
  const dt = Math.min((now - lastTime) / 1000, 0.05);
  lastTime = now;

  // Sync slider state → creature targets
  syncStateFromSliders();

  // Tick interpolation
  tickCreatureState(creatureState, dt);

  // Feed state to renderer
  renderer.updateState(creatureState);

  // Render
  renderer.render(dt);

  requestAnimationFrame(frame);
}

requestAnimationFrame(frame);

console.log(
  'Hephia Creature Playground loaded — %d behaviors available: %s',
  Object.keys(BEHAVIOR_MOTIFS).length,
  Object.keys(BEHAVIOR_MOTIFS).join(', '),
);
