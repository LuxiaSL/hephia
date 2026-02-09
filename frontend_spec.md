# Hephia Frontend Specification

**Version:** 1.0
**Date:** February 9, 2026
**Status:** Ready for Implementation
**Depends on:** `pet_implementation_spec.md` (v2.1), `internal_update_spec.md`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [The Creature: Procedural Particle Entity](#3-the-creature-procedural-particle-entity)
4. [Window System](#4-window-system)
5. [Tauri Rust Layer](#5-tauri-rust-layer)
6. [Frontend State Management](#6-frontend-state-management)
7. [Pet Overlay Window](#7-pet-overlay-window)
8. [Chat Window](#8-chat-window)
9. [Dashboard Window](#9-dashboard-window)
10. [Thought Bubble System](#10-thought-bubble-system)
11. [Right-Click Menu](#11-right-click-menu)
12. [System Tray](#12-system-tray)
13. [Hotkeys](#13-hotkeys)
14. [First-Run Wizard](#14-first-run-wizard)
15. [Backend Lifecycle Management](#15-backend-lifecycle-management)
16. [Error Handling](#16-error-handling)
17. [Dev Workflow & Shader Playground](#17-dev-workflow--shader-playground)
18. [MVP Scope](#18-mvp-scope)
19. [Deferred Features](#19-deferred-features)
20. [Project Structure](#20-project-structure)

---

## 1. Executive Summary

Hephia's frontend ("body") is a Tauri v2 desktop application that gives visual form to the Python soul server. The pet manifests as a **procedural particle cloud** rendered via WebGL fragment shaders on a transparent overlay window. The creature's visual state — color, coherence, movement, dispersion — is driven directly by the backend's internal state system (mood, emotions, needs, behaviors) via WebSocket, with the Rust layer owning the connection and distributing state to Svelte webviews through Tauri IPC.

The user interacts through right-click context menus, a separate chat window, a dashboard with state/settings panels, and thought bubble notifications. The entire experience runs as a single app: Tauri manages the Python backend lifecycle, spawning it on launch and killing it on quit.

**Key technical decisions:**
- Transparent overlay via XWayland (`GDK_BACKEND=x11`) on GNOME/Fedora/Wayland
- WebGL particle system with GLSL shaders, state mapped to shader uniforms
- Svelte + Vite for UI, Tauri v2 for windowing/IPC/lifecycle
- Rust side owns WebSocket connection, emits typed events to all webviews
- Interpolation-based rendering: WebSocket updates set targets, render loop lerps toward them at 60fps
- Linux-first, cross-platform later

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TAURI APPLICATION                             │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    RUST LAYER (src-tauri/)                    │   │
│  │                                                               │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────┐   │   │
│  │  │  Backend     │  │  WebSocket   │  │  Window           │   │   │
│  │  │  Process     │  │  Client      │  │  Manager          │   │   │
│  │  │  Manager     │  │              │  │                   │   │   │
│  │  │  (spawn,     │  │  /ws state   │  │  Overlay window   │   │   │
│  │  │   monitor,   │  │  /ws/chat    │  │  Chat window      │   │   │
│  │  │   restart,   │  │              │  │  Dashboard window  │   │   │
│  │  │   kill)      │  │  Reconnect   │  │  Wizard window    │   │   │
│  │  └──────┬───────┘  │  logic       │  └───────────────────┘   │   │
│  │         │          └──────┬───────┘                           │   │
│  │         │                 │                                    │   │
│  │         │          ┌──────┴───────┐                           │   │
│  │         │          │  IPC Event   │                           │   │
│  │         │          │  Emitter     │                           │   │
│  │         │          │              │                           │   │
│  │         │          │  soul:state  │                           │   │
│  │         │          │  soul:chat   │                           │   │
│  │         │          │  soul:conn   │                           │   │
│  │         │          │  soul:error  │                           │   │
│  │         │          └──────┬───────┘                           │   │
│  └─────────┼─────────────────┼──────────────────────────────────┘   │
│            │                 │                                       │
│            │    ┌────────────┼────────────────┐                     │
│            │    │            │                │                      │
│  ┌─────────▼────▼──┐  ┌─────▼──────┐  ┌─────▼──────┐              │
│  │  Pet Overlay    │  │  Chat      │  │  Dashboard │              │
│  │  Window         │  │  Window    │  │  Window    │              │
│  │                 │  │            │  │            │              │
│  │  WebGL Canvas   │  │  Svelte    │  │  Svelte    │              │
│  │  Particle       │  │  Chat UI   │  │  Tabs:     │              │
│  │  System         │  │  Message   │  │  - State   │              │
│  │                 │  │  history   │  │  - Settings│              │
│  │  Thought        │  │  Input     │  │  - Worker  │              │
│  │  Bubbles        │  │            │  │  - Actions │              │
│  │                 │  │            │  │            │              │
│  │  Right-click    │  │            │  │            │              │
│  │  Menu           │  │            │  │            │              │
│  └─────────────────┘  └────────────┘  └────────────┘              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                          │
                          │ HTTP :5517 / WebSocket
                          ▼
              ┌──────────────────────┐
              │   SOUL SERVER        │
              │   (Python/FastAPI)   │
              │                      │
              │   Internal State     │
              │   Memory System      │
              │   Mind Layer         │
              └──────────────────────┘
```

### Data Flow

1. Soul server internal events (need change, emotion, behavior transition) trigger state broadcast via `/ws`
2. Rust WebSocket client receives state payload
3. Rust emits `soul:state` IPC event to all webview windows
4. Pet overlay's state manager updates interpolation targets
5. Render loop (requestAnimationFrame, 60fps) lerps current values toward targets
6. Shader uniforms updated each frame, particle system renders

Chat flow:
1. User types in chat window → Svelte component emits Tauri command
2. Rust side sends message to backend via `/ws/chat` or REST `/chat`
3. Response arrives → Rust emits `soul:chat` IPC event
4. Chat window renders response; if chat is closed, thought bubble fires on overlay

---

## 3. The Creature: Procedural Particle Entity

### 3.1 Concept

The pet manifests as a **particle cloud with emergent geometric structure**. There is no fixed shape — the "self" is the pattern of organization among particles. Coherence is the primary visual signal: a well-cared-for pet self-organizes into recognizable motifs (rings, lattices, crystalline moments); a neglected or distressed pet scatters and fragments.

This maps directly to the soul server's state:
- **Mood valence** → color temperature (warm hues positive, cool/dark negative)
- **Mood arousal** → animation speed, oscillation frequency, particle velocity
- **Need satisfaction (aggregate)** → overall coherence / structural integrity
- **Individual needs** → structural distortion (hunger thins/contracts, loneliness fragments/scatters, low stamina dims)
- **Behavior** → movement pattern and structural motif
- **Emotion vectors** → transient visual events (color flash, ripple, dimming, brightening)

### 3.2 Rendering Technology

- **WebGL 2.0** via raw API (no three.js — keep it lean for the overlay)
- **GLSL fragment shaders** for particle rendering, SDF composition, and post-processing
- Particle positions computed in vertex shader or via transform feedback
- Post-processing: glow/bloom pass for the energy field aesthetic
- Alpha-blended output on transparent background (premultiplied alpha for correct compositing)

### 3.3 Particle System Parameters

| Parameter | Range | Driven By |
|-----------|-------|-----------|
| Particle count | 200-800 | Fixed (maybe adjustable in settings for performance) |
| Base color (hue) | 0-360 | `mood_valence`: warm at +1, cool at -1 |
| Color saturation | 0.3-1.0 | Emotion intensity (higher = more saturated) |
| Color brightness | 0.4-1.0 | Stamina satisfaction |
| Coherence radius | 0.2-1.0 (normalized) | Aggregate need satisfaction |
| Oscillation speed | 0.2-3.0 | `mood_arousal`: slow at -1, fast at +1 |
| Particle size | 1-4px | Arousal (larger when aroused) |
| Dispersion factor | 0.0-1.0 | Inverse of coherence; loneliness increases this |
| Glow intensity | 0.3-1.0 | Valence (brighter when positive) |
| Structural motif | enum | Current behavior (see 3.4) |

### 3.4 Behavior → Visual Motif Mapping

Each behavior produces a distinct movement/structure pattern in the particle system:

| Behavior | Movement | Structure | Visual Character |
|----------|----------|-----------|-----------------|
| **idle** | Gentle drift in place, breathing rhythm | Loose sphere, slow rotation | Resting, ambient presence |
| **walk** | Directed linear movement across screen | Elongated in direction of travel, flowing | Purposeful drift |
| **chase** | Fast movement following cursor | Tight excited swarm, trailing particles | Energetic pursuit |
| **sleep** | Stationary | Tight contracted cluster, very slow pulse, dimmed | Minimal, peaceful |
| **relax** | Stationary or very slow drift | Expanded form, slow gentle color cycling | Open, comfortable |
| **attentive** | Stationary, oriented toward chat window | Particles align/focus toward a point | Alert, engaged |

**Behavior transitions**: Crossfade over 0.5-1.0 seconds. A `behavior_transition` float (0→1) blends the old motif's parameters toward the new motif's parameters. During transition, particles may briefly scatter before reorganizing into the new pattern.

### 3.5 Emotion Flash Events

When the backend emits a new emotion vector (via `emotion:new` events in the state payload), the creature produces a transient visual response:

| Emotion Quality | Visual Effect | Duration |
|----------------|---------------|----------|
| Positive valence | Warm color flash, brief expansion | 0.5-1.0s |
| Negative valence | Cool color flash, brief contraction | 0.5-1.0s |
| High arousal | Particle speed spike, scatter-and-reform | 0.3-0.5s |
| Low arousal | Particle slow-down, condensation | 0.5-1.0s |
| High intensity | Stronger effect magnitude | — |
| Low intensity | Subtle, almost imperceptible | — |

These overlay on top of the ongoing behavior motif and decay back to baseline.

### 3.6 Canvas and Sizing

- **Base canvas**: 192x192 logical pixels (midpoint of 128-256 range)
- **Resizable**: user can scale, minimum 96px, maximum 384px
- **DPI-aware**: canvas renders at device pixel ratio (2x on HiDPI) for crisp particles
- **Hit area**: circular region matching the particle cloud's approximate radius, for click detection

---

## 4. Window System

### 4.1 Platform Strategy

- **Linux-first** on Fedora with GNOME/Wayland
- **XWayland mode**: the Tauri app launches with `GDK_BACKEND=x11` to get X11 windowing semantics, enabling reliable transparency, always-on-top, input region shaping, and click-through
- Cross-platform (macOS, Windows) deferred — architecture should not preclude it

### 4.2 Window Inventory

| Window | Type | Lifecycle | Always-on-Top | Transparent |
|--------|------|-----------|---------------|-------------|
| Pet Overlay | Tauri WebviewWindow | Always exists while app runs | Yes | Yes (fully transparent background) |
| Chat | Tauri WebviewWindow | Created on first open, hidden on close, shown on re-open | Yes | No (has background) |
| Dashboard | Tauri WebviewWindow | Created on first open, hidden on close | No | No |
| Wizard | Tauri WebviewWindow | Only on first run or re-config | No | No |

### 4.3 Pet Overlay Window Properties

- **Decorations**: none (frameless)
- **Background**: fully transparent
- **Always on top**: yes
- **Skip taskbar**: yes
- **Resizable**: no (size controlled by creature scale setting)
- **Position**: random on launch, moves with creature during walk/chase behaviors
- **Focus behavior**: does not steal focus on click (X11 window hints: `_NET_WM_STATE_SKIP_PAGER`, override-redirect or utility window type)
- **Click-through mode**: toggleable via hotkey. When active, entire window passes input through. When inactive, hit-testing on the circular creature region.
- **Size**: creature canvas size + margin for glow/particle overflow. For 192px creature: ~256x256 window.

### 4.4 Chat Window Properties

- **Decorations**: custom (frameless with Svelte-rendered title bar for close/minimize)
- **Size**: ~400x600px default, resizable
- **Position**: opens near the pet's current position, edge-aware (flips side if pet is near screen edge). Free-floating after initial placement.
- **State preservation**: Svelte component state persists across hide/show cycles (window is hidden, not destroyed)
- **Always on top**: yes (stays visible when user clicks other apps)

### 4.5 Dashboard Window Properties

- **Decorations**: custom (frameless with Svelte title bar)
- **Size**: ~500x700px default, resizable
- **Position**: opens near pet, free-floating after
- **Tabs**: State, Settings, Worker, Actions (see Section 9)

### 4.6 Multi-Monitor Handling

- MVP: pet confined to the monitor where it spawns
- Screen bounds detected on launch from the primary monitor dimensions
- Edge handling uses these bounds for boundary enforcement

---

## 5. Tauri Rust Layer

The Rust side is the application's backbone. It manages the Python backend process, owns the WebSocket connections, and coordinates all windows.

### 5.1 Responsibilities

1. **Backend process management** — spawn, health-check, restart, kill (Section 15)
2. **WebSocket client** — connect to `/ws` and `/ws/chat`, handle reconnection, parse messages
3. **IPC event emission** — broadcast typed events to all webview windows
4. **Window management** — create/show/hide windows, handle positioning logic
5. **System tray** — icon, menu, click handlers
6. **Hotkey registration** — global hotkeys for passthrough toggle, chat open, show/hide
7. **REST proxy** — Tauri commands that call backend REST endpoints on behalf of webviews (avoids CORS complexity and centralizes the backend URL)

### 5.2 IPC Event Schema

Events emitted from Rust to webviews via `app_handle.emit()`:

#### `soul:state`
Emitted on every `/ws` message from the backend.

```typescript
{
  event_type: "TUI_INITIAL_STATE" | "TUI_REFRESH_DATA",
  payload: {
    system_context: {
      mood: { name: string, valence: number, arousal: number },
      needs: Record<string, { satisfaction: number }>,
      behavior: { name: string | null, active: boolean },
      emotional_state: Array<{
        name: string,
        intensity: number,
        valence: number,
        arousal: number,
        type?: "overall" | "individual"
      }>
    },
    recent_messages: Array<{ role: string, content: string, metadata?: object }>,
    cognitive_summary: string,
    current_model_name: string
  },
  timestamp: string  // ISO 8601
}
```

#### `soul:chat`
Emitted when a chat response arrives.

```typescript
{
  role: "assistant",
  content: string,
  was_task: boolean,
  memories_used: string[]
}
```

#### `soul:connected`
Emitted when WebSocket connection establishes/re-establishes.

```typescript
{
  initial: boolean  // true on first connect, false on reconnect
}
```

#### `soul:disconnected`
Emitted when WebSocket connection drops. Frontend should NOT show this to user — Rust side handles reconnection silently.

```typescript
{
  reason: string,
  will_retry: boolean
}
```

#### `soul:error`
Emitted when an unrecoverable error occurs (backend won't restart, etc).

```typescript
{
  message: string,
  fatal: boolean  // if true, app should show error modal
}
```

### 5.3 Tauri Commands (Frontend → Rust)

Webviews invoke Rust functions via Tauri commands:

| Command | Called By | Parameters | Returns | Action |
|---------|-----------|------------|---------|--------|
| `send_chat_message` | Chat window | `{ message: string }` | `void` | Sends via `/ws/chat` or REST, response comes back as `soul:chat` event |
| `perform_action` | Dashboard/Menu | `{ action: string, params?: object }` | `ActionResponse` | Calls `POST /v1/actions/{name}` |
| `get_actions` | Dashboard | — | `ActionInfo[]` | Calls `GET /v1/actions` |
| `get_memories` | Dashboard | `{ limit?: number }` | `MemoryResponse[]` | Calls `GET /memory/recent` |
| `search_memories` | Dashboard | `{ query: string, limit?: number }` | `MemoryResponse[]` | Calls `GET /memory/search` |
| `get_notes` | Dashboard | `{ tag?: string, sticky?: bool, limit?: number }` | `NoteResponse[]` | Calls `GET /notes` |
| `create_note` | Dashboard | `{ content: string, tags?: string[], sticky?: bool }` | `NoteResponse` | Calls `POST /notes` |
| `update_note` | Dashboard | `{ id: string, content?: string, tags?: string[], sticky?: bool }` | `NoteResponse` | Calls `PUT /notes/{id}` |
| `delete_note` | Dashboard | `{ id: string }` | `void` | Calls `DELETE /notes/{id}` |
| `submit_worker_task` | Dashboard/Chat | `{ task: string, context?: string }` | `{ task_id: string }` | Calls `POST /worker/task` |
| `get_worker_status` | Dashboard | `{ task_id: string }` | `WorkerTaskStatus` | Calls `GET /worker/{id}` |
| `get_settings` | Dashboard/Wizard | — | `SettingsResponse` | Calls `GET /settings` |
| `update_settings` | Dashboard/Wizard | `PetSettings` | `SettingsResponse` | Calls `PUT /settings` |
| `get_state_snapshot` | Any | — | `StateContext` | Calls `GET /state` |
| `toggle_passthrough` | Hotkey/Menu | — | `boolean` | Toggles overlay click-through, returns new state |
| `open_chat` | Hotkey/Menu | — | `void` | Shows/creates chat window |
| `open_dashboard` | Menu | `{ tab?: string }` | `void` | Shows/creates dashboard, optionally to specific tab |
| `dismiss_thought_bubble` | Overlay | — | `void` | Clears current thought bubble |

### 5.4 WebSocket Reconnection Strategy

- On disconnect: immediate first retry, then exponential backoff (1s, 2s, 4s, 8s, max 30s)
- During reconnection: Rust side does NOT emit `soul:disconnected` to webviews (invisible to user)
- On reconnect: request full state snapshot via REST `GET /state` to resync, then emit `soul:connected` + `soul:state`
- After 5 failed retries: attempt to restart the Python backend process
- After backend restart + 3 more failed retries: emit `soul:error` with `fatal: true`

---

## 6. Frontend State Management

### 6.1 Creature State (Pet Overlay)

The overlay window maintains a real-time creature state that interpolates toward backend targets.

```typescript
interface CreatureState {
  // Interpolation targets (set by soul:state events)
  target: {
    mood_valence: number;      // -1 to 1
    mood_arousal: number;      // -1 to 1
    coherence: number;         // 0 to 1, derived from avg need satisfaction
    needs: {
      hunger: number;          // 0 to 1 (satisfaction)
      thirst: number;
      boredom: number;
      loneliness: number;
      stamina: number;
    };
    behavior: string;          // behavior name
  };

  // Current interpolated values (updated each frame)
  current: {
    mood_valence: number;
    mood_arousal: number;
    coherence: number;
    needs: { hunger: number; thirst: number; boredom: number; loneliness: number; stamina: number; };
  };

  // Behavior transition state
  behavior: {
    current: string;
    previous: string | null;
    transition: number;        // 0 to 1 (1 = fully transitioned)
  };

  // Transient emotion flash (decays to zero)
  emotion_flash: {
    valence: number;
    arousal: number;
    intensity: number;
    remaining: number;         // seconds remaining
  };

  // Position and movement
  position: { x: number; y: number };
  velocity: { x: number; y: number };
  screen_bounds: { width: number; height: number };
}
```

### 6.2 Interpolation

Each frame (requestAnimationFrame):

```
dt = time since last frame (capped at 0.05s to handle tab-away)
lerp_factor = 1 - Math.pow(0.01, dt)   // ~0.95 convergence per second

for each float field in current:
    current[field] += lerp_factor * (target[field] - current[field])

if behavior.transition < 1.0:
    behavior.transition = min(1.0, behavior.transition + dt / TRANSITION_DURATION)

if emotion_flash.remaining > 0:
    emotion_flash.remaining -= dt
    emotion_flash.intensity *= (emotion_flash.remaining / original_duration)
```

The `lerp_factor` formula gives smooth exponential decay toward the target regardless of frame rate. At 60fps, values reach ~95% of target within 1 second.

### 6.3 Coherence Derivation

Coherence is not sent by the backend — it's derived on the frontend from need satisfactions:

```typescript
function deriveCoherence(needs: Record<string, number>): number {
  const values = Object.values(needs);
  const avg = values.reduce((a, b) => a + b, 0) / values.length;
  // Weight: low satisfaction hurts coherence more than high satisfaction helps
  const penalty = values.filter(v => v < 0.3).length / values.length;
  return Math.max(0, avg - penalty * 0.3);
}
```

### 6.4 State Update Handler

When `soul:state` arrives:

1. Parse `system_context` from payload
2. Set `target.mood_valence` = `mood.valence`
3. Set `target.mood_arousal` = `mood.arousal`
4. Set `target.needs.*` = need satisfactions
5. Set `target.coherence` = `deriveCoherence(needs)`
6. If `behavior.name` changed from current:
   - Set `behavior.previous` = `behavior.current`
   - Set `behavior.current` = new behavior name
   - Set `behavior.transition` = 0.0
7. If `emotional_state` contains new high-intensity vectors:
   - Set `emotion_flash` with the strongest new emotion's V/A/intensity
   - Set `emotion_flash.remaining` = 0.5-1.0s based on intensity

---

## 7. Pet Overlay Window

### 7.1 HTML Structure

Minimal DOM — almost entirely WebGL:

```html
<!DOCTYPE html>
<html>
<head>
  <style>
    html, body { margin: 0; padding: 0; overflow: hidden; background: transparent; }
    canvas { display: block; }
  </style>
</head>
<body>
  <canvas id="creature"></canvas>
  <div id="thought-bubble-container"></div>
  <div id="context-menu-container"></div>
  <script type="module" src="/src/overlay/main.ts"></script>
</body>
</html>
```

### 7.2 Render Pipeline

1. **Particle update pass**: update particle positions based on behavior motif, time, and noise
2. **Particle render pass**: draw particles as points/sprites with per-particle color/size
3. **Glow pass**: horizontal + vertical Gaussian blur on a downsampled FBO, additive blend back
4. **Composite**: output with premultiplied alpha on transparent background

### 7.3 Movement System

The creature's screen position is managed by a simple movement controller:

```typescript
interface MovementController {
  position: { x: number; y: number };
  target_position: { x: number; y: number } | null;
  speed: number;              // pixels per second, varies by behavior
  screen_bounds: { width: number; height: number; margin: number };
}
```

**Behavior-driven movement:**

| Behavior | Movement Logic |
|----------|---------------|
| idle | No target. Small random drift (±2px/s Perlin noise offset). |
| walk | Pick random target position on screen. Move toward it at moderate speed (~40px/s). On arrival, idle briefly, then pick new target. |
| chase | Target = cursor position (queried via X11). Move toward cursor at high speed (~120px/s). |
| sleep | No movement. Position locked. |
| relax | No movement, or very slow drift (~5px/s). |
| attentive | No movement. Position locked. Particles orient toward chat window position. |

**Edge handling**: hard boundary. When the creature's center reaches `margin` pixels from screen edge, it stops and a new direction/target is chosen. Margin = half the canvas size (so the creature's visual extent stays on screen).

**Window position**: the Tauri overlay window moves to follow the creature's position. The WebGL canvas is always centered in the window. The window position updates each frame during movement.

### 7.4 Hit Testing

The overlay window receives all mouse events (when not in passthrough mode). Hit testing determines if a click lands on the creature:

```typescript
function isHit(mouseX: number, mouseY: number, creature: CreatureState): boolean {
  const dx = mouseX - creature.position.x;
  const dy = mouseY - creature.position.y;
  const distance = Math.sqrt(dx * dx + dy * dy);
  return distance <= CREATURE_HIT_RADIUS;  // ~half the canvas size
}
```

- Click on creature → open right-click menu (left or right click)
- Click outside creature → pass through (window should forward the event to the window below, or use input region shaping to only capture the creature area)

**Input region shaping** (preferred approach on X11): set the window's X11 input region to a circle matching the creature's hit area. This way, clicks outside the creature naturally fall through to the desktop/other windows without any JS hit-testing. Update the input region when the creature moves within the window.

---

## 8. Chat Window

### 8.1 Layout

```
┌──────────────────────────────┐
│  Hephia Chat            — ✕  │  ← Custom title bar (draggable)
├──────────────────────────────┤
│                              │
│  [Pet]: Hey, what's up?      │  ← Message history (scrollable)
│                              │
│  [You]: How are you feeling? │
│                              │
│  [Pet]: I've been thinking   │
│  about our conversation      │
│  yesterday...                │
│                              │
│  ┌────────────────────────┐  │
│  │ Worker: Researching... │  │  ← Worker task inline (if active)
│  │ ████████░░░░ 65%       │  │
│  └────────────────────────┘  │
│                              │
├──────────────────────────────┤
│  Type a message...     [Send]│  ← Input area
└──────────────────────────────┘
```

### 8.2 Message Types

| Type | Rendering | Source |
|------|-----------|--------|
| User message | Right-aligned, styled differently | User input |
| Pet response | Left-aligned | `soul:chat` event |
| System message | Centered, muted styling | Errors, status updates |
| Worker task card | Inline card with progress bar | Worker task submission + polling |

### 8.3 Behavior

- **Opening**: Rust creates/shows the chat window, positioned near the pet with edge-awareness. If pet is in the right half of the screen, chat opens to the left; if near bottom, chat opens above; etc.
- **Closing**: window hides (not destroyed). Svelte component state preserved. On re-open, previous messages still visible.
- **Session fresh start**: on app launch, chat history starts empty in the UI (even though the backend preserves conversation context for the LLM). The user sees a fresh conversation. Backend's conversation context provides continuity for the pet's responses.
- **Sending**: user types message → invokes `send_chat_message` Tauri command → Rust sends to backend → response arrives as `soul:chat` event → chat window renders it.
- **Worker tasks**: when the pet identifies a task (response has `was_task: true`), a worker task card appears inline. The chat window polls `get_worker_status` periodically until completion. On completion, result appears in chat. If chat is closed when task completes, a thought bubble fires on the overlay.
- **Plain text**: no markdown rendering for MVP. Messages displayed as plain text with line breaks preserved.

### 8.4 Edge-Aware Positioning

```typescript
function calculateChatPosition(
  petPos: { x: number; y: number },
  screenBounds: { width: number; height: number },
  chatSize: { width: number; height: number }
): { x: number; y: number } {
  const MARGIN = 20;  // gap between pet and chat
  let x: number, y: number;

  // Horizontal: prefer right of pet, flip to left if near right edge
  if (petPos.x + MARGIN + chatSize.width < screenBounds.width) {
    x = petPos.x + MARGIN;
  } else {
    x = petPos.x - MARGIN - chatSize.width;
  }

  // Vertical: center on pet, clamp to screen
  y = Math.max(0, Math.min(
    petPos.y - chatSize.height / 2,
    screenBounds.height - chatSize.height
  ));

  return { x, y };
}
```

---

## 9. Dashboard Window

### 9.1 Tab Structure

```
┌──────────────────────────────────────┐
│  Hephia Dashboard              — ✕   │
├──────────────────────────────────────┤
│  [State] [Actions] [Worker] [Settings│]
├──────────────────────────────────────┤
│                                      │
│  (Tab content area)                  │
│                                      │
└──────────────────────────────────────┘
```

### 9.2 State Tab

Visual representation of current internal state:

- **Mood**: circular indicator with valence (color) and arousal (animation speed) + text label
- **Needs**: 5 arc/bar indicators arranged in a circle or row. Each shows:
  - Need name
  - Satisfaction percentage (filled arc)
  - Color coding: green (>70%), yellow (30-70%), red (<30%)
- **Behavior**: current behavior name with duration
- **Emotions**: list of active emotion vectors with name + intensity bar
- **Memory activity**: count of recent memories, last echo/ghosting event

All values update in real-time from `soul:state` events.

### 9.3 Actions Tab

Grid or list of available pet actions:

- Each action shows: name, description, cooldown status
- Click to execute (invokes `perform_action` Tauri command)
- Cooldown shown as a greyed-out state with remaining time
- After execution: brief success/failure feedback

### 9.4 Worker Tab

- **Submit task**: text input + submit button
- **Active tasks**: list of pending/running tasks with status
- **Completed tasks**: history of results, expandable
- **Progress**: polling-based updates from `get_worker_status`

### 9.5 Settings Tab

- **Pet name**: text input
- **Pet model**: dropdown from available models
- **Worker model**: dropdown from available models
- **Personality prompt**: textarea
- **Save button**: invokes `update_settings`
- **Advanced section** (collapsible): memory significance threshold, max conversation turns. Labeled as "you probably don't need to change these."

---

## 10. Thought Bubble System

### 10.1 Visual Design

A small floating element near the creature on the overlay window:

```
        ╭──────────────────╮
        │ I was thinking    │
        │ about yesterday...│
        ╰──────────┬───────╯
                   ○
                  ○
                 ●  ← creature
```

- Positioned above or to the side of the creature, edge-aware
- Semi-transparent background with text
- Click to dismiss (or open related window — e.g., clicking a worker completion bubble opens the dashboard worker tab)

### 10.2 Trigger Sources

| Source | Content | On Click |
|--------|---------|----------|
| Introspection | Summary of what the pet reflected on | Dismiss |
| Worker task complete | "Finished: {task summary}" | Open dashboard worker tab |
| Cognitive action | Context-dependent | Dismiss or open chat |

### 10.3 Stacking

- Multiple unattended bubbles stack: only the latest is shown, with a badge showing count (e.g., "3")
- Clicking the badge reveals a small list of all pending notifications
- Bubbles persist until explicitly dismissed by click
- Maximum stack: 10 (oldest auto-dismissed when exceeded)

### 10.4 Implementation

Thought bubbles are Svelte components rendered in the overlay window's DOM layer, positioned above the WebGL canvas via CSS. They don't live in the WebGL render pipeline.

---

## 11. Right-Click Menu

### 11.1 Structure

Context menu appears near the creature on click:

```
┌───────────────┐
│  Chat         │  → opens chat window
│  Dashboard    │  → opens dashboard window
├───────────────┤
│  Actions  ▸   │  → submenu: Feed, Play, Pet, Rest, etc.
│  Notes    ▸   │  → submenu: View Notes, New Note
│  Memories ▸   │  → submenu: Recent, Search
├───────────────┤
│  Passthrough  │  → toggles click-through mode
│  Hide         │  → hides pet (tray only)
│  Quit         │  → kills everything, exits
└───────────────┘
```

### 11.2 Implementation

Svelte component in the overlay window DOM. Positioned at click location, edge-aware. Closes on click-outside or selection.

---

## 12. System Tray

### 12.1 Icon

Static icon for MVP. Simple recognizable glyph (particle cloud silhouette or abstract dot cluster).

### 12.2 Left Click

Toggle pet visibility (show/hide overlay window).

### 12.3 Right-Click Menu

```
┌───────────────┐
│  Show/Hide    │
│  Open Chat    │
│  Passthrough  │  ← checkbox/toggle
├───────────────┤
│  Quit         │
└───────────────┘
```

Simpler subset of the full right-click menu. No nested submenus.

---

## 13. Hotkeys

Fixed bindings for MVP. All global (work regardless of focused window).

| Hotkey | Action |
|--------|--------|
| `Ctrl+Shift+H` | Toggle pet visibility (show/hide) |
| `Ctrl+Shift+C` | Open/focus chat window |
| `Ctrl+Shift+P` | Toggle passthrough mode |

Registered via Tauri's global shortcut API. If a binding conflicts with an existing app, the user can change it via config file (not UI — deferred).

---

## 14. First-Run Wizard

### 14.1 Flow

```
[Welcome] → [Environment] → [API Keys] → [Models] → [Identity] → [Launch]
```

### 14.2 Step Details

#### Step 1: Welcome
- Brief introduction: "Hephia is a desktop companion that remembers, feels, and thinks."
- "Let's get set up" button

#### Step 2: Environment Check
- Auto-detect:
  - Python 3.10+ installation
  - `uv` package manager
  - Project virtual environment
- Status indicators: checkmark (found) or warning (missing)
- For missing components: "Install" button that runs the appropriate command
  - Missing `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh` (with user confirmation)
  - Missing venv/deps: `uv sync`
- "Next" enabled only when all checks pass

#### Step 3: API Keys
- Input fields grouped by provider:
  - Anthropic API Key (`ANTHROPIC_API_KEY`)
  - OpenAI API Key (`OPENAI_API_KEY`) — optional
  - OpenRouter API Key (`OPENROUTER_API_KEY`) — optional
  - Google AI API Key (`GOOGLE_API_KEY`) — optional
  - Local Inference URL (`LOCAL_INFERENCE_BASE_URL`) — optional
- Each with a "test" button that validates the key
- "Skip" option for providers the user doesn't use
- Keys saved to `.env` file in the project directory

#### Step 4: Model Selection
- **Pet Model** (personality/chat): dropdown of available models, filtered by configured providers. Default: `haiku-4.5` (if Anthropic key provided).
- **Worker Model** (tasks): dropdown, default: `opus-4.6`
- Brief description of each model shown on selection
- Validation: warn if selected model's provider has no API key

#### Step 5: Pet Identity
- **Name**: text input, default "Hephia"
- **Personality prompt**: textarea, optional. Placeholder text explaining what this does.
- Preview: "Your pet [Name] will use [Pet Model] for conversation and [Worker Model] for tasks."

#### Step 6: Launch
- Summary of configuration
- "Launch Hephia" button
- Starts the backend, waits for connection, transitions to the pet overlay

### 14.3 Re-Running the Wizard

Accessible from Dashboard → Settings → "Reconfigure" button. Opens the wizard from Step 2 (skips welcome), pre-populated with current values.

### 14.4 Configuration Storage

Wizard writes:
- `.env` file: API keys, model selections, pet name
- `~/.config/hephia/frontend.json`: window preferences, wizard-completed flag

The backend already reads from `.env` and `Config` class. The wizard writes the same format.

---

## 15. Backend Lifecycle Management

### 15.1 Startup Sequence

1. Tauri app launches
2. Check `~/.config/hephia/frontend.json` for `wizard_completed` flag
3. If not completed: show wizard window (Section 14)
4. If completed (or wizard just finished):
   a. Detect Python environment (look for `uv` in PATH, check for `.venv/` in project dir)
   b. Spawn Python backend: `uv run python -m uvicorn core.server:app --host 0.0.0.0 --port 5517` (or equivalent entry point)
   c. Capture stdout/stderr for logging
   d. Wait for backend to become available: poll `GET /state` with 500ms interval, timeout after 30s
   e. Connect WebSocket to `/ws` and `/ws/chat`
   f. Emit `soul:connected` with `initial: true`
   g. Show pet overlay window

### 15.2 Health Monitoring

- Rust side monitors the Python child process (exit code detection)
- WebSocket connection serves as heartbeat (if WS drops, backend may be down)
- On process exit:
  - If exit code 0: clean shutdown, don't restart
  - If exit code non-zero: crash, attempt restart (up to 3 times with 2s delay)
  - After 3 failed restarts: emit `soul:error` with `fatal: true`

### 15.3 Shutdown Sequence

1. User clicks "Quit" (tray or menu)
2. Close all Tauri windows
3. Close WebSocket connections
4. Send SIGTERM to Python process
5. Wait up to 5s for clean shutdown
6. If still running: SIGKILL
7. Exit Tauri app

### 15.4 Environment Detection

```rust
struct BackendEnvironment {
    python_path: PathBuf,      // Path to Python binary (in .venv)
    project_dir: PathBuf,      // Path to hephia project root
    uv_path: Option<PathBuf>,  // Path to uv binary
}

// Detection order:
// 1. Check for .venv/bin/python relative to the Tauri app's resource dir
// 2. Check for uv in PATH, use `uv run` as launcher
// 3. Fall back to system Python (warn user)
```

---

## 16. Error Handling

### 16.1 Error Categories and Display

| Category | Where Shown | Example |
|----------|-------------|---------|
| Chat errors (API failure, invalid key) | Chat window as system message | "Couldn't reach the API. Check your API key in settings." |
| Action errors (cooldown, invalid action) | Dashboard actions tab / brief toast | "Feed is on cooldown (30s remaining)" |
| Backend crash (unrecoverable) | Modal over pet overlay | "Hephia's soul server has stopped. [Restart] [Quit]" |
| Worker task failure | Dashboard worker tab + thought bubble | "Task failed: {error}" |
| WebSocket drop (recoverable) | Hidden from user | Rust handles reconnection silently |
| First-run setup errors | Wizard step inline | "Python not found. [Install] [Browse]" |

### 16.2 Fatal Error Modal

When `soul:error` with `fatal: true` is emitted:

```
┌───────────────────────────────────┐
│                                   │
│   Hephia has lost connection      │
│   to its soul server.             │
│                                   │
│   The backend process exited      │
│   unexpectedly and could not      │
│   be restarted.                   │
│                                   │
│   [Try Again]    [Quit]           │
│                                   │
└───────────────────────────────────┘
```

Rendered as a Svelte component in the overlay window DOM, above the WebGL canvas with a semi-transparent backdrop.

---

## 17. Dev Workflow & Shader Playground

### 17.1 Development Loop

| Change | Iteration Speed |
|--------|----------------|
| Svelte/CSS (UI components) | Vite HMR, <1s |
| WebGL shaders (GLSL) | Vite HMR on shader file change, <1s |
| TypeScript (state management, logic) | Vite HMR, <1s |
| Rust (Tauri commands, IPC) | Cargo rebuild, ~5-10s |
| Python (backend) | Uvicorn `--reload`, ~2-3s |

### 17.2 Shader Playground

A standalone HTML page (`playground/index.html`) that loads the identical particle system with mock data:

```
playground/
├── index.html          # Self-contained playground page
├── mock-state.ts       # Mock state generator (random walks, presets)
└── controls.ts         # dat.gui or custom sliders for all shader uniforms
```

**Features:**
- All shader code imported from the same source as the real app (shared `creature/` module)
- Slider controls for every state dimension: mood V/A, each need satisfaction, behavior selection, emotion flash trigger
- Preset buttons: "happy idle", "stressed chase", "sleepy", "lonely walk", "excited"
- Real-time uniform inspector showing current values
- Canvas with transparent background (checkered pattern behind to verify alpha)
- No Tauri dependency — runs with `npx vite` standalone
- Can optionally connect to a running soul server WebSocket for live data

**Guaranteed parity**: the playground imports the same WebGL initialization, shader compilation, and render functions as the pet overlay. No rendering code is duplicated — both the overlay's `main.ts` and the playground's `index.html` import from `creature/`.

### 17.3 Shader Module Structure

```typescript
// creature/index.ts — public API used by both overlay and playground
export function createCreatureRenderer(canvas: HTMLCanvasElement): CreatureRenderer;
export interface CreatureRenderer {
  updateState(state: CreatureState): void;
  render(dt: number): void;
  resize(width: number, height: number, dpr: number): void;
  destroy(): void;
}
```

---

## 18. MVP Scope

### 18.1 Must Have (Launch)

| Feature | Section |
|---------|---------|
| Transparent overlay window with particle creature | 4, 7 |
| WebGL particle system with mood/need/behavior mapping | 3 |
| Creature movement: idle, walk (random targets), sleep, relax | 7.3 |
| Right-click menu with Chat, Dashboard, Actions, Passthrough, Quit | 11 |
| Chat window: send messages, receive responses, plain text | 8 |
| Dashboard: State tab with need bars and mood indicator | 9.2 |
| Dashboard: Settings tab with model/name/personality config | 9.5 |
| Dashboard: Actions tab with all defined actions | 9.3 |
| System tray: show/hide, open chat, quit | 12 |
| Hotkeys: passthrough, chat, show/hide | 13 |
| Thought bubbles for introspection + worker completion | 10 |
| Rust WebSocket client with reconnection | 5.4 |
| Backend process management (spawn, monitor, restart, kill) | 15 |
| First-run wizard | 14 |
| Shader playground | 17.2 |
| Interpolation-based state rendering | 6.2 |

### 18.2 Nice to Have (Launch)

| Feature | Section |
|---------|---------|
| Chase behavior (cursor following) | 7.3 |
| Attentive behavior (orient toward chat window) | 3.4 |
| Worker tab in dashboard | 9.4 |
| Notes access from dashboard | — |
| Memory browsing in dashboard | — |

### 18.3 Not in MVP

See Section 19.

---

## 19. Deferred Features

| Feature | When | Notes |
|---------|------|-------|
| Window interaction (sit on title bars) | Post-MVP | Requires window enumeration, complex positioning |
| Layer 3.4 new behaviors (eat, drink, play, groom, stretch, nap, explore) | With internal state Layer 3 work | Backend behaviors must exist first |
| Sound/audio | Post-MVP | Architecture supports it (event-driven) |
| User-configurable hotkeys | Post-MVP | Config file, then settings UI |
| Custom user models config UI | Post-MVP | Backend already supports `models.json` |
| Multi-monitor awareness | Post-MVP | Pick monitor on launch for now |
| Visual click-through indicator | Post-MVP | Tray icon change possible |
| Position persistence | Post-MVP | Random spawn is fine |
| Markdown/rich text in chat | Post-MVP | Plain text first |
| Cursor hover acknowledgment | Post-MVP | Subtle particle lean |
| macOS / Windows support | Post-MVP | Architecture doesn't preclude it |

---

## 20. Project Structure

```
hephia/
├── frontend_spec.md                    # This document
├── pet_implementation_spec.md          # Backend spec
├── internal_update_spec.md             # Internal state spec
│
├── body/                               # Tauri frontend (this project)
│   ├── src-tauri/                      # Rust side
│   │   ├── Cargo.toml
│   │   ├── tauri.conf.json             # Tauri config (windows, permissions, etc.)
│   │   ├── icons/                      # App + tray icons
│   │   └── src/
│   │       ├── main.rs                 # Entry point
│   │       ├── backend.rs              # Python process manager
│   │       ├── websocket.rs            # WebSocket client + reconnection
│   │       ├── ipc.rs                  # IPC event definitions + emission
│   │       ├── windows.rs              # Window creation, positioning, management
│   │       ├── tray.rs                 # System tray setup
│   │       └── commands.rs             # Tauri command handlers (REST proxies)
│   │
│   ├── src/                            # Web frontend (Svelte + TypeScript)
│   │   ├── overlay/                    # Pet overlay window
│   │   │   ├── index.html
│   │   │   ├── main.ts                 # Overlay entry: init WebGL, listen IPC
│   │   │   ├── Overlay.svelte          # Root component (canvas + bubbles + menu)
│   │   │   ├── ThoughtBubble.svelte
│   │   │   └── ContextMenu.svelte
│   │   │
│   │   ├── chat/                       # Chat window
│   │   │   ├── index.html
│   │   │   ├── main.ts
│   │   │   ├── Chat.svelte             # Root chat component
│   │   │   ├── MessageList.svelte
│   │   │   ├── MessageInput.svelte
│   │   │   └── WorkerTaskCard.svelte
│   │   │
│   │   ├── dashboard/                  # Dashboard window
│   │   │   ├── index.html
│   │   │   ├── main.ts
│   │   │   ├── Dashboard.svelte        # Root with tab navigation
│   │   │   ├── StateTab.svelte
│   │   │   ├── ActionsTab.svelte
│   │   │   ├── WorkerTab.svelte
│   │   │   └── SettingsTab.svelte
│   │   │
│   │   ├── wizard/                     # First-run wizard
│   │   │   ├── index.html
│   │   │   ├── main.ts
│   │   │   ├── Wizard.svelte
│   │   │   ├── StepWelcome.svelte
│   │   │   ├── StepEnvironment.svelte
│   │   │   ├── StepApiKeys.svelte
│   │   │   ├── StepModels.svelte
│   │   │   ├── StepIdentity.svelte
│   │   │   └── StepLaunch.svelte
│   │   │
│   │   ├── creature/                   # WebGL particle system (shared)
│   │   │   ├── index.ts                # Public API: createCreatureRenderer()
│   │   │   ├── renderer.ts             # WebGL setup, render loop
│   │   │   ├── particles.ts            # Particle system simulation
│   │   │   ├── movement.ts             # Screen movement controller
│   │   │   ├── shaders/
│   │   │   │   ├── particle.vert       # Vertex shader
│   │   │   │   ├── particle.frag       # Fragment shader
│   │   │   │   └── glow.frag           # Post-processing glow
│   │   │   └── behaviors/              # Behavior-specific motif parameters
│   │   │       ├── idle.ts
│   │   │       ├── walk.ts
│   │   │       ├── chase.ts
│   │   │       ├── sleep.ts
│   │   │       ├── relax.ts
│   │   │       └── attentive.ts
│   │   │
│   │   ├── state/                      # Frontend state management
│   │   │   ├── creature-state.ts       # CreatureState + interpolation
│   │   │   ├── ipc-listener.ts         # Tauri IPC event subscriptions
│   │   │   └── stores.ts              # Svelte stores for shared state
│   │   │
│   │   ├── lib/                        # Shared utilities
│   │   │   ├── tauri-commands.ts       # Typed wrappers for Tauri invoke()
│   │   │   ├── types.ts               # Shared TypeScript types
│   │   │   └── utils.ts               # Edge-aware positioning, etc.
│   │   │
│   │   └── styles/                     # Shared styles
│   │       ├── global.css
│   │       └── theme.ts               # Color tokens, spacing, etc.
│   │
│   ├── playground/                     # Standalone shader playground
│   │   ├── index.html
│   │   ├── main.ts                     # Playground entry with controls
│   │   ├── mock-state.ts              # State presets + random walks
│   │   └── controls.ts               # Slider UI for all uniforms
│   │
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts                  # Multi-page config (overlay, chat, dashboard, wizard)
│   └── svelte.config.js
│
├── core/                               # Python backend (existing)
│   └── ...
├── internal/                           # Internal state (existing)
│   └── ...
├── mind/                               # Mind layer (existing)
│   └── ...
├── config.py                           # Backend config (existing)
├── data/                               # Runtime data (gitignored)
└── .env                                # API keys (gitignored, written by wizard)
```

### 20.1 Vite Multi-Page Configuration

Each window (overlay, chat, dashboard, wizard) is a separate Vite entry point with its own `index.html`. Vite's multi-page mode builds them all from a single config:

```typescript
// vite.config.ts
import { resolve } from 'path';
import { defineConfig } from 'vite';
import { svelte } from '@sveltejs/vite-plugin-svelte';

export default defineConfig({
  plugins: [svelte()],
  build: {
    rollupOptions: {
      input: {
        overlay: resolve(__dirname, 'src/overlay/index.html'),
        chat: resolve(__dirname, 'src/chat/index.html'),
        dashboard: resolve(__dirname, 'src/dashboard/index.html'),
        wizard: resolve(__dirname, 'src/wizard/index.html'),
      },
    },
  },
});
```

### 20.2 Tauri Window Configuration

Each window defined in `tauri.conf.json` with appropriate properties:

```jsonc
{
  "app": {
    "windows": [
      {
        "label": "overlay",
        "url": "/src/overlay/index.html",
        "transparent": true,
        "decorations": false,
        "alwaysOnTop": true,
        "skipTaskbar": true,
        "resizable": false,
        "width": 256,
        "height": 256,
        "visible": false  // shown after backend connects
      }
    ]
    // Chat, dashboard, wizard created dynamically via Rust
  }
}
```

---

*End of specification.*
