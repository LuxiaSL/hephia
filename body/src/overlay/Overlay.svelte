<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { invoke } from '@tauri-apps/api/core';
  import { setupIPCListeners } from '$state/ipc-listener';
  import { systemContext, thoughtBubbles, fatalError, addThoughtBubble, dismissTopThoughtBubble } from '$state/stores';
  import { createCreatureState, applyStateUpdate, tickCreatureState } from '$state/creature-state';
  import { createCreatureRenderer, type CreatureRenderer } from '$creature/index';
  import { createMovementController } from '$state/movement';
  import { getCursorPosition, showContextMenu } from '$lib/tauri-commands';
  import type { SoulStatePayload, ThoughtBubblePayload, ErrorPayload } from '$lib/types';

  let canvas: HTMLCanvasElement;
  let unlisten: (() => void) | null = null;
  let loopTimerId: number = 0;
  let lastTime = 0;
  let renderer: CreatureRenderer | null = null;

  const CANVAS_SIZE = 192; // logical pixels
  const dpr = window.devicePixelRatio || 1;

  // Movement controller
  const movement = createMovementController(CANVAS_SIZE);
  let moveInFlight = false;
  let lastMoveX = -1;
  let lastMoveY = -1;
  let cursorPollInFlight = false;

  // Creature state with interpolation
  let creatureState = createCreatureState(
    window.screen.width,
    window.screen.height,
  );

  // Reactive thought bubble state
  let bubbles: ThoughtBubblePayload[] = [];
  let showError = false;
  let errorMessage = '';

  thoughtBubbles.subscribe((v) => (bubbles = v));
  fatalError.subscribe((v) => {
    if (v?.fatal) {
      showError = true;
      errorMessage = v.message;
    }
  });

  onMount(async () => {
    // Initialize WebGL creature renderer
    try {
      renderer = createCreatureRenderer(canvas);
      renderer.resize(CANVAS_SIZE, CANVAS_SIZE, dpr);
    } catch (e) {
      console.error('Failed to initialize creature renderer:', e);
    }

    // Position overlay window at initial creature position
    try {
      const ix = Math.round(creatureState.position.x - CANVAS_SIZE / 2);
      const iy = Math.round(creatureState.position.y - CANVAS_SIZE / 2);
      await invoke('move_overlay', { x: ix, y: iy });
    } catch {
      // Will retry from render loop once window is visible
    }

    // Set up IPC listeners
    unlisten = await setupIPCListeners({
      onState: (payload: SoulStatePayload) => {
        const ctx = payload.payload?.system_context;
        if (ctx) {
          systemContext.set(ctx);
          applyStateUpdate(creatureState, ctx);
        }
      },
      onThought: (thought: ThoughtBubblePayload) => {
        addThoughtBubble(thought);
      },
      onThoughtDismiss: () => {
        dismissTopThoughtBubble();
      },
      onConnected: () => {
        // Window was just shown — unstick movement and force position sync
        moveInFlight = false;
        lastMoveX = -1;
        lastMoveY = -1;
      },
      onError: (error: ErrorPayload) => {
        if (error.fatal) {
          fatalError.set(error);
        }
      },
    });

    // Start render loop — use setInterval instead of requestAnimationFrame
    // so the pet keeps animating when the overlay loses focus (alt-tab, etc.)
    lastTime = performance.now();
    loopTimerId = window.setInterval(() => renderLoop(performance.now()), 16);
  });

  onDestroy(() => {
    if (unlisten) unlisten();
    if (loopTimerId) window.clearInterval(loopTimerId);
    if (renderer) renderer.destroy();
  });

  function renderLoop(now: number) {
    const dt = Math.min((now - lastTime) / 1000, 0.05);
    lastTime = now;

    // Advance state interpolation
    tickCreatureState(creatureState, dt);

    // Update movement (pure computation — updates position + velocity)
    movement.tick(creatureState, dt);

    // Move overlay window to follow creature position
    moveWindowIfNeeded();

    // Poll cursor during chase behavior (throttled to 10 Hz)
    if (creatureState.behavior.current === 'chase' && movement.shouldPollCursor(dt)) {
      pollCursorPosition();
    }

    // Feed state to renderer and draw
    if (renderer) {
      renderer.updateState(creatureState);
      renderer.render(dt);
    }

    // loop continues via setInterval — no need to re-schedule
  }

  /** Move the Tauri overlay window via Rust command. Only caches position on success. */
  function moveWindowIfNeeded(): void {
    const wx = Math.round(creatureState.position.x - CANVAS_SIZE / 2);
    const wy = Math.round(creatureState.position.y - CANVAS_SIZE / 2);

    if (moveInFlight || (wx === lastMoveX && wy === lastMoveY)) return;

    moveInFlight = true;

    invoke('move_overlay', { x: wx, y: wy })
      .then(() => {
        lastMoveX = wx;
        lastMoveY = wy;
      })
      .catch(() => {})
      .finally(() => { moveInFlight = false; });
  }

  /** Query cursor screen position from Rust for chase behavior. */
  async function pollCursorPosition(): Promise<void> {
    if (cursorPollInFlight) return;
    cursorPollInFlight = true;
    try {
      const [physX, physY] = await getCursorPosition();
      // Convert physical pixels to logical
      movement.updateCursorPosition(physX / dpr, physY / dpr);
    } catch {
      // Cursor unavailable — chase degrades to orbit-in-place
    }
    cursorPollInFlight = false;
  }

  function handleBubbleClick() {
    dismissTopThoughtBubble();
  }

  function handleContextMenu(e: MouseEvent) {
    e.preventDefault();
    showContextMenu().catch(() => {});
  }
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="overlay-root" oncontextmenu={handleContextMenu}>
  <canvas bind:this={canvas} class="creature-canvas"></canvas>

  <!-- Thought bubbles -->
  {#if bubbles.length > 0}
    <div class="thought-bubble" onclick={handleBubbleClick} role="button" tabindex="0" onkeydown={(e) => e.key === 'Enter' && handleBubbleClick()}>
      <div class="bubble-content">{bubbles[0].content}</div>
      {#if bubbles.length > 1}
        <div class="bubble-count">{bubbles.length}</div>
      {/if}
      <div class="bubble-tail"></div>
    </div>
  {/if}

  <!-- Fatal error modal -->
  {#if showError}
    <div class="error-overlay">
      <div class="error-modal">
        <div class="error-title">Hephia has lost connection</div>
        <div class="error-message">{errorMessage}</div>
        <div class="error-actions">
          <button class="btn" onclick={() => window.location.reload()}>Try Again</button>
          <button class="btn" onclick={() => { showError = false; }}>Dismiss</button>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .overlay-root {
    width: 192px;
    height: 192px;
    position: relative;
  }

  .creature-canvas {
    width: 192px;
    height: 192px;
    display: block;
    background: transparent;
  }

  .thought-bubble {
    position: absolute;
    top: -80px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(18, 18, 26, 0.95);
    border: 1px solid rgba(99, 102, 241, 0.4);
    border-radius: 12px;
    padding: 8px 12px;
    max-width: 200px;
    color: #e0e0e8;
    font-size: 12px;
    cursor: pointer;
    backdrop-filter: blur(8px);
  }

  .bubble-content {
    line-height: 1.4;
  }

  .bubble-count {
    position: absolute;
    top: -8px;
    right: -8px;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: #6366f1;
    color: white;
    font-size: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
  }

  .bubble-tail {
    position: absolute;
    bottom: -6px;
    left: 50%;
    transform: translateX(-50%);
    width: 0;
    height: 0;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-top: 6px solid rgba(18, 18, 26, 0.95);
  }

  .error-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .error-modal {
    background: #12121a;
    border: 1px solid #2a2a38;
    border-radius: 12px;
    padding: 24px;
    max-width: 320px;
    text-align: center;
  }

  .error-title {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 8px;
  }

  .error-message {
    font-size: 13px;
    color: #8888a0;
    margin-bottom: 16px;
  }

  .error-actions {
    display: flex;
    gap: 8px;
    justify-content: center;
  }

  .btn {
    padding: 6px 16px;
    border: 1px solid #2a2a38;
    border-radius: 8px;
    background: #1a1a25;
    color: #e0e0e8;
    cursor: pointer;
    font-size: 13px;
  }

  .btn:hover {
    background: #22222f;
  }
</style>
