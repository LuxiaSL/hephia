/**
 * WebGL 2 creature renderer.
 *
 * Renders the particle cloud creature with bloom post-processing.
 * Consumes CreatureState (interpolated targets from the state system)
 * and drives the particle simulation + GPU rendering pipeline.
 *
 * Pipeline:
 *   1. Update particle positions (CPU)
 *   2. Render particles to scene FBO
 *   3. Downsample + 2-pass Gaussian blur → bloom FBO
 *   4. Composite scene + bloom to screen
 */

import { createProgram, createBuffer, createFBO } from './gl-utils';
import {
  PARTICLE_VERT,
  PARTICLE_FRAG,
  FULLSCREEN_VERT,
  BLUR_FRAG,
  COMPOSITE_FRAG,
} from './shaders';
import { ParticleSystem, type SimulationParams, type RenderParams } from './particles';
import { getMotif, blendMotifs } from './behaviors';
import type { CreatureState } from '$state/creature-state';
import { valenceToHue, hslToRgb, clamp, mapRange } from '$lib/utils';

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

export interface CreatureRenderer {
  /** Push new creature state (call when soul:state arrives or each frame). */
  updateState(state: CreatureState): void;
  /** Advance simulation and render one frame. dt in seconds. */
  render(dt: number): void;
  /** Resize the canvas (logical w/h + device pixel ratio). */
  resize(width: number, height: number, dpr: number): void;
  /** Release all GPU resources. */
  destroy(): void;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_PARTICLE_COUNT = 400;
const BLOOM_SCALE = 0.25; // bloom FBO is 1/4 resolution
const BLOOM_STRENGTH = 0.6;
const BLUR_RADIUS = 1.5;

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export function createCreatureRenderer(
  canvas: HTMLCanvasElement,
  particleCount: number = DEFAULT_PARTICLE_COUNT,
): CreatureRenderer {
  const gl = canvas.getContext('webgl2', {
    alpha: true,
    premultipliedAlpha: true,
    antialias: false,
    preserveDrawingBuffer: false,
  });

  if (!gl) throw new Error('WebGL 2 not available');

  // Check float texture support (needed for bloom FBOs)
  const floatExt = gl.getExtension('EXT_color_buffer_float');
  const useBloom = !!floatExt;

  // --- Shader programs ---
  const particleProgram = createProgram(gl, PARTICLE_VERT, PARTICLE_FRAG);
  let blurProgram: WebGLProgram | null = null;
  let compositeProgram: WebGLProgram | null = null;

  if (useBloom) {
    blurProgram = createProgram(gl, FULLSCREEN_VERT, BLUR_FRAG);
    compositeProgram = createProgram(gl, FULLSCREEN_VERT, COMPOSITE_FRAG);
  }

  // --- Particle uniform locations ---
  const pu = {
    time: gl.getUniformLocation(particleProgram, 'u_time'),
    baseColor: gl.getUniformLocation(particleProgram, 'u_baseColor'),
    flashColor: gl.getUniformLocation(particleProgram, 'u_flashColor'),
    flashIntensity: gl.getUniformLocation(particleProgram, 'u_flashIntensity'),
    glowIntensity: gl.getUniformLocation(particleProgram, 'u_glowIntensity'),
  };

  // --- Blur uniform locations ---
  let bu: { texture: WebGLUniformLocation | null; direction: WebGLUniformLocation | null; radius: WebGLUniformLocation | null } | null = null;
  if (blurProgram) {
    bu = {
      texture: gl.getUniformLocation(blurProgram, 'u_texture'),
      direction: gl.getUniformLocation(blurProgram, 'u_direction'),
      radius: gl.getUniformLocation(blurProgram, 'u_radius'),
    };
  }

  // --- Composite uniform locations ---
  let cu: { scene: WebGLUniformLocation | null; bloom: WebGLUniformLocation | null; bloomStrength: WebGLUniformLocation | null } | null = null;
  if (compositeProgram) {
    cu = {
      scene: gl.getUniformLocation(compositeProgram, 'u_scene'),
      bloom: gl.getUniformLocation(compositeProgram, 'u_bloom'),
      bloomStrength: gl.getUniformLocation(compositeProgram, 'u_bloomStrength'),
    };
  }

  // --- Particle system ---
  const particles = new ParticleSystem(particleCount);

  // --- Vertex buffer + VAO ---
  const vbo = createBuffer(gl);
  const vao = gl.createVertexArray()!;

  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);

  const stride = 5 * 4; // x, y, size, alpha, phase
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, stride, 0);       // a_position
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 1, gl.FLOAT, false, stride, 2 * 4);   // a_size
  gl.enableVertexAttribArray(2);
  gl.vertexAttribPointer(2, 1, gl.FLOAT, false, stride, 3 * 4);   // a_alpha
  gl.enableVertexAttribArray(3);
  gl.vertexAttribPointer(3, 1, gl.FLOAT, false, stride, 4 * 4);   // a_phase
  gl.bindVertexArray(null);

  // --- Empty VAO for fullscreen triangle draws ---
  const emptyVao = gl.createVertexArray()!;

  // --- FBOs for bloom ---
  let physW = canvas.width;
  let physH = canvas.height;
  let sceneFBO: ReturnType<typeof createFBO> | null = null;
  let bloomFBO1: ReturnType<typeof createFBO> | null = null;
  let bloomFBO2: ReturnType<typeof createFBO> | null = null;

  function rebuildFBOs() {
    // Clean old
    if (sceneFBO) {
      gl.deleteFramebuffer(sceneFBO.fbo);
      gl.deleteTexture(sceneFBO.texture);
    }
    if (bloomFBO1) {
      gl.deleteFramebuffer(bloomFBO1.fbo);
      gl.deleteTexture(bloomFBO1.texture);
    }
    if (bloomFBO2) {
      gl.deleteFramebuffer(bloomFBO2.fbo);
      gl.deleteTexture(bloomFBO2.texture);
    }

    if (useBloom) {
      sceneFBO = createFBO(gl, physW, physH);
      const bw = Math.max(1, Math.floor(physW * BLOOM_SCALE));
      const bh = Math.max(1, Math.floor(physH * BLOOM_SCALE));
      bloomFBO1 = createFBO(gl, bw, bh);
      bloomFBO2 = createFBO(gl, bw, bh);
    }
  }
  rebuildFBOs();

  // --- Renderer state ---
  let time = 0;
  let state: CreatureState | null = null;
  let dpr = 1;

  // --- Helpers ---

  function deriveSimParams(): SimulationParams {
    if (!state) {
      return {
        coherence: 0.5,
        speed: 1.0,
        motif: getMotif('idle'),
        driftAngle: 0,
        emotionFlash: { valence: 0, arousal: 0, intensity: 0 },
      };
    }

    const c = state.current;
    const b = state.behavior;

    // Blend motifs during behavior transitions
    let motif = getMotif(b.current);
    if (b.previous && b.transition < 1.0) {
      const prev = getMotif(b.previous);
      motif = blendMotifs(prev, motif, b.transition);
    }

    // Drift angle from velocity
    const vx = state.velocity.x;
    const vy = state.velocity.y;
    const driftAngle =
      Math.abs(vx) + Math.abs(vy) > 0.1
        ? Math.atan2(vy, vx)
        : 0;

    return {
      coherence: c.coherence,
      speed: 1.0 + c.mood_arousal * 1.5,
      motif,
      driftAngle,
      emotionFlash: {
        valence: state.emotion_flash.valence,
        arousal: state.emotion_flash.arousal,
        intensity: state.emotion_flash.intensity,
      },
    };
  }

  function deriveRenderParams(): RenderParams {
    const simP = deriveSimParams();
    return {
      dpr,
      arousal: state ? state.current.mood_arousal : 0,
      coherence: state ? state.current.coherence : 0.5,
      motif: simP.motif,
    };
  }

  function deriveBaseColor(): [number, number, number] {
    if (!state) return [0.5, 0.5, 0.7];

    const valence = state.current.mood_valence;
    const stamina = state.current.needs.stamina;
    const hue = valenceToHue(valence);
    const saturation = clamp(0.5 + Math.abs(valence) * 0.3, 0.3, 0.9);
    const lightness = clamp(0.45 + stamina * 0.25, 0.35, 0.75);

    return hslToRgb(hue, saturation, lightness);
  }

  function deriveFlashColor(): [number, number, number] {
    if (!state || state.emotion_flash.intensity <= 0) return [0, 0, 0];

    const v = state.emotion_flash.valence;
    if (v >= 0) {
      // Warm flash: gold/amber
      return hslToRgb(45, 0.9, 0.7);
    } else {
      // Cool flash: blue/purple
      return hslToRgb(250, 0.8, 0.5);
    }
  }

  function deriveGlowIntensity(): number {
    if (!state) return 0.5;
    // Brighter glow when positive valence, dimmer when negative
    return clamp(0.4 + state.current.mood_valence * 0.4, 0.2, 0.9);
  }

  // --- Draw helpers ---

  function drawParticles() {
    gl.useProgram(particleProgram);

    gl.uniform1f(pu.time, time);

    const [r, g, b] = deriveBaseColor();
    gl.uniform3f(pu.baseColor, r, g, b);

    const [fr, fg, fb] = deriveFlashColor();
    gl.uniform3f(pu.flashColor, fr, fg, fb);

    const flashI = state ? state.emotion_flash.intensity : 0;
    gl.uniform1f(pu.flashIntensity, flashI);
    gl.uniform1f(pu.glowIntensity, deriveGlowIntensity());

    gl.bindVertexArray(vao);
    gl.drawArrays(gl.POINTS, 0, particles.count);
    gl.bindVertexArray(null);
  }

  function drawFullscreenTriangle() {
    gl.bindVertexArray(emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }

  // --- Public API ---

  return {
    updateState(newState: CreatureState) {
      state = newState;
    },

    render(dt: number) {
      time += dt;

      // 1. Simulate particles
      const simParams = deriveSimParams();
      particles.update(dt, time, simParams);

      // 2. Build render data and upload
      const renderParams = deriveRenderParams();
      const renderData = particles.buildRenderData(renderParams);
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, renderData, gl.DYNAMIC_DRAW);

      // Setup blending
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

      if (useBloom && sceneFBO && bloomFBO1 && bloomFBO2 && blurProgram && compositeProgram && bu && cu) {
        // --- Full bloom pipeline ---

        // 2a. Render particles to scene FBO
        gl.bindFramebuffer(gl.FRAMEBUFFER, sceneFBO.fbo);
        gl.viewport(0, 0, physW, physH);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        drawParticles();

        // 2b. Horizontal blur: scene → bloomFBO1
        const bw = Math.max(1, Math.floor(physW * BLOOM_SCALE));
        const bh = Math.max(1, Math.floor(physH * BLOOM_SCALE));

        gl.useProgram(blurProgram);
        gl.bindFramebuffer(gl.FRAMEBUFFER, bloomFBO1.fbo);
        gl.viewport(0, 0, bw, bh);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, sceneFBO.texture);
        gl.uniform1i(bu.texture, 0);
        gl.uniform2f(bu.direction, 1.0 / bw, 0.0);
        gl.uniform1f(bu.radius, BLUR_RADIUS);
        drawFullscreenTriangle();

        // 2c. Vertical blur: bloomFBO1 → bloomFBO2
        gl.bindFramebuffer(gl.FRAMEBUFFER, bloomFBO2.fbo);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.bindTexture(gl.TEXTURE_2D, bloomFBO1.texture);
        gl.uniform2f(bu.direction, 0.0, 1.0 / bh);
        drawFullscreenTriangle();

        // 2d. Composite: scene + bloom → screen
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, physW, physH);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(compositeProgram);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, sceneFBO.texture);
        gl.uniform1i(cu.scene, 0);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, bloomFBO2.texture);
        gl.uniform1i(cu.bloom, 1);
        gl.uniform1f(cu.bloomStrength, BLOOM_STRENGTH);
        drawFullscreenTriangle();
      } else {
        // --- No bloom: render particles directly to screen ---
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, physW, physH);
        gl.clearColor(0, 0, 0, 0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        drawParticles();
      }
    },

    resize(width: number, height: number, newDpr: number) {
      dpr = newDpr;
      physW = Math.floor(width * dpr);
      physH = Math.floor(height * dpr);
      canvas.width = physW;
      canvas.height = physH;
      rebuildFBOs();
    },

    destroy() {
      gl.deleteProgram(particleProgram);
      if (blurProgram) gl.deleteProgram(blurProgram);
      if (compositeProgram) gl.deleteProgram(compositeProgram);
      gl.deleteBuffer(vbo);
      gl.deleteVertexArray(vao);
      gl.deleteVertexArray(emptyVao);

      if (sceneFBO) {
        gl.deleteFramebuffer(sceneFBO.fbo);
        gl.deleteTexture(sceneFBO.texture);
      }
      if (bloomFBO1) {
        gl.deleteFramebuffer(bloomFBO1.fbo);
        gl.deleteTexture(bloomFBO1.texture);
      }
      if (bloomFBO2) {
        gl.deleteFramebuffer(bloomFBO2.fbo);
        gl.deleteTexture(bloomFBO2.texture);
      }
    },
  };
}
