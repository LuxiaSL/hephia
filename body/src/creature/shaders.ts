/**
 * GLSL shader sources for the particle creature renderer.
 *
 * All shaders target WebGL 2 (GLSL ES 300).
 */

// ---------------------------------------------------------------------------
// Particle rendering
// ---------------------------------------------------------------------------

export const PARTICLE_VERT = /* glsl */ `#version 300 es
precision highp float;

layout(location = 0) in vec2 a_position;
layout(location = 1) in float a_size;
layout(location = 2) in float a_alpha;
layout(location = 3) in float a_phase;

uniform float u_time;

out float v_alpha;
out float v_phase;

void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    gl_PointSize = a_size;
    v_alpha = a_alpha;
    v_phase = a_phase;
}
`;

export const PARTICLE_FRAG = /* glsl */ `#version 300 es
precision highp float;

in float v_alpha;
in float v_phase;

uniform vec3 u_baseColor;
uniform vec3 u_flashColor;
uniform float u_flashIntensity;
uniform float u_glowIntensity;
uniform float u_time;

out vec4 fragColor;

void main() {
    vec2 coord = gl_PointCoord - 0.5;
    float dist = length(coord) * 2.0;

    // Discard corners of the point quad
    if (dist > 1.0) discard;

    // Tight bright core
    float core = exp(-dist * dist * 8.0);

    // Wider soft glow envelope
    float glow = exp(-dist * dist * 2.0) * u_glowIntensity * 0.4;

    float alpha = (core * 0.7 + glow) * v_alpha;

    // Per-particle color shimmer driven by phase
    float shimmer = sin(u_time * 1.5 + v_phase * 6.283185) * 0.08 + 1.0;

    // Blend base color toward flash color during emotion events
    vec3 color = mix(u_baseColor, u_flashColor, u_flashIntensity) * shimmer;

    // Premultiplied alpha for correct compositing on transparent background
    fragColor = vec4(color * alpha, alpha);
}
`;

// ---------------------------------------------------------------------------
// Bloom post-processing (fullscreen quad passes)
// ---------------------------------------------------------------------------

export const FULLSCREEN_VERT = /* glsl */ `#version 300 es
precision highp float;

// Fullscreen triangle trick: 3 vertices, no buffer needed
out vec2 v_uv;

void main() {
    // Generate fullscreen triangle from vertex ID
    float x = float((gl_VertexID & 1) << 2) - 1.0;
    float y = float((gl_VertexID & 2) << 1) - 1.0;
    v_uv = vec2(x, y) * 0.5 + 0.5;
    gl_Position = vec4(x, y, 0.0, 1.0);
}
`;

export const BLUR_FRAG = /* glsl */ `#version 300 es
precision highp float;

in vec2 v_uv;

uniform sampler2D u_texture;
uniform vec2 u_direction;  // (1/w, 0) for horizontal, (0, 1/h) for vertical
uniform float u_radius;

out vec4 fragColor;

void main() {
    vec4 sum = vec4(0.0);
    // 9-tap Gaussian: weights for sigma ~2.0
    float weights[5] = float[](0.227027, 0.194946, 0.121621, 0.054054, 0.016216);

    sum += texture(u_texture, v_uv) * weights[0];

    for (int i = 1; i < 5; i++) {
        vec2 offset = u_direction * float(i) * u_radius;
        sum += texture(u_texture, v_uv + offset) * weights[i];
        sum += texture(u_texture, v_uv - offset) * weights[i];
    }

    fragColor = sum;
}
`;

export const COMPOSITE_FRAG = /* glsl */ `#version 300 es
precision highp float;

in vec2 v_uv;

uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_bloomStrength;

out vec4 fragColor;

void main() {
    vec4 scene = texture(u_scene, v_uv);
    vec4 bloom = texture(u_bloom, v_uv);

    // Additive bloom
    fragColor = scene + bloom * u_bloomStrength;
}
`;
