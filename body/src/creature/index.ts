/**
 * Creature renderer — public API.
 *
 * Used by both the pet overlay and the standalone shader playground.
 * Guarantees visual parity: same simulation, same shaders, same renderer.
 */

export { createCreatureRenderer } from './renderer';
export type { CreatureRenderer } from './renderer';
export { BEHAVIOR_MOTIFS, getMotif, blendMotifs } from './behaviors';
export type { BehaviorMotif } from './behaviors';
