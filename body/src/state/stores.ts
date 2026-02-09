/**
 * Svelte-compatible reactive stores for shared frontend state.
 *
 * These stores are used across windows (each window gets its own instance,
 * but they all receive the same IPC events from Rust).
 */

import type {
  SystemContext,
  ChatMessage,
  ChatResponse,
  ErrorPayload,
  ThoughtBubblePayload,
} from '$lib/types';

// ---------------------------------------------------------------------------
// Simple store implementation (Svelte 5 compatible, no dependency on svelte/store)
// ---------------------------------------------------------------------------

type Subscriber<T> = (value: T) => void;

export interface Store<T> {
  subscribe(fn: Subscriber<T>): () => void;
  get(): T;
  set(value: T): void;
  update(fn: (current: T) => T): void;
}

export function createStore<T>(initial: T): Store<T> {
  let value = initial;
  const subscribers = new Set<Subscriber<T>>();

  return {
    subscribe(fn: Subscriber<T>) {
      subscribers.add(fn);
      fn(value);
      return () => subscribers.delete(fn);
    },
    get() {
      return value;
    },
    set(newValue: T) {
      value = newValue;
      for (const fn of subscribers) fn(value);
    },
    update(fn: (current: T) => T) {
      value = fn(value);
      for (const fn of subscribers) fn(value);
    },
  };
}

// ---------------------------------------------------------------------------
// Application stores
// ---------------------------------------------------------------------------

/** Whether the soul server connection is active. */
export const connected = createStore<boolean>(false);

/** Latest system context from the soul server. */
export const systemContext = createStore<SystemContext | null>(null);

/** Chat message history (local to this session). */
export const chatMessages = createStore<ChatMessage[]>([]);

/** Pending thought bubble notifications. */
export const thoughtBubbles = createStore<ThoughtBubblePayload[]>([]);

/** Current fatal error, if any. */
export const fatalError = createStore<ErrorPayload | null>(null);

/** Whether passthrough mode is active. */
export const passthroughMode = createStore<boolean>(false);

// ---------------------------------------------------------------------------
// Store actions
// ---------------------------------------------------------------------------

export function addChatMessage(msg: ChatMessage): void {
  chatMessages.update((msgs) => [...msgs, msg]);
}

export function addChatResponse(response: ChatResponse): void {
  addChatMessage({
    role: 'assistant',
    content: response.content,
    metadata: {
      was_task: response.was_task,
      memories_used: response.memories_used,
    },
  });
}

export function addThoughtBubble(thought: ThoughtBubblePayload): void {
  thoughtBubbles.update((bubbles) => {
    const updated = [...bubbles, thought];
    // Cap at 10
    if (updated.length > 10) {
      return updated.slice(updated.length - 10);
    }
    return updated;
  });
}

export function dismissTopThoughtBubble(): void {
  thoughtBubbles.update((bubbles) => bubbles.slice(1));
}

export function clearThoughtBubbles(): void {
  thoughtBubbles.set([]);
}
