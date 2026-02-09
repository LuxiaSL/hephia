/**
 * Tauri IPC event listener setup.
 *
 * Subscribes to soul:state, soul:chat, soul:connected, soul:disconnected,
 * soul:error, and soul:thought events from the Rust side.
 */

import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import type {
  SoulStatePayload,
  ChatResponse,
  ConnectionPayload,
  ErrorPayload,
  ThoughtBubblePayload,
} from '$lib/types';

export interface IPCCallbacks {
  onState?: (payload: SoulStatePayload) => void;
  onChat?: (payload: ChatResponse) => void;
  onConnected?: (payload: ConnectionPayload) => void;
  onDisconnected?: (payload: { reason: string; will_retry: boolean }) => void;
  onError?: (payload: ErrorPayload) => void;
  onThought?: (payload: ThoughtBubblePayload) => void;
  onThoughtDismiss?: () => void;
}

/**
 * Set up all IPC listeners. Returns an unlisten function that tears down all listeners.
 */
export async function setupIPCListeners(callbacks: IPCCallbacks): Promise<UnlistenFn> {
  const unlisteners: UnlistenFn[] = [];

  if (callbacks.onState) {
    const cb = callbacks.onState;
    unlisteners.push(
      await listen<SoulStatePayload>('soul:state', (event) => cb(event.payload)),
    );
  }

  if (callbacks.onChat) {
    const cb = callbacks.onChat;
    unlisteners.push(
      await listen<ChatResponse>('soul:chat', (event) => cb(event.payload)),
    );
  }

  if (callbacks.onConnected) {
    const cb = callbacks.onConnected;
    unlisteners.push(
      await listen<ConnectionPayload>('soul:connected', (event) => cb(event.payload)),
    );
  }

  if (callbacks.onDisconnected) {
    const cb = callbacks.onDisconnected;
    unlisteners.push(
      await listen<{ reason: string; will_retry: boolean }>('soul:disconnected', (event) =>
        cb(event.payload),
      ),
    );
  }

  if (callbacks.onError) {
    const cb = callbacks.onError;
    unlisteners.push(
      await listen<ErrorPayload>('soul:error', (event) => cb(event.payload)),
    );
  }

  if (callbacks.onThought) {
    const cb = callbacks.onThought;
    unlisteners.push(
      await listen<ThoughtBubblePayload>('soul:thought', (event) => cb(event.payload)),
    );
  }

  if (callbacks.onThoughtDismiss) {
    const cb = callbacks.onThoughtDismiss;
    unlisteners.push(
      await listen<void>('thought:dismiss', () => cb()),
    );
  }

  // Return a function that removes all listeners
  return () => {
    for (const unlisten of unlisteners) {
      unlisten();
    }
  };
}
