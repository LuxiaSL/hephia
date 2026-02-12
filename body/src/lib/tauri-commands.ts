/**
 * Typed wrappers around Tauri invoke() for all backend commands.
 */

import { invoke } from '@tauri-apps/api/core';
import type {
  ActionInfo,
  ActionResponse,
  ChatHistoryResponse,
  EnvironmentStatus,
  MemoryResponse,
  NoteResponse,
  PetSettings,
  SettingsResponse,
  WorkerStarResponse,
  WorkerTaskStatus,
} from './types';

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

export async function sendChatMessage(message: string): Promise<void> {
  return invoke('send_chat_message', { message });
}

export async function getChatHistory(limit?: number): Promise<ChatHistoryResponse> {
  return invoke('get_chat_history', { limit });
}

// ---------------------------------------------------------------------------
// State & Actions
// ---------------------------------------------------------------------------

export async function getStateSnapshot(): Promise<Record<string, unknown>> {
  return invoke('get_state_snapshot');
}

export async function performAction(
  action: string,
  params?: Record<string, unknown>,
): Promise<ActionResponse> {
  return invoke('perform_action', { action, params });
}

export async function getActions(): Promise<Record<string, ActionInfo>> {
  return invoke('get_actions');
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

export async function getMemories(limit?: number): Promise<MemoryResponse[]> {
  return invoke('get_memories', { limit });
}

export async function searchMemories(
  query: string,
  limit?: number,
): Promise<MemoryResponse[]> {
  return invoke('search_memories', { query, limit });
}

// ---------------------------------------------------------------------------
// Notes
// ---------------------------------------------------------------------------

export async function getNotes(opts?: {
  tag?: string;
  stickyOnly?: boolean;
  limit?: number;
  offset?: number;
}): Promise<NoteResponse[]> {
  return invoke('get_notes', {
    tag: opts?.tag,
    sticky_only: opts?.stickyOnly,
    limit: opts?.limit,
    offset: opts?.offset,
  });
}

export async function createNote(
  content: string,
  tags?: string[],
  sticky?: boolean,
): Promise<NoteResponse> {
  return invoke('create_note', { content, tags, sticky });
}

export async function updateNote(
  id: string,
  updates: { content?: string; tags?: string[]; sticky?: boolean },
): Promise<NoteResponse> {
  return invoke('update_note', { id, ...updates });
}

export async function deleteNote(id: string): Promise<void> {
  return invoke('delete_note', { id });
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

export async function submitWorkerTask(
  task: string,
  context?: string,
): Promise<{ task_id: string }> {
  return invoke('submit_worker_task', { task, context });
}

export async function getWorkerStatus(taskId: string): Promise<WorkerTaskStatus> {
  return invoke('get_worker_status', { taskId });
}

export async function replyToWorkerTask(taskId: string, message: string): Promise<WorkerTaskStatus> {
  return invoke('reply_to_worker_task', { taskId, message });
}

export async function stopWorkerTask(taskId: string): Promise<WorkerTaskStatus> {
  return invoke('stop_worker_task', { taskId });
}

export async function deleteWorkerTask(taskId: string): Promise<void> {
  return invoke('delete_worker_task', { taskId });
}

export async function clearWorkerTasks(completedOnly: boolean): Promise<{ removed: number }> {
  return invoke('clear_worker_tasks', { completedOnly });
}

export async function starWorkerTask(taskId: string): Promise<WorkerStarResponse> {
  return invoke('star_worker_task', { taskId });
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

export async function getSettings(): Promise<SettingsResponse> {
  return invoke('get_settings');
}

export async function updateSettings(settings: PetSettings): Promise<SettingsResponse> {
  return invoke('update_settings', { settings });
}

// ---------------------------------------------------------------------------
// Window management
// ---------------------------------------------------------------------------

export async function togglePassthrough(): Promise<boolean> {
  return invoke('toggle_passthrough');
}

export async function openChat(): Promise<void> {
  return invoke('open_chat');
}

export async function openDashboard(tab?: string): Promise<void> {
  return invoke('open_dashboard', { tab });
}

export async function dismissThoughtBubble(): Promise<void> {
  return invoke('dismiss_thought_bubble');
}

// ---------------------------------------------------------------------------
// Backend lifecycle
// ---------------------------------------------------------------------------

export async function writeEnvKeys(
  keys: Record<string, string>,
): Promise<void> {
  return invoke('write_env_keys', { keys });
}

export async function startBackend(): Promise<void> {
  return invoke('start_backend');
}

export async function stopBackend(): Promise<void> {
  return invoke('stop_backend');
}

export async function checkEnvironment(): Promise<EnvironmentStatus> {
  return invoke('check_environment');
}

export async function markWizardComplete(): Promise<void> {
  return invoke('mark_wizard_complete');
}

// ---------------------------------------------------------------------------
// Context menu
// ---------------------------------------------------------------------------

export async function showContextMenu(): Promise<void> {
  return invoke('show_context_menu');
}

// ---------------------------------------------------------------------------
// Cursor position
// ---------------------------------------------------------------------------

/** Get cursor screen position in physical pixels via Rust. */
export async function getCursorPosition(): Promise<[number, number]> {
  return invoke('get_cursor_position');
}
