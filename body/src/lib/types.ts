/**
 * Shared TypeScript types for the Hephia frontend.
 * These mirror the soul server's API response shapes and the Rust IPC event payloads.
 */

// ---------------------------------------------------------------------------
// Backend state types (from soul server /ws payload)
// ---------------------------------------------------------------------------

export interface MoodState {
  name: string;
  valence: number;  // -1 to 1
  arousal: number;  // -1 to 1
}

export interface NeedState {
  satisfaction: number;  // 0 to 1
}

export interface NeedsMap {
  hunger: NeedState;
  thirst: NeedState;
  boredom: NeedState;
  loneliness: NeedState;
  stamina: NeedState;
}

export interface BehaviorState {
  name: string | null;
  active: boolean;
}

export interface EmotionVector {
  name: string;
  intensity: number;
  valence: number;
  arousal: number;
  type?: 'overall' | 'individual';
}

export interface SystemContext {
  mood: MoodState;
  needs: NeedsMap;
  behavior: BehaviorState;
  emotional_state: EmotionVector[];
}

export interface SoulStatePayload {
  event_type: 'TUI_INITIAL_STATE' | 'TUI_REFRESH_DATA';
  payload: {
    system_context: SystemContext;
    recent_messages: ChatMessage[];
    cognitive_summary: string;
    current_model_name: string;
  };
  timestamp: string;
}

// ---------------------------------------------------------------------------
// Chat types
// ---------------------------------------------------------------------------

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  metadata?: Record<string, unknown>;
}

export interface ChatResponse {
  role: 'assistant';
  content: string;
  was_task: boolean;
  memories_used: string[];
}

// ---------------------------------------------------------------------------
// Action types
// ---------------------------------------------------------------------------

export interface ActionStatus {
  last_execution: number;
  total_executions: number;
  successful_executions: number;
  failed_executions: number;
  on_cooldown: boolean;
  remaining_cooldown: number;
}

export interface ActionInfo {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  status: ActionStatus;
}

export interface ActionResponse {
  success: boolean;
  message: string;
  state_changes: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Memory types
// ---------------------------------------------------------------------------

export interface MemoryResponse {
  id: string;
  content: string;
  timestamp: number;
  strength: number;
  relevance: number;
}

// ---------------------------------------------------------------------------
// Notes types
// ---------------------------------------------------------------------------

export interface NoteResponse {
  id: string;
  content: string;
  created_at: string;
  updated_at: string;
  context: string | null;
  sticky: boolean;
  tags: string[];
}

// ---------------------------------------------------------------------------
// Worker types
// ---------------------------------------------------------------------------

export interface WorkerTaskStatus {
  task_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result: string | null;
  error: string | null;
  created_at: number;
  completed_at: number | null;
}

// ---------------------------------------------------------------------------
// Settings types
// ---------------------------------------------------------------------------

export interface PetSettings {
  pet_name: string;
  pet_model: string;
  worker_model: string;
  personality_prompt: string;
  memory_significance_threshold: number;
  max_conversation_turns: number;
}

export interface SettingsResponse {
  settings: PetSettings;
  available_models: string[];
  version: string;
}

// ---------------------------------------------------------------------------
// IPC event types
// ---------------------------------------------------------------------------

export interface ConnectionPayload {
  initial: boolean;
}

export interface ErrorPayload {
  message: string;
  fatal: boolean;
}

export interface ThoughtBubblePayload {
  source: 'introspection' | 'worker' | 'cognitive';
  content: string;
  action?: string;
}

// ---------------------------------------------------------------------------
// Environment check (wizard)
// ---------------------------------------------------------------------------

export interface EnvironmentStatus {
  python_found: boolean;
  python_version: string | null;
  uv_found: boolean;
  uv_version: string | null;
  venv_exists: boolean;
  project_dir: string | null;
  deps_installed: boolean;
}
