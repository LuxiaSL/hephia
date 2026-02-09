//! IPC event types and emission helpers.
//!
//! The Rust side owns the WebSocket connection and broadcasts typed events
//! to all webview windows via Tauri's emit system.

use serde::{Deserialize, Serialize};
use tauri::Emitter;

// ---------------------------------------------------------------------------
// Event payload types
// ---------------------------------------------------------------------------

/// Full state snapshot from the soul server /ws endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoulStatePayload {
    pub event_type: String,
    pub payload: serde_json::Value,
    pub timestamp: String,
}

/// Chat response from the soul server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponsePayload {
    pub role: String,
    pub content: String,
    pub was_task: bool,
    pub memories_used: Vec<String>,
}

/// Connection status event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionPayload {
    pub initial: bool,
}

/// Disconnection event (not shown to user — Rust handles reconnection).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisconnectionPayload {
    pub reason: String,
    pub will_retry: bool,
}

/// Unrecoverable error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPayload {
    pub message: String,
    pub fatal: bool,
}

/// Thought bubble notification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtBubblePayload {
    pub source: String, // "introspection", "worker", "cognitive"
    pub content: String,
    pub action: Option<String>, // optional: "open_worker", "open_chat"
}

// ---------------------------------------------------------------------------
// Emission helpers
// ---------------------------------------------------------------------------

/// Emit a full state update to all windows.
pub fn emit_state(app: &tauri::AppHandle, state: SoulStatePayload) {
    if let Err(e) = app.emit("soul:state", state) {
        log::error!("Failed to emit soul:state: {}", e);
    }
}

/// Emit a chat response to all windows.
pub fn emit_chat(app: &tauri::AppHandle, response: ChatResponsePayload) {
    if let Err(e) = app.emit("soul:chat", response) {
        log::error!("Failed to emit soul:chat: {}", e);
    }
}

/// Emit connection established.
pub fn emit_connected(app: &tauri::AppHandle, initial: bool) {
    if let Err(e) = app.emit("soul:connected", ConnectionPayload { initial }) {
        log::error!("Failed to emit soul:connected: {}", e);
    }
}

/// Emit disconnection (frontend ignores this — it's for internal tracking).
pub fn emit_disconnected(app: &tauri::AppHandle, reason: &str, will_retry: bool) {
    if let Err(e) = app.emit(
        "soul:disconnected",
        DisconnectionPayload {
            reason: reason.to_string(),
            will_retry,
        },
    ) {
        log::error!("Failed to emit soul:disconnected: {}", e);
    }
}

/// Emit an unrecoverable error.
pub fn emit_error(app: &tauri::AppHandle, message: &str, fatal: bool) {
    if let Err(e) = app.emit(
        "soul:error",
        ErrorPayload {
            message: message.to_string(),
            fatal,
        },
    ) {
        log::error!("Failed to emit soul:error: {}", e);
    }
}

/// Emit a thought bubble notification.
pub fn emit_thought_bubble(app: &tauri::AppHandle, source: &str, content: &str, action: Option<&str>) {
    if let Err(e) = app.emit(
        "soul:thought",
        ThoughtBubblePayload {
            source: source.to_string(),
            content: content.to_string(),
            action: action.map(|s| s.to_string()),
        },
    ) {
        log::error!("Failed to emit soul:thought: {}", e);
    }
}
