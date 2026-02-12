//! Tauri command handlers.
//!
//! Each command is invoked from the frontend via `invoke()`.
//! Most proxy to the soul server REST API; some manage local state.

use log::info;
use serde_json::Value;
use tauri::{Emitter, Manager, State};

use crate::AppState;

const BACKEND_URL: &str = "http://127.0.0.1:5517";

/// Helper: parse a response, checking the HTTP status code.
/// Returns `Err` for 4xx/5xx responses so the JS side gets an exception.
async fn parse_response(resp: reqwest::Response) -> Result<Value, String> {
    let status = resp.status();
    let body = resp
        .json::<Value>()
        .await
        .map_err(|e| format!("Parse failed: {}", e))?;

    if status.is_client_error() || status.is_server_error() {
        // Extract FastAPI's "detail" field — may be a string or array
        let detail = match body.get("detail") {
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => "Unknown error".to_string(),
        };
        return Err(format!("Backend error {}: {}", status.as_u16(), detail));
    }

    Ok(body)
}

/// Helper: make a GET request to the backend.
async fn backend_get(path: &str) -> Result<Value, String> {
    let url = format!("{}{}", BACKEND_URL, path);
    let resp = reqwest::get(&url)
        .await
        .map_err(|e| format!("Request failed: {}", e))?;
    parse_response(resp).await
}

/// Helper: make a POST request to the backend.
async fn backend_post(path: &str, body: &Value) -> Result<Value, String> {
    let url = format!("{}{}", BACKEND_URL, path);
    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(body)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;
    parse_response(resp).await
}

/// Helper: make a PUT request to the backend.
async fn backend_put(path: &str, body: &Value) -> Result<Value, String> {
    let url = format!("{}{}", BACKEND_URL, path);
    let client = reqwest::Client::new();
    let resp = client
        .put(&url)
        .json(body)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;
    parse_response(resp).await
}

/// Helper: make a DELETE request to the backend.
async fn backend_delete(path: &str) -> Result<Value, String> {
    let url = format!("{}{}", BACKEND_URL, path);
    let client = reqwest::Client::new();
    let resp = client
        .delete(&url)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;
    parse_response(resp).await
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn send_chat_message(
    state: State<'_, AppState>,
    message: String,
) -> Result<(), String> {
    let ws = state.ws_client.lock().await;
    if let Some(conn) = ws.as_ref() {
        conn.send_chat_message(&message).await
    } else {
        Err("Not connected to soul server".to_string())
    }
}

// ---------------------------------------------------------------------------
// State & Actions
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn get_state_snapshot() -> Result<Value, String> {
    backend_get("/state").await
}

#[tauri::command]
pub async fn perform_action(action: String, params: Option<Value>) -> Result<Value, String> {
    let body = serde_json::json!({
        "action": action,
        "parameters": params.unwrap_or(serde_json::json!({})),
    });
    backend_post(&format!("/v1/actions/{}", action), &body).await
}

#[tauri::command]
pub async fn get_actions() -> Result<Value, String> {
    backend_get("/v1/actions").await
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn get_memories(limit: Option<u32>) -> Result<Value, String> {
    let limit = limit.unwrap_or(5);
    backend_get(&format!("/memory/recent?limit={}", limit)).await
}

#[tauri::command]
pub async fn search_memories(query: String, limit: Option<u32>) -> Result<Value, String> {
    let limit = limit.unwrap_or(5);
    backend_get(&format!(
        "/memory/search?q={}&limit={}",
        urlencoding::encode(&query),
        limit
    ))
    .await
}

// ---------------------------------------------------------------------------
// Notes
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn get_notes(
    tag: Option<String>,
    sticky_only: Option<bool>,
    limit: Option<u32>,
    offset: Option<u32>,
) -> Result<Value, String> {
    let mut params = vec![];
    if let Some(t) = tag {
        params.push(format!("tag={}", urlencoding::encode(&t)));
    }
    if let Some(true) = sticky_only {
        params.push("sticky_only=true".to_string());
    }
    if let Some(l) = limit {
        params.push(format!("limit={}", l));
    }
    if let Some(o) = offset {
        params.push(format!("offset={}", o));
    }
    let qs = if params.is_empty() {
        String::new()
    } else {
        format!("?{}", params.join("&"))
    };
    backend_get(&format!("/notes{}", qs)).await
}

#[tauri::command]
pub async fn create_note(
    content: String,
    tags: Option<Vec<String>>,
    sticky: Option<bool>,
) -> Result<Value, String> {
    let body = serde_json::json!({
        "content": content,
        "tags": tags.unwrap_or_default(),
        "sticky": sticky.unwrap_or(false),
    });
    backend_post("/notes", &body).await
}

#[tauri::command]
pub async fn update_note(
    id: String,
    content: Option<String>,
    tags: Option<Vec<String>>,
    sticky: Option<bool>,
) -> Result<Value, String> {
    let mut body = serde_json::Map::new();
    if let Some(c) = content {
        body.insert("content".to_string(), serde_json::json!(c));
    }
    if let Some(t) = tags {
        body.insert("tags".to_string(), serde_json::json!(t));
    }
    if let Some(s) = sticky {
        body.insert("sticky".to_string(), serde_json::json!(s));
    }
    backend_put(&format!("/notes/{}", id), &Value::Object(body)).await
}

#[tauri::command]
pub async fn delete_note(id: String) -> Result<Value, String> {
    backend_delete(&format!("/notes/{}", id)).await
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn submit_worker_task(task: String, context: Option<String>) -> Result<Value, String> {
    let body = serde_json::json!({
        "task": task,
        "context": context.unwrap_or_default(),
    });
    backend_post("/worker/task", &body).await
}

#[tauri::command]
pub async fn get_worker_status(task_id: String) -> Result<Value, String> {
    backend_get(&format!("/worker/{}", task_id)).await
}

#[tauri::command]
pub async fn reply_to_worker_task(task_id: String, message: String) -> Result<Value, String> {
    let body = serde_json::json!({ "message": message });
    backend_post(&format!("/worker/{}/reply", task_id), &body).await
}

#[tauri::command]
pub async fn stop_worker_task(task_id: String) -> Result<Value, String> {
    backend_post(&format!("/worker/{}/stop", task_id), &serde_json::json!({})).await
}

#[tauri::command]
pub async fn get_chat_history(limit: Option<u32>) -> Result<Value, String> {
    let mut path = "/chat/history".to_string();
    if let Some(l) = limit {
        path = format!("{}?limit={}", path, l);
    }
    backend_get(&path).await
}

#[tauri::command]
pub async fn delete_worker_task(task_id: String) -> Result<Value, String> {
    backend_delete(&format!("/worker/{}", task_id)).await
}

#[tauri::command]
pub async fn clear_worker_tasks(completed_only: bool) -> Result<Value, String> {
    let body = serde_json::json!({ "completed_only": completed_only });
    backend_post("/worker/clear", &body).await
}

#[tauri::command]
pub async fn star_worker_task(task_id: String) -> Result<Value, String> {
    backend_post(&format!("/worker/{}/star", task_id), &serde_json::json!({})).await
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn get_settings() -> Result<Value, String> {
    backend_get("/settings").await
}

#[tauri::command]
pub async fn update_settings(settings: Value) -> Result<Value, String> {
    backend_put("/settings", &settings).await
}

// ---------------------------------------------------------------------------
// Window management
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn toggle_passthrough(
    state: State<'_, AppState>,
    app: tauri::AppHandle,
) -> Result<bool, String> {
    let mut passthrough = state.passthrough.lock().await;
    *passthrough = !*passthrough;
    let new_state = *passthrough;

    // Update overlay window input behavior
    if let Some(overlay) = app.get_webview_window("overlay") {
        let _ = overlay.set_ignore_cursor_events(new_state);
    }

    Ok(new_state)
}

#[tauri::command]
pub async fn open_chat(app: tauri::AppHandle) -> Result<(), String> {
    crate::windows::show_chat_window(&app)
}

#[tauri::command]
pub async fn open_dashboard(app: tauri::AppHandle, tab: Option<String>) -> Result<(), String> {
    crate::windows::show_dashboard_window(&app, tab.as_deref())
}

#[tauri::command]
pub async fn dismiss_thought_bubble(app: tauri::AppHandle) -> Result<(), String> {
    // Emit to overlay to clear the current thought bubble
    let _ = app.emit("thought:dismiss", ());
    Ok(())
}

// ---------------------------------------------------------------------------
// Config file management (for wizard)
// ---------------------------------------------------------------------------

/// Write API keys to the backend .env file.
/// Reads existing .env (preserving comments and other keys), updates/adds the
/// provided keys, and writes back.
#[tauri::command]
pub async fn write_env_keys(
    app: tauri::AppHandle,
    keys: std::collections::HashMap<String, String>,
) -> Result<(), String> {
    use std::collections::HashMap;
    use std::io::Write;

    let project_dir = crate::backend::BackendManager::detect_project_dir(&app)?;
    let env_path = project_dir.join(".env");

    // Read existing .env content (or start fresh)
    let existing = std::fs::read_to_string(&env_path).unwrap_or_default();

    // Parse existing lines, tracking which keys we've seen
    let mut updated_keys: HashMap<String, bool> = HashMap::new();
    let mut lines: Vec<String> = Vec::new();

    for line in existing.lines() {
        let trimmed = line.trim();
        // Check if this line sets one of the keys we want to update
        if !trimmed.is_empty() && !trimmed.starts_with('#') {
            if let Some(eq_pos) = trimmed.find('=') {
                let key = trimmed[..eq_pos].trim();
                if let Some(new_value) = keys.get(key) {
                    // Replace this line with the new value
                    lines.push(format!("{}={}", key, new_value));
                    updated_keys.insert(key.to_string(), true);
                    continue;
                }
            }
        }
        lines.push(line.to_string());
    }

    // Append any keys that weren't already in the file
    let mut appended = false;
    for (key, value) in &keys {
        if value.is_empty() {
            continue; // Skip empty keys
        }
        if !updated_keys.contains_key(key.as_str()) {
            if !appended && !lines.is_empty() {
                lines.push(String::new()); // blank line separator
            }
            lines.push(format!("{}={}", key, value));
            appended = true;
        }
    }

    // Write back
    let content = lines.join("\n");
    let mut file = std::fs::File::create(&env_path)
        .map_err(|e| format!("Failed to write .env: {}", e))?;
    file.write_all(content.as_bytes())
        .map_err(|e| format!("Failed to write .env: {}", e))?;
    // Ensure trailing newline
    file.write_all(b"\n")
        .map_err(|e| format!("Failed to write .env: {}", e))?;

    info!("Wrote {} API key(s) to {}", keys.len(), env_path.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Backend lifecycle (for wizard)
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn start_backend(
    state: State<'_, AppState>,
    app: tauri::AppHandle,
) -> Result<(), String> {
    let mut backend = state.backend.lock().await;
    backend.start(&app).await
}

#[tauri::command]
pub async fn stop_backend(state: State<'_, AppState>) -> Result<(), String> {
    let mut backend = state.backend.lock().await;
    backend.stop().await
}

#[tauri::command]
pub async fn check_environment(
    app: tauri::AppHandle,
) -> Result<crate::backend::EnvironmentStatus, String> {
    Ok(crate::backend::check_environment(&app).await)
}

#[tauri::command]
pub async fn mark_wizard_complete() -> Result<(), String> {
    let config_path = crate::windows::get_config_path();
    crate::windows::mark_wizard_completed(&config_path)
}

// ---------------------------------------------------------------------------
// Context menu
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn show_context_menu(
    app: tauri::AppHandle,
) -> Result<(), String> {
    use tauri::menu::{MenuBuilder, MenuItemBuilder, PredefinedMenuItem, SubmenuBuilder};

    let overlay = app
        .get_webview_window("overlay")
        .ok_or("No overlay window")?;

    let chat = MenuItemBuilder::with_id("ctx_chat", "Chat")
        .build(&app)
        .map_err(|e| e.to_string())?;
    let dashboard = MenuItemBuilder::with_id("ctx_dashboard", "Dashboard")
        .build(&app)
        .map_err(|e| e.to_string())?;

    let sep1 = PredefinedMenuItem::separator(&app).map_err(|e| e.to_string())?;

    let feed = MenuItemBuilder::with_id("ctx_action_feed", "Feed")
        .build(&app)
        .map_err(|e| e.to_string())?;
    let water = MenuItemBuilder::with_id("ctx_action_give_water", "Give Water")
        .build(&app)
        .map_err(|e| e.to_string())?;
    let play = MenuItemBuilder::with_id("ctx_action_play", "Play")
        .build(&app)
        .map_err(|e| e.to_string())?;
    let rest = MenuItemBuilder::with_id("ctx_action_rest", "Rest")
        .build(&app)
        .map_err(|e| e.to_string())?;

    let actions_sub = SubmenuBuilder::with_id(&app, "ctx_actions_sub", "Actions")
        .items(&[&feed, &water, &play, &rest])
        .build()
        .map_err(|e| e.to_string())?;

    let sep2 = PredefinedMenuItem::separator(&app).map_err(|e| e.to_string())?;

    let hide = MenuItemBuilder::with_id("ctx_hide", "Hide")
        .build(&app)
        .map_err(|e| e.to_string())?;
    let quit = MenuItemBuilder::with_id("ctx_quit", "Quit")
        .build(&app)
        .map_err(|e| e.to_string())?;

    let menu = MenuBuilder::new(&app)
        .items(&[
            &chat,
            &dashboard,
            &sep1,
            &actions_sub,
            &sep2,
            &hide,
            &quit,
        ])
        .build()
        .map_err(|e| e.to_string())?;

    overlay
        .popup_menu(&menu)
        .map_err(|e| e.to_string())?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Cursor position (for chase behavior)
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn move_overlay(app: tauri::AppHandle, x: f64, y: f64) -> Result<(), String> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static CALL_COUNT: AtomicU64 = AtomicU64::new(0);

    let n = CALL_COUNT.fetch_add(1, Ordering::Relaxed);
    if n < 5 || n % 300 == 0 {
        info!("move_overlay #{} to ({:.0}, {:.0})", n, x, y);
    }

    if let Some(overlay) = app.get_webview_window("overlay") {
        overlay
            .set_position(tauri::Position::Logical(tauri::LogicalPosition::new(x, y)))
            .map_err(|e| format!("set_position: {}", e))
    } else {
        Err("No overlay window".into())
    }
}

#[tauri::command]
pub async fn get_cursor_position(app: tauri::AppHandle) -> Result<(f64, f64), String> {
    if let Some(overlay) = app.get_webview_window("overlay") {
        let cursor = overlay
            .cursor_position()
            .map_err(|e| format!("cursor_position: {}", e))?;
        let win_pos = overlay
            .outer_position()
            .map_err(|e| format!("outer_position: {}", e))?;
        // cursor_position() is relative to client area; convert to screen coords
        Ok((
            win_pos.x as f64 + cursor.x,
            win_pos.y as f64 + cursor.y,
        ))
    } else {
        Err("Overlay window not found".into())
    }
}
