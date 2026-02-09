//! Window management: creation, positioning, hotkey registration.

use log::info;
use std::path::PathBuf;
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder};

/// Get the path to the frontend config file.
pub fn get_config_path() -> PathBuf {
    let config_dir = dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("~/.config"))
        .join("hephia");
    std::fs::create_dir_all(&config_dir).ok();
    config_dir.join("frontend.json")
}

/// Check if the setup wizard has been completed.
pub fn is_wizard_completed(config_path: &PathBuf) -> bool {
    if let Ok(contents) = std::fs::read_to_string(config_path) {
        if let Ok(config) = serde_json::from_str::<serde_json::Value>(&contents) {
            return config
                .get("wizard_completed")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
        }
    }
    false
}

/// Mark the wizard as completed.
pub fn mark_wizard_completed(config_path: &PathBuf) -> Result<(), String> {
    let config = serde_json::json!({
        "wizard_completed": true,
    });
    std::fs::write(config_path, serde_json::to_string_pretty(&config).unwrap())
        .map_err(|e| format!("Failed to write config: {}", e))
}

/// Create the wizard window.
pub fn create_wizard_window(app: &tauri::AppHandle) -> Result<(), String> {
    info!("Creating wizard window");
    WebviewWindowBuilder::new(app, "wizard", WebviewUrl::App("/src/wizard/index.html".into()))
        .title("Hephia Setup")
        .inner_size(600.0, 700.0)
        .resizable(false)
        .center()
        .build()
        .map_err(|e| format!("Failed to create wizard window: {}", e))?;
    Ok(())
}

/// Show (or create) the chat window, positioned near the pet.
pub fn show_chat_window(app: &tauri::AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("chat") {
        // Reposition near current overlay location every time
        let (x, y) = get_chat_position(app);
        let _ = window.set_position(tauri::Position::Logical(tauri::LogicalPosition::new(x, y)));
        let _ = window.show();
        let _ = window.set_focus();
        return Ok(());
    }

    // Create new chat window
    let (x, y) = get_chat_position(app);

    WebviewWindowBuilder::new(app, "chat", WebviewUrl::App("/src/chat/index.html".into()))
        .title("Hephia Chat")
        .inner_size(400.0, 600.0)
        .position(x, y)
        .decorations(false)
        .always_on_top(true)
        .build()
        .map_err(|e| format!("Failed to create chat window: {}", e))?;

    Ok(())
}

/// Show (or create) the dashboard window.
pub fn show_dashboard_window(app: &tauri::AppHandle, _tab: Option<&str>) -> Result<(), String> {
    if let Some(window) = app.get_webview_window("dashboard") {
        let _ = window.show();
        let _ = window.set_focus();
        // TODO: emit event to switch to requested tab
        return Ok(());
    }

    let (x, y) = get_dashboard_position(app);

    WebviewWindowBuilder::new(
        app,
        "dashboard",
        WebviewUrl::App("/src/dashboard/index.html".into()),
    )
    .title("Hephia Dashboard")
    .inner_size(500.0, 700.0)
    .position(x, y)
    .decorations(false)
    .build()
    .map_err(|e| format!("Failed to create dashboard window: {}", e))?;

    Ok(())
}

/// Calculate chat window position near the pet overlay, edge-aware.
fn get_chat_position(app: &tauri::AppHandle) -> (f64, f64) {
    let overlay = app.get_webview_window("overlay");
    let margin = 20.0;
    let chat_width = 400.0;
    let chat_height = 600.0;

    if let Some(overlay) = overlay {
        if let (Ok(pos), Ok(monitor)) = (overlay.outer_position(), overlay.current_monitor()) {
            if let Some(monitor) = monitor {
                let screen_w = monitor.size().width as f64;
                let screen_h = monitor.size().height as f64;
                let pet_x = pos.x as f64;
                let pet_y = pos.y as f64;

                // Prefer right of pet, flip to left if near right edge
                let x = if pet_x + margin + chat_width < screen_w {
                    pet_x + margin + 192.0 // pet overlay window width
                } else {
                    pet_x - margin - chat_width
                };

                // Center vertically on pet, clamp to screen
                let y = (pet_y - chat_height / 2.0).max(0.0).min(screen_h - chat_height);

                return (x, y);
            }
        }
    }

    // Fallback: center of screen
    (200.0, 100.0)
}

/// Calculate dashboard window position.
fn get_dashboard_position(app: &tauri::AppHandle) -> (f64, f64) {
    // Open near center of screen
    if let Some(overlay) = app.get_webview_window("overlay") {
        if let Ok(Some(monitor)) = overlay.current_monitor() {
            let screen_w = monitor.size().width as f64;
            let screen_h = monitor.size().height as f64;
            return ((screen_w - 500.0) / 2.0, (screen_h - 700.0) / 2.0);
        }
    }
    (200.0, 100.0)
}

/// Register global hotkeys.
pub fn register_hotkeys(app: &tauri::App) -> Result<(), String> {
    use tauri_plugin_global_shortcut::GlobalShortcutExt;

    let app_handle = app.handle().clone();

    // Ctrl+Shift+H: toggle pet visibility
    let handle1 = app_handle.clone();
    app.global_shortcut().on_shortcut("CmdOrCtrl+Shift+H", move |_app, _shortcut, _event| {
        if let Some(overlay) = handle1.get_webview_window("overlay") {
            if overlay.is_visible().unwrap_or(false) {
                let _ = overlay.hide();
            } else {
                let _ = overlay.show();
            }
        }
    }).map_err(|e| format!("Failed to register show/hide hotkey: {}", e))?;

    // Ctrl+Shift+C: open chat
    let handle2 = app_handle.clone();
    app.global_shortcut().on_shortcut("CmdOrCtrl+Shift+C", move |_app, _shortcut, _event| {
        let _ = show_chat_window(&handle2);
    }).map_err(|e| format!("Failed to register chat hotkey: {}", e))?;

    // Ctrl+Shift+P: toggle passthrough
    let handle3 = app_handle.clone();
    app.global_shortcut().on_shortcut("CmdOrCtrl+Shift+P", move |_app, _shortcut, _event| {
        let h = handle3.clone();
        tauri::async_runtime::spawn(async move {
            let state = h.state::<crate::AppState>();
            let mut passthrough = state.passthrough.lock().await;
            *passthrough = !*passthrough;
            if let Some(overlay) = h.get_webview_window("overlay") {
                let _ = overlay.set_ignore_cursor_events(*passthrough);
            }
        });
    }).map_err(|e| format!("Failed to register passthrough hotkey: {}", e))?;

    info!("Registered global hotkeys: Ctrl+Shift+H (show/hide), Ctrl+Shift+C (chat), Ctrl+Shift+P (passthrough)");
    Ok(())
}
