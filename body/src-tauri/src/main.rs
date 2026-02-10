// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;
mod commands;
mod ipc;
mod tray;
mod websocket;
mod windows;

use log::info;
use std::sync::Arc;
use tauri::Manager;
use tokio::sync::Mutex;

/// Shared application state accessible from commands and event handlers.
pub struct AppState {
    pub backend: Arc<Mutex<backend::BackendManager>>,
    pub ws_client: Arc<Mutex<Option<websocket::SoulConnection>>>,
    pub passthrough: Arc<Mutex<bool>>,
}

fn main() {
    env_logger::init();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .manage(AppState {
            backend: Arc::new(Mutex::new(backend::BackendManager::new())),
            ws_client: Arc::new(Mutex::new(None)),
            passthrough: Arc::new(Mutex::new(false)),
        })
        .invoke_handler(tauri::generate_handler![
            commands::send_chat_message,
            commands::perform_action,
            commands::get_actions,
            commands::get_state_snapshot,
            commands::get_memories,
            commands::search_memories,
            commands::get_notes,
            commands::create_note,
            commands::update_note,
            commands::delete_note,
            commands::submit_worker_task,
            commands::get_worker_status,
            commands::get_settings,
            commands::update_settings,
            commands::toggle_passthrough,
            commands::open_chat,
            commands::open_dashboard,
            commands::dismiss_thought_bubble,
            commands::write_env_keys,
            commands::start_backend,
            commands::stop_backend,
            commands::check_environment,
            commands::mark_wizard_complete,
            commands::get_cursor_position,
            commands::move_overlay,
            commands::show_context_menu,
        ])
        .setup(|app| {
            info!("Hephia body starting up...");

            // Set up system tray
            tray::setup_tray(app)?;

            // Register global hotkeys
            windows::register_hotkeys(app)?;

            // Register context menu event handler on overlay window
            if let Some(overlay) = app.get_webview_window("overlay") {
                let ctx_handle = app.handle().clone();
                overlay.on_menu_event(move |_win, event| {
                    handle_context_menu_event(&ctx_handle, &event);
                });
            }

            // Check if first run — if wizard not completed, show wizard
            // Otherwise, start backend and show overlay
            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let config_path = windows::get_config_path();
                if windows::is_wizard_completed(&config_path) {
                    // Normal startup: start backend, connect, show overlay
                    if let Err(e) = startup_sequence(&app_handle).await {
                        log::error!("Startup failed: {}", e);
                        ipc::emit_error(&app_handle, &format!("Startup failed: {}", e), true);
                    }
                } else {
                    // First run: show wizard
                    if let Err(e) = windows::create_wizard_window(&app_handle) {
                        log::error!("Failed to create wizard window: {}", e);
                    }
                }
            });

            Ok(())
        })
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                let label = window.label();
                match label {
                    // Chat and dashboard hide instead of closing (preserve state)
                    "chat" | "dashboard" => {
                        api.prevent_close();
                        let _ = window.hide();
                    }
                    // Wizard closing: trigger normal startup (backend already running)
                    "wizard" => {
                        let handle = window.app_handle().clone();
                        tauri::async_runtime::spawn(async move {
                            if let Err(e) = startup_sequence(&handle).await {
                                log::error!("Post-wizard startup failed: {}", e);
                                ipc::emit_error(
                                    &handle,
                                    &format!("Post-wizard startup failed: {}", e),
                                    true,
                                );
                            }
                        });
                    }
                    _ => {}
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running Hephia");
}

/// Normal startup: launch backend, connect WebSocket, show overlay.
/// Tolerates backend already running (e.g. wizard started it).
async fn startup_sequence(app: &tauri::AppHandle) -> Result<(), String> {
    let state = app.state::<AppState>();

    // Start the Python backend (skip if already running)
    {
        let mut backend = state.backend.lock().await;
        if !backend.is_running() {
            backend
                .start(app)
                .await
                .map_err(|e| format!("Backend start failed: {}", e))?;
        }
    }

    // Connect WebSocket (skip if already connected)
    {
        let ws = state.ws_client.lock().await;
        if ws.is_some() {
            drop(ws);
            if let Some(overlay) = app.get_webview_window("overlay") {
                let _ = overlay.show();
            }
            return Ok(());
        }
    }

    let connection = websocket::SoulConnection::connect(app.clone())
        .await
        .map_err(|e| format!("WebSocket connect failed: {}", e))?;

    {
        let mut ws = state.ws_client.lock().await;
        *ws = Some(connection);
    }

    // Show the overlay window
    if let Some(overlay) = app.get_webview_window("overlay") {
        let _ = overlay.show();
    }

    ipc::emit_connected(app, true);
    info!("Startup complete — Hephia is alive");
    Ok(())
}

/// Handle context menu item selection from the overlay popup menu.
fn handle_context_menu_event(app: &tauri::AppHandle, event: &tauri::menu::MenuEvent) {
    let id = event.id().as_ref();
    match id {
        "ctx_chat" => {
            let _ = windows::show_chat_window(app);
        }
        "ctx_dashboard" => {
            let _ = windows::show_dashboard_window(app, None);
        }
        "ctx_hide" => {
            if let Some(overlay) = app.get_webview_window("overlay") {
                let _ = overlay.hide();
            }
        }
        "ctx_quit" => {
            info!("Quit requested from context menu");
            let h = app.clone();
            tauri::async_runtime::spawn(async move {
                let state = h.state::<AppState>();
                let mut backend = state.backend.lock().await;
                let _ = backend.stop().await;
                drop(backend);
                h.exit(0);
            });
        }
        _ if id.starts_with("ctx_action_") => {
            let action = id.strip_prefix("ctx_action_").unwrap().to_string();
            info!("Action from context menu: {}", action);
            tauri::async_runtime::spawn(async move {
                let body = serde_json::json!({
                    "action": &action,
                    "parameters": {},
                });
                let url = format!("http://127.0.0.1:5517/v1/actions/{}", action);
                match reqwest::Client::new().post(&url).json(&body).send().await {
                    Ok(resp) => {
                        if !resp.status().is_success() {
                            log::warn!("Action {} failed: {}", action, resp.status());
                        }
                    }
                    Err(e) => log::warn!("Action {} request failed: {}", action, e),
                }
            });
        }
        _ => {}
    }
}
