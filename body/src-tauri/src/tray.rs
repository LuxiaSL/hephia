//! System tray setup and event handling.

use log::info;
use tauri::{
    menu::{MenuBuilder, MenuItemBuilder, PredefinedMenuItem},
    tray::TrayIconEvent,
    Manager,
};

pub fn setup_tray(app: &tauri::App) -> Result<(), String> {
    let show_hide = MenuItemBuilder::with_id("show_hide", "Show/Hide")
        .build(app)
        .map_err(|e| format!("Menu item error: {}", e))?;

    let open_chat = MenuItemBuilder::with_id("open_chat", "Open Chat")
        .build(app)
        .map_err(|e| format!("Menu item error: {}", e))?;

    let passthrough = MenuItemBuilder::with_id("passthrough", "Passthrough Mode")
        .build(app)
        .map_err(|e| format!("Menu item error: {}", e))?;

    let separator = PredefinedMenuItem::separator(app)
        .map_err(|e| format!("Separator error: {}", e))?;

    let quit = MenuItemBuilder::with_id("quit", "Quit")
        .build(app)
        .map_err(|e| format!("Menu item error: {}", e))?;

    let menu = MenuBuilder::new(app)
        .items(&[&show_hide, &open_chat, &passthrough, &separator, &quit])
        .build()
        .map_err(|e| format!("Menu build error: {}", e))?;

    let tray = app
        .tray_by_id("main-tray")
        .ok_or("Tray icon not found")?;

    tray.set_menu(Some(menu))
        .map_err(|e| format!("Set menu error: {}", e))?;

    // Handle tray left-click: toggle visibility
    let app_handle = app.handle().clone();
    tray.on_tray_icon_event(move |_tray, event| {
        if let TrayIconEvent::Click { button, .. } = event {
            if button == tauri::tray::MouseButton::Left {
                if let Some(overlay) = app_handle.get_webview_window("overlay") {
                    if overlay.is_visible().unwrap_or(false) {
                        let _ = overlay.hide();
                    } else {
                        let _ = overlay.show();
                    }
                }
            }
        }
    });

    // Handle menu item clicks
    let app_handle2 = app.handle().clone();
    tray.on_menu_event(move |_tray, event| {
        match event.id().as_ref() {
            "show_hide" => {
                if let Some(overlay) = app_handle2.get_webview_window("overlay") {
                    if overlay.is_visible().unwrap_or(false) {
                        let _ = overlay.hide();
                    } else {
                        let _ = overlay.show();
                    }
                }
            }
            "open_chat" => {
                let _ = crate::windows::show_chat_window(&app_handle2);
            }
            "passthrough" => {
                let h = app_handle2.clone();
                tauri::async_runtime::spawn(async move {
                    let state = h.state::<crate::AppState>();
                    let mut passthrough = state.passthrough.lock().await;
                    *passthrough = !*passthrough;
                    if let Some(overlay) = h.get_webview_window("overlay") {
                        let _ = overlay.set_ignore_cursor_events(*passthrough);
                    }
                });
            }
            "quit" => {
                info!("Quit requested from tray");
                let h = app_handle2.clone();
                tauri::async_runtime::spawn(async move {
                    // Stop backend before exiting
                    let state = h.state::<crate::AppState>();
                    let mut backend = state.backend.lock().await;
                    let _ = backend.stop().await;
                    drop(backend);
                    h.exit(0);
                });
            }
            _ => {}
        }
    });

    info!("System tray configured");
    Ok(())
}
