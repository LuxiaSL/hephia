//! WebSocket client connecting to the soul server.
//!
//! Maintains connections to /ws (state stream) and /ws/chat (chat stream).
//! Handles reconnection with exponential backoff.
//! Emits IPC events to all webview windows.
//!
//! Uses raw TCP (no TLS) since we connect to localhost:5517.

use futures_util::{SinkExt, StreamExt};
use log::{error, info, warn};
use std::sync::Arc;
use std::time::Duration;
use tauri::Manager;
use tokio::net::TcpStream;
use tokio::sync::Mutex;
use tokio_tungstenite::{client_async, tungstenite::Message, WebSocketStream};

use crate::ipc;

const WS_STATE_URL: &str = "ws://127.0.0.1:5517/ws";
const WS_CHAT_URL: &str = "ws://127.0.0.1:5517/ws/chat";
const BACKEND_HOST: &str = "127.0.0.1:5517";

const MAX_RECONNECT_ATTEMPTS: u32 = 5;
const INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const MAX_BACKOFF: Duration = Duration::from_secs(30);

type WsSink = futures_util::stream::SplitSink<WebSocketStream<TcpStream>, Message>;
type WsStream = futures_util::stream::SplitStream<WebSocketStream<TcpStream>>;

/// Connect to a WebSocket endpoint over plain TCP.
async fn ws_connect(url: &str) -> Result<(WsSink, WsStream), String> {
    let tcp = TcpStream::connect(BACKEND_HOST)
        .await
        .map_err(|e| format!("TCP connect failed: {}", e))?;

    let (ws, _) = client_async(url, tcp)
        .await
        .map_err(|e| format!("WebSocket handshake failed: {}", e))?;

    Ok(ws.split())
}

/// Holds the active WebSocket connections and manages their lifecycle.
pub struct SoulConnection {
    /// Sink for sending chat messages via /ws/chat
    chat_sink: Arc<Mutex<Option<WsSink>>>,
}

impl SoulConnection {
    /// Connect to both WebSocket endpoints and start listening.
    /// Spawns a background task that manages the full connection lifecycle
    /// including reconnection.
    pub async fn connect(app: tauri::AppHandle) -> Result<Self, String> {
        let (_, state_stream) = ws_connect(WS_STATE_URL).await?;
        info!("Connected to soul server state stream");

        let (chat_sink, chat_stream) = ws_connect(WS_CHAT_URL).await?;
        info!("Connected to soul server chat stream");

        let chat_sink_shared = Arc::new(Mutex::new(Some(chat_sink)));

        // Spawn chat listener
        tokio::spawn(chat_listen_loop(app.clone(), chat_stream));

        // Spawn the connection lifecycle manager — handles state listening
        // and reconnection in a single non-recursive loop
        let sink_ref = chat_sink_shared.clone();
        tokio::spawn(connection_lifecycle(app, sink_ref, state_stream));

        Ok(Self {
            chat_sink: chat_sink_shared,
        })
    }

    /// Send a chat message through the /ws/chat connection.
    pub async fn send_chat_message(&self, message: &str) -> Result<(), String> {
        let payload = serde_json::json!({ "message": message });
        let msg = Message::Text(payload.to_string().into());

        let mut sink_guard = self.chat_sink.lock().await;
        if let Some(sink) = sink_guard.as_mut() {
            sink.send(msg)
                .await
                .map_err(|e| format!("Failed to send chat message: {}", e))?;
            Ok(())
        } else {
            Err("Chat WebSocket not connected".to_string())
        }
    }
}

/// Manages the full connection lifecycle: listen → disconnect → reconnect → listen.
///
/// This is a single non-recursive loop — no mutual recursion between functions,
/// so the future is guaranteed `Send`.
async fn connection_lifecycle(
    app: tauri::AppHandle,
    chat_sink: Arc<Mutex<Option<WsSink>>>,
    initial_state_stream: WsStream,
) {
    // Listen on the initial connection
    state_listen(initial_state_stream, &app).await;

    // Initial connection dropped — clear chat sink and enter reconnection
    *chat_sink.lock().await = None;
    ipc::emit_disconnected(&app, "Connection lost", true);

    let mut attempts = 0u32;
    let mut backoff = INITIAL_BACKOFF;

    loop {
        attempts += 1;
        if attempts > MAX_RECONNECT_ATTEMPTS {
            warn!("Max reconnect attempts reached, trying backend restart...");
            let state = app.state::<crate::AppState>();
            let mut backend = state.backend.lock().await;
            if let Err(e) = backend.try_restart(&app).await {
                error!("Backend restart failed: {}", e);
                return;
            }
            drop(backend);
            attempts = 0;
            backoff = INITIAL_BACKOFF;
        }

        info!(
            "Reconnection attempt {} (backoff: {:?})",
            attempts, backoff
        );
        tokio::time::sleep(backoff).await;

        let state_result = ws_connect(WS_STATE_URL).await;
        let chat_result = ws_connect(WS_CHAT_URL).await;

        match (state_result, chat_result) {
            (Ok((_, state_stream)), Ok((new_chat_sink, chat_stream))) => {
                info!("Reconnected successfully");

                // Update the chat sink
                *chat_sink.lock().await = Some(new_chat_sink);

                ipc::emit_connected(&app, false);

                // Spawn new chat listener
                tokio::spawn(chat_listen_loop(app.clone(), chat_stream));

                // Reset counters
                attempts = 0;
                backoff = INITIAL_BACKOFF;

                // Listen until disconnect — blocks here
                state_listen(state_stream, &app).await;

                // State connection lost again
                *chat_sink.lock().await = None;
                ipc::emit_disconnected(&app, "Connection lost", true);
            }
            _ => {
                warn!("Reconnection failed on attempt {}", attempts);
                backoff = std::cmp::min(backoff * 2, MAX_BACKOFF);
            }
        }
    }
}

/// Listen for state updates from /ws. Returns when disconnected.
async fn state_listen(mut stream: WsStream, app: &tauri::AppHandle) {
    while let Some(msg) = stream.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                match serde_json::from_str::<ipc::SoulStatePayload>(&text) {
                    Ok(state) => ipc::emit_state(app, state),
                    Err(e) => {
                        warn!("Failed to parse state message: {}", e);
                    }
                }
            }
            Ok(Message::Close(_)) => {
                warn!("State WebSocket closed by server");
                break;
            }
            Err(e) => {
                error!("State WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }
}

/// Listen for chat responses from /ws/chat and emit IPC events.
async fn chat_listen_loop(app: tauri::AppHandle, mut stream: WsStream) {
    while let Some(msg) = stream.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                match serde_json::from_str::<ipc::ChatResponsePayload>(&text) {
                    Ok(response) => ipc::emit_chat(&app, response),
                    Err(e) => {
                        warn!("Non-chat message from /ws/chat: {} ({})", text, e);
                    }
                }
            }
            Ok(Message::Close(_)) => {
                warn!("Chat WebSocket closed by server");
                break;
            }
            Err(e) => {
                error!("Chat WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }
}
