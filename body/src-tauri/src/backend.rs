//! Python backend process lifecycle manager.
//!
//! Spawns the soul server via `uv run`, monitors the child process,
//! handles restart on crash, and clean shutdown on quit.

use log::{error, info, warn};
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tauri::Manager;
use tokio::process::{Child, Command};
use tokio::time::sleep;

use crate::ipc;

const BACKEND_PORT: u16 = 5517;
const HEALTH_CHECK_URL: &str = "http://127.0.0.1:5517/state";
const HEALTH_CHECK_INTERVAL: Duration = Duration::from_millis(500);
const HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_RESTART_ATTEMPTS: u32 = 3;
const RESTART_DELAY: Duration = Duration::from_secs(2);

pub struct BackendManager {
    child: Option<Child>,
    project_dir: Option<PathBuf>,
    restart_count: u32,
}

impl BackendManager {
    pub fn new() -> Self {
        Self {
            child: None,
            project_dir: None,
            restart_count: 0,
        }
    }

    /// Detect the project directory (where the Python backend lives).
    /// Looks relative to the Tauri app's resource directory, then falls back to CWD parent.
    pub fn detect_project_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
        // In dev, the body/ dir is inside the project root
        // The src-tauri/ is at body/src-tauri/, so project root is ../../
        if let Ok(resource_dir) = app.path().resource_dir() {
            let project_root = resource_dir
                .parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.parent());
            if let Some(root) = project_root {
                if root.join("config.py").exists() {
                    return Ok(root.to_path_buf());
                }
            }
        }

        // Try relative to current working directory
        if let Ok(cwd) = std::env::current_dir() {
            // If we're in body/ or body/src-tauri/
            for ancestor in cwd.ancestors() {
                if ancestor.join("config.py").exists() && ancestor.join("core").exists() {
                    return Ok(ancestor.to_path_buf());
                }
            }
        }

        // Last resort: environment variable
        if let Ok(dir) = std::env::var("HEPHIA_PROJECT_DIR") {
            let path = PathBuf::from(dir);
            if path.join("config.py").exists() {
                return Ok(path);
            }
        }

        Err("Could not detect Hephia project directory. Set HEPHIA_PROJECT_DIR.".to_string())
    }

    /// Start the Python backend process.
    pub async fn start(&mut self, app: &tauri::AppHandle) -> Result<(), String> {
        if self.child.is_some() {
            return Err("Backend is already running".to_string());
        }

        let project_dir = Self::detect_project_dir(app)?;
        info!("Starting backend in: {}", project_dir.display());
        self.project_dir = Some(project_dir.clone());

        self.spawn_process(&project_dir).await?;
        self.wait_for_health().await?;
        self.restart_count = 0;

        info!("Backend is healthy on port {}", BACKEND_PORT);
        Ok(())
    }

    /// Spawn the actual Python process.
    async fn spawn_process(&mut self, project_dir: &PathBuf) -> Result<(), String> {
        // Try `uv run` first, fall back to direct python
        let (program, args) = if Self::has_uv() {
            (
                "uv".to_string(),
                vec![
                    "run".to_string(),
                    "python".to_string(),
                    "-c".to_string(),
                    "import uvicorn; from core.server import HephiaServer; import asyncio; server = asyncio.run(HephiaServer.create()); server.run()".to_string(),
                ],
            )
        } else if project_dir.join(".venv/bin/python").exists() {
            (
                project_dir
                    .join(".venv/bin/python")
                    .to_string_lossy()
                    .to_string(),
                vec![
                    "-c".to_string(),
                    "import uvicorn; from core.server import HephiaServer; import asyncio; server = asyncio.run(HephiaServer.create()); server.run()".to_string(),
                ],
            )
        } else {
            return Err("Neither `uv` nor .venv/bin/python found. Run the setup wizard.".to_string());
        };

        let child = Command::new(&program)
            .args(&args)
            .current_dir(project_dir)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("Failed to spawn backend process: {}", e))?;

        info!("Spawned backend process (pid: {:?})", child.id());
        self.child = Some(child);
        Ok(())
    }

    /// Poll GET /state until the backend responds.
    async fn wait_for_health(&self) -> Result<(), String> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .map_err(|e| format!("HTTP client error: {}", e))?;

        let deadline = tokio::time::Instant::now() + HEALTH_CHECK_TIMEOUT;

        while tokio::time::Instant::now() < deadline {
            match client.get(HEALTH_CHECK_URL).send().await {
                Ok(resp) if resp.status().is_success() => return Ok(()),
                _ => sleep(HEALTH_CHECK_INTERVAL).await,
            }
        }

        Err(format!(
            "Backend did not become healthy within {}s",
            HEALTH_CHECK_TIMEOUT.as_secs()
        ))
    }

    /// Stop the backend process gracefully.
    pub async fn stop(&mut self) -> Result<(), String> {
        if let Some(mut child) = self.child.take() {
            info!("Stopping backend process...");

            // Send SIGTERM first
            #[cfg(unix)]
            {
                use nix::sys::signal::{kill, Signal};
                use nix::unistd::Pid;
                if let Some(pid) = child.id() {
                    let _ = kill(Pid::from_raw(pid as i32), Signal::SIGTERM);
                }
            }

            // Wait up to 5 seconds for graceful shutdown
            match tokio::time::timeout(Duration::from_secs(5), child.wait()).await {
                Ok(Ok(status)) => {
                    info!("Backend exited with: {}", status);
                }
                _ => {
                    warn!("Backend didn't stop gracefully, killing...");
                    let _ = child.kill().await;
                }
            }
        }
        Ok(())
    }

    /// Attempt to restart the backend after a crash.
    pub async fn try_restart(&mut self, app: &tauri::AppHandle) -> Result<(), String> {
        self.restart_count += 1;

        if self.restart_count > MAX_RESTART_ATTEMPTS {
            let msg = format!(
                "Backend crashed {} times and won't restart",
                MAX_RESTART_ATTEMPTS
            );
            error!("{}", msg);
            ipc::emit_error(app, &msg, true);
            return Err(msg);
        }

        warn!(
            "Restarting backend (attempt {}/{})",
            self.restart_count, MAX_RESTART_ATTEMPTS
        );
        sleep(RESTART_DELAY).await;

        // Clean up old process
        self.child = None;

        if let Some(project_dir) = self.project_dir.clone() {
            self.spawn_process(&project_dir).await?;
            self.wait_for_health().await?;
            info!("Backend restarted successfully");
            Ok(())
        } else {
            Err("No project directory stored for restart".to_string())
        }
    }

    /// Check if the backend process is still alive.
    pub fn is_running(&mut self) -> bool {
        if let Some(child) = &mut self.child {
            match child.try_wait() {
                Ok(Some(_status)) => {
                    // Process has exited
                    false
                }
                Ok(None) => true, // Still running
                Err(_) => false,
            }
        } else {
            false
        }
    }

    /// Check if `uv` is available on PATH.
    fn has_uv() -> bool {
        std::process::Command::new("uv")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
}

/// Environment detection result for the wizard.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EnvironmentStatus {
    pub python_found: bool,
    pub python_version: Option<String>,
    pub uv_found: bool,
    pub uv_version: Option<String>,
    pub venv_exists: bool,
    pub project_dir: Option<String>,
    pub deps_installed: bool,
}

/// Check the system environment for the wizard.
pub async fn check_environment(app: &tauri::AppHandle) -> EnvironmentStatus {
    let project_dir = BackendManager::detect_project_dir(app).ok();

    let (python_found, python_version) = check_python(&project_dir);
    let (uv_found, uv_version) = check_uv();
    let venv_exists = project_dir
        .as_ref()
        .map(|d| d.join(".venv").exists())
        .unwrap_or(false);
    let deps_installed = venv_exists
        && project_dir
            .as_ref()
            .map(|d| d.join(".venv/lib").exists())
            .unwrap_or(false);

    EnvironmentStatus {
        python_found,
        python_version,
        uv_found,
        uv_version,
        venv_exists,
        project_dir: project_dir.map(|p| p.to_string_lossy().to_string()),
        deps_installed,
    }
}

fn check_python(project_dir: &Option<PathBuf>) -> (bool, Option<String>) {
    // Check .venv first
    if let Some(dir) = project_dir {
        let venv_python = dir.join(".venv/bin/python");
        if venv_python.exists() {
            if let Ok(output) = std::process::Command::new(&venv_python)
                .arg("--version")
                .output()
            {
                let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
                return (true, Some(version));
            }
        }
    }

    // Fall back to system python
    if let Ok(output) = std::process::Command::new("python3").arg("--version").output() {
        let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
        return (true, Some(version));
    }

    (false, None)
}

fn check_uv() -> (bool, Option<String>) {
    if let Ok(output) = std::process::Command::new("uv").arg("--version").output() {
        let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
        (true, Some(version))
    } else {
        (false, None)
    }
}
