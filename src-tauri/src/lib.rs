// Kraken Art — Tauri host. Spawns and supervises the Python sidecar.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use tauri::{Emitter, RunEvent};

const SIDECAR_HOST: &str = "127.0.0.1";
const SIDECAR_PORT: u16 = 7780;

// Watchdog poll interval. 2 s is a good middle: fast enough that a crash is
// noticed before the user re-clicks Generate, slow enough that the wait/kill
// overhead is negligible.
const WATCHDOG_POLL: Duration = Duration::from_secs(2);

// Max time to wait for the sidecar's HTTP server to come up after spawn before
// we declare the spawn "failed" and try again. Cold-start of FastAPI + torch
// import is ~3 s on a fast machine, ~10 s on a slow disk.
const SIDECAR_BOOT_TIMEOUT: Duration = Duration::from_secs(20);

fn project_root() -> PathBuf {
    // src-tauri/ is CWD in dev. In prod, resources sit alongside the binary;
    // we'll refine when bundling lands.
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if cwd.ends_with("src-tauri") {
        cwd.parent().map(PathBuf::from).unwrap_or(cwd)
    } else {
        cwd
    }
}

fn venv_python() -> PathBuf {
    project_root().join("python").join("venv").join("Scripts").join("python.exe")
}

fn sidecar_entry() -> PathBuf {
    project_root().join("python").join("main.py")
}

fn already_listening() -> bool {
    std::net::TcpStream::connect_timeout(
        &format!("{SIDECAR_HOST}:{SIDECAR_PORT}").parse().unwrap(),
        std::time::Duration::from_millis(150),
    )
    .is_ok()
}

fn spawn_sidecar() -> Result<Child, String> {
    let py = venv_python();
    let entry = sidecar_entry();
    if !py.exists() {
        return Err(format!("venv python missing at {}", py.display()));
    }
    if !entry.exists() {
        return Err(format!("sidecar entry missing at {}", entry.display()));
    }
    let mut child = Command::new(&py)
        .arg(&entry)
        .current_dir(project_root().join("python"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn failed: {e}"))?;

    if let Some(out) = child.stdout.take() {
        thread::spawn(move || {
            for line in BufReader::new(out).lines().map_while(Result::ok) {
                eprintln!("[sidecar] {line}");
            }
        });
    }
    if let Some(err) = child.stderr.take() {
        thread::spawn(move || {
            for line in BufReader::new(err).lines().map_while(Result::ok) {
                eprintln!("[sidecar] {line}");
            }
        });
    }
    Ok(child)
}

#[tauri::command]
fn sidecar_url() -> String {
    format!("http://{SIDECAR_HOST}:{SIDECAR_PORT}")
}

#[tauri::command]
fn sidecar_ws_url() -> String {
    format!("ws://{SIDECAR_HOST}:{SIDECAR_PORT}")
}

/// Wait for the sidecar's HTTP server to start accepting connections, up to
/// `SIDECAR_BOOT_TIMEOUT`. Returns true if it's ready, false if we gave up.
fn wait_for_sidecar_health() -> bool {
    let deadline = std::time::Instant::now() + SIDECAR_BOOT_TIMEOUT;
    while std::time::Instant::now() < deadline {
        if already_listening() {
            return true;
        }
        thread::sleep(Duration::from_millis(250));
    }
    false
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // The currently-running sidecar (None when between respawn attempts).
    let sidecar: Arc<Mutex<Option<Child>>> = Arc::new(Mutex::new(None));
    // Set during graceful shutdown so the watchdog stops trying to respawn.
    let shutting_down = Arc::new(AtomicBool::new(false));

    let sidecar_setup = sidecar.clone();
    let sidecar_exit = sidecar.clone();
    let sidecar_watch = sidecar.clone();
    let shutdown_watch = shutting_down.clone();
    let shutdown_exit = shutting_down.clone();

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .setup(move |app| {
            if already_listening() {
                eprintln!("[host] sidecar already on :{SIDECAR_PORT}, not respawning");
                return Ok(());
            }
            match spawn_sidecar() {
                Ok(child) => {
                    eprintln!("[host] sidecar started (pid {})", child.id());
                    *sidecar_setup.lock().unwrap() = Some(child);
                }
                Err(e) => eprintln!("[host] sidecar spawn skipped: {e}"),
            }

            // ----- Watchdog -----
            // Polls the sidecar's child handle every WATCHDOG_POLL. If the
            // process has exited (try_wait returns Some) AND we're not in
            // the middle of a graceful shutdown, spawn a replacement. The
            // frontend gets a `sidecar-restarted` event so it can clear any
            // in-flight job state and surface a banner.
            let app_handle = app.handle().clone();
            thread::spawn(move || {
                loop {
                    thread::sleep(WATCHDOG_POLL);
                    if shutdown_watch.load(Ordering::SeqCst) {
                        break;
                    }
                    let needs_respawn = {
                        let mut guard = sidecar_watch.lock().unwrap();
                        match guard.as_mut() {
                            Some(child) => match child.try_wait() {
                                Ok(Some(status)) => {
                                    eprintln!(
                                        "[host] sidecar exited (status {status}); will respawn"
                                    );
                                    *guard = None;
                                    true
                                }
                                Ok(None) => false, // still running
                                Err(e) => {
                                    eprintln!("[host] try_wait error ({e}); treating as dead");
                                    *guard = None;
                                    true
                                }
                            },
                            None => {
                                // No child tracked. Either a previous respawn
                                // failed or someone else is on the port. If
                                // it's the latter, leave well enough alone.
                                !already_listening()
                            }
                        }
                    };

                    if !needs_respawn {
                        continue;
                    }
                    if shutdown_watch.load(Ordering::SeqCst) {
                        break;
                    }

                    eprintln!("[host] respawning sidecar...");
                    match spawn_sidecar() {
                        Ok(child) => {
                            eprintln!("[host] sidecar respawned (pid {})", child.id());
                            *sidecar_watch.lock().unwrap() = Some(child);

                            // Wait for HTTP to come up, then notify the UI.
                            let ready = wait_for_sidecar_health();
                            if ready {
                                let _ = app_handle.emit("sidecar-restarted", ());
                            } else {
                                eprintln!(
                                    "[host] sidecar respawned but not healthy after {}s",
                                    SIDECAR_BOOT_TIMEOUT.as_secs()
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!("[host] respawn failed: {e}");
                            // Back off a little before retrying so we don't
                            // burn CPU spinning on a misconfigured venv.
                            thread::sleep(Duration::from_secs(3));
                        }
                    }
                }
                eprintln!("[host] watchdog stopped (graceful shutdown)");
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![sidecar_url, sidecar_ws_url])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(move |_app_handle, event| {
            if matches!(event, RunEvent::ExitRequested { .. } | RunEvent::Exit) {
                shutdown_exit.store(true, Ordering::SeqCst);
                if let Some(mut child) = sidecar_exit.lock().unwrap().take() {
                    let _ = child.kill();
                }
            }
        });
}
