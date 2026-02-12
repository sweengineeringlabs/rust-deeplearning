//! AudioLearn Tauri Backend
//!
//! This module provides the native backend for AudioLearn using Tauri v2.
//! It includes TTS functionality, file system access, and state management.

pub mod api;
mod core;
mod saf;

use core::commands::*;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let builder = tauri::Builder::default();

    // Add plugins
    let builder = builder
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_store::Builder::new().build());

    // Add logging in debug mode
    #[cfg(debug_assertions)]
    let builder = builder.plugin(
        tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
    );

    // Desktop-specific plugins
    #[cfg(not(any(target_os = "android", target_os = "ios")))]
    let builder = builder
        .plugin(tauri_plugin_single_instance::init(|app, argv, cwd| {
            log::info!("Single instance triggered: {:?} {:?}", argv, cwd);
            // Focus the main window
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            None,
        ));

    builder
        .setup(|app| {
            log::info!("AudioLearn starting...");

            // Get the main window
            if let Some(window) = app.get_webview_window("main") {
                log::info!("Main window created");

                // Set minimum size
                let _ = window.set_min_size(Some(tauri::Size::Logical(tauri::LogicalSize {
                    width: 800.0,
                    height: 600.0,
                })));
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            speak_text,
            stop_tts,
            get_voices,
            save_app_state,
            load_app_state,
            get_app_version,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
