//! Tauri commands for AudioLearn native functionality

use serde::{Deserialize, Serialize};
use tauri::command;

/// Voice information returned to frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceInfo {
    pub id: String,
    pub name: String,
    pub language: String,
    pub is_neural: bool,
}

/// TTS options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsOptions {
    pub voice_id: Option<String>,
    pub rate: f32,
    pub pitch: f32,
    pub volume: f32,
}

impl Default for TtsOptions {
    fn default() -> Self {
        Self {
            voice_id: None,
            rate: 1.0,
            pitch: 1.0,
            volume: 1.0,
        }
    }
}

/// Result type for TTS operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsResult {
    pub success: bool,
    pub error: Option<String>,
}

/// Speak text using native TTS
#[command]
pub async fn speak_text(text: String, options: Option<TtsOptions>) -> TtsResult {
    let _options = options.unwrap_or_default();

    // Use native TTS
    match tts::Tts::default() {
        Ok(mut tts) => {
            // Set rate if specified
            if let Ok(rate_range) = tts.get_rate() {
                let _ = tts.set_rate(rate_range);
            }

            match tts.speak(&text, false) {
                Ok(_) => TtsResult {
                    success: true,
                    error: None,
                },
                Err(e) => TtsResult {
                    success: false,
                    error: Some(format!("TTS speak error: {}", e)),
                },
            }
        }
        Err(e) => TtsResult {
            success: false,
            error: Some(format!("TTS initialization error: {}", e)),
        },
    }
}

/// Stop TTS playback
#[command]
pub async fn stop_tts() -> TtsResult {
    match tts::Tts::default() {
        Ok(mut tts) => {
            match tts.stop() {
                Ok(_) => TtsResult {
                    success: true,
                    error: None,
                },
                Err(e) => TtsResult {
                    success: false,
                    error: Some(format!("TTS stop error: {}", e)),
                },
            }
        }
        Err(e) => TtsResult {
            success: false,
            error: Some(format!("TTS initialization error: {}", e)),
        },
    }
}

/// Get available TTS voices
#[command]
pub async fn get_voices() -> Result<Vec<VoiceInfo>, String> {
    match tts::Tts::default() {
        Ok(tts) => {
            match tts.voices() {
                Ok(voices) => {
                    let voice_infos: Vec<VoiceInfo> = voices
                        .into_iter()
                        .map(|v| VoiceInfo {
                            id: v.id().to_string(),
                            name: v.name().to_string(),
                            language: v.language().to_string(),
                            is_neural: v.name().to_lowercase().contains("neural"),
                        })
                        .collect();
                    Ok(voice_infos)
                }
                Err(e) => Err(format!("Failed to get voices: {}", e)),
            }
        }
        Err(e) => Err(format!("TTS initialization error: {}", e)),
    }
}

/// App state for persistent storage
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppState {
    pub last_course_id: Option<String>,
    pub last_lesson_id: Option<String>,
    pub last_position: u32,
    pub playback_speed: f32,
    pub volume: f32,
}

/// Save app state
#[command]
pub async fn save_app_state(state: AppState) -> Result<(), String> {
    log::info!("Saving app state: {:?}", state);
    // State will be saved using tauri-plugin-store
    Ok(())
}

/// Load app state
#[command]
pub async fn load_app_state() -> Result<AppState, String> {
    log::info!("Loading app state");
    // State will be loaded using tauri-plugin-store
    Ok(AppState::default())
}

/// Get app version
#[command]
pub fn get_app_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}
