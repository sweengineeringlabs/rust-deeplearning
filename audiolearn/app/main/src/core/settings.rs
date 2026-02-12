//! Settings and preferences persistence

use serde::{Deserialize, Serialize};
use crate::api::PlaybackSpeed;
use crate::core::playback_state::{PlaybackData, SleepTimer};

/// User preferences
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    /// Display preferences
    pub display: DisplaySettings,
    /// Playback preferences
    pub playback: PlaybackSettings,
    /// TTS preferences
    pub tts: TtsSettings,
    /// Notification preferences
    pub notifications: NotificationSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            display: DisplaySettings::default(),
            playback: PlaybackSettings::default(),
            tts: TtsSettings::default(),
            notifications: NotificationSettings::default(),
        }
    }
}

/// Display settings
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DisplaySettings {
    /// Dark mode enabled
    pub dark_mode: bool,
    /// Compact view
    pub compact_view: bool,
    /// Show transcript while playing
    pub show_transcript: bool,
    /// Font size multiplier (1.0 = normal)
    pub font_scale: f32,
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            dark_mode: true,
            compact_view: false,
            show_transcript: true,
            font_scale: 1.0,
        }
    }
}

/// Playback settings
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlaybackSettings {
    /// Default playback speed
    pub default_speed: PlaybackSpeed,
    /// Skip forward/backward interval in seconds
    pub skip_interval: u32,
    /// Auto-play next lesson
    pub auto_play_next: bool,
    /// Resume from last position
    pub resume_playback: bool,
    /// Default sleep timer
    pub default_sleep_timer: SleepTimer,
    /// Crossfade between lessons (seconds)
    pub crossfade_duration: u32,
}

impl Default for PlaybackSettings {
    fn default() -> Self {
        Self {
            default_speed: PlaybackSpeed::Normal,
            skip_interval: 15,
            auto_play_next: true,
            resume_playback: true,
            default_sleep_timer: SleepTimer::Off,
            crossfade_duration: 0,
        }
    }
}

/// TTS settings
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TtsSettings {
    /// Selected voice ID
    pub voice_id: Option<String>,
    /// Speech rate (0.5 to 2.0)
    pub rate: f32,
    /// Speech pitch (0.5 to 2.0)
    pub pitch: f32,
    /// Volume (0.0 to 1.0)
    pub volume: f32,
    /// Use Edge TTS (higher quality, requires internet)
    pub use_edge_tts: bool,
}

impl Default for TtsSettings {
    fn default() -> Self {
        Self {
            voice_id: None,
            rate: 1.0,
            pitch: 1.0,
            volume: 1.0,
            use_edge_tts: true,
        }
    }
}

/// Notification settings
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NotificationSettings {
    /// Daily reminder enabled
    pub daily_reminder: bool,
    /// Reminder time (hour of day, 0-23)
    pub reminder_hour: u8,
    /// Streak warning
    pub streak_warning: bool,
    /// Achievement notifications
    pub achievement_notifications: bool,
}

impl Default for NotificationSettings {
    fn default() -> Self {
        Self {
            daily_reminder: false,
            reminder_hour: 20,
            streak_warning: true,
            achievement_notifications: true,
        }
    }
}

/// Complete app state that can be persisted
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AppState {
    pub settings: Settings,
    pub playback_data: PlaybackData,
}

impl AppState {
    /// Save state to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
    
    /// Load state from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
    
    /// Save state to a file (desktop only)
    #[cfg(feature = "desktop")]
    pub fn save_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = self.to_json().map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }
    
    /// Load state from a file (desktop only)
    #[cfg(feature = "desktop")]
    pub fn load_from_file(path: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
    
    /// Get default file path for settings
    #[cfg(feature = "desktop")]
    pub fn default_path() -> std::path::PathBuf {
        dirs::data_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("audiolearn")
            .join("state.json")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_settings_serialization() {
        let settings = Settings::default();
        let json = serde_json::to_string(&settings).unwrap();
        let parsed: Settings = serde_json::from_str(&json).unwrap();
        
        assert_eq!(parsed.playback.default_speed, PlaybackSpeed::Normal);
    }
    
    #[test]
    fn test_app_state_roundtrip() {
        let state = AppState::default();
        let json = state.to_json().unwrap();
        let parsed = AppState::from_json(&json).unwrap();
        
        assert_eq!(parsed.settings.display.dark_mode, true);
    }
}
