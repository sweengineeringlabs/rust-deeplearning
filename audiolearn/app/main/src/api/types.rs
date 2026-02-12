//! Common types used across the application

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Unique identifier type
pub type Id = String;

/// Duration in seconds
pub type Seconds = u32;

/// Playback speed multiplier
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum PlaybackSpeed {
    Half,       // 0.5x
    ThreeQuarter, // 0.75x
    Normal,     // 1.0x
    OneQuarter, // 1.25x
    OneHalf,    // 1.5x
    OneThreeQuarter, // 1.75x
    Double,     // 2.0x
}

impl PlaybackSpeed {
    pub fn as_f32(&self) -> f32 {
        match self {
            Self::Half => 0.5,
            Self::ThreeQuarter => 0.75,
            Self::Normal => 1.0,
            Self::OneQuarter => 1.25,
            Self::OneHalf => 1.5,
            Self::OneThreeQuarter => 1.75,
            Self::Double => 2.0,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Half => "0.5x",
            Self::ThreeQuarter => "0.75x",
            Self::Normal => "1x",
            Self::OneQuarter => "1.25x",
            Self::OneHalf => "1.5x",
            Self::OneThreeQuarter => "1.75x",
            Self::Double => "2x",
        }
    }
}

impl Default for PlaybackSpeed {
    fn default() -> Self {
        Self::Normal
    }
}

/// Playback state
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum PlaybackState {
    #[default]
    Stopped,
    Playing,
    Paused,
    Loading,
}

/// Progress tracking
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Progress {
    pub completed: u32,
    pub total: u32,
}

impl Progress {
    pub fn new(completed: u32, total: u32) -> Self {
        Self { completed, total }
    }

    pub fn percentage(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            (self.completed as f32 / self.total as f32) * 100.0
        }
    }

    pub fn is_complete(&self) -> bool {
        self.completed >= self.total
    }
}

/// Difficulty level
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
pub enum Difficulty {
    #[default]
    Beginner,
    Intermediate,
    Advanced,
}

impl Difficulty {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Beginner => "Beginner",
            Self::Intermediate => "Intermediate",
            Self::Advanced => "Advanced",
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            Self::Beginner => "📗",
            Self::Intermediate => "📙",
            Self::Advanced => "📕",
        }
    }
}

/// Timestamp for audio position
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct Timestamp {
    pub seconds: u32,
}

impl Timestamp {
    pub fn new(seconds: u32) -> Self {
        Self { seconds }
    }

    pub fn from_minutes(minutes: u32, secs: u32) -> Self {
        Self { seconds: minutes * 60 + secs }
    }

    pub fn format(&self) -> String {
        let mins = self.seconds / 60;
        let secs = self.seconds % 60;
        format!("{:02}:{:02}", mins, secs)
    }

    pub fn format_long(&self) -> String {
        let hours = self.seconds / 3600;
        let mins = (self.seconds % 3600) / 60;
        let secs = self.seconds % 60;

        if hours > 0 {
            format!("{}:{:02}:{:02}", hours, mins, secs)
        } else {
            format!("{:02}:{:02}", mins, secs)
        }
    }
}

/// Bookmark
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Bookmark {
    pub id: Id,
    pub lesson_id: Id,
    pub timestamp: Timestamp,
    pub note: Option<String>,
    pub created_at: DateTime<Utc>,
}

/// User streak
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Streak {
    pub current_days: u32,
    pub longest_days: u32,
    pub last_activity: DateTime<Utc>,
}

impl Default for Streak {
    fn default() -> Self {
        Self {
            current_days: 0,
            longest_days: 0,
            last_activity: Utc::now(),
        }
    }
}

/// Achievement
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Achievement {
    pub id: Id,
    pub name: String,
    pub description: String,
    pub icon: String,
    pub unlocked_at: Option<DateTime<Utc>>,
}

impl Achievement {
    pub fn is_unlocked(&self) -> bool {
        self.unlocked_at.is_some()
    }
}
