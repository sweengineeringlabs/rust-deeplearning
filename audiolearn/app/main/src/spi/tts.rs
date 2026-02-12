//! Text-to-Speech SPI
//!
//! Trait definitions for TTS engines. Implementations can use
//! native system TTS, cloud-based TTS, or neural voice services.

use crate::api::Result;

/// Voice properties for TTS
#[derive(Debug, Clone, PartialEq)]
pub struct Voice {
    /// Unique identifier for the voice
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Language code (e.g., "en-US", "es-ES")
    pub language: String,
    /// Gender of the voice
    pub gender: VoiceGender,
    /// Whether this is a neural/AI voice
    pub is_neural: bool,
}

/// Voice gender
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoiceGender {
    Male,
    Female,
    Neutral,
}

/// Speech synthesis options
#[derive(Debug, Clone)]
pub struct SpeechOptions {
    /// Voice to use (None = default)
    pub voice: Option<Voice>,
    /// Speech rate (0.5 = half speed, 2.0 = double speed)
    pub rate: f32,
    /// Pitch adjustment (-1.0 to 1.0)
    pub pitch: f32,
    /// Volume (0.0 to 1.0)
    pub volume: f32,
}

impl Default for SpeechOptions {
    fn default() -> Self {
        Self {
            voice: None,
            rate: 1.0,
            pitch: 0.0,
            volume: 1.0,
        }
    }
}

/// Text-to-Speech engine interface
pub trait TtsEngine: Send + Sync {
    /// Get the name of this TTS engine
    fn name(&self) -> &str;
    
    /// Check if the engine is available and working
    fn is_available(&self) -> bool;
    
    /// Get available voices
    fn voices(&self) -> Result<Vec<Voice>>;
    
    /// Speak text immediately (blocks until complete)
    fn speak(&mut self, text: &str, options: &SpeechOptions) -> Result<()>;
    
    /// Stop speaking
    fn stop(&mut self) -> Result<()>;
    
    /// Convert text to audio bytes (WAV format)
    /// This is useful for playing through rodio or saving to file
    fn synthesize(&self, text: &str, options: &SpeechOptions) -> Result<Vec<u8>>;
    
    /// Check if currently speaking
    fn is_speaking(&self) -> bool;
}
