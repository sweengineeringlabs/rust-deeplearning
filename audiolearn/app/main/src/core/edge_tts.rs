//! Microsoft Edge TTS Implementation
//!
//! Uses the `msedge-tts` crate for high-quality neural voices.
//! This provides access to Microsoft's neural text-to-speech voices
//! without requiring an API key.

use crate::api::{AudioLearnError, Result};
use crate::spi::tts::{SpeechOptions, Voice, VoiceGender};
use msedge_tts::{
    tts::{client::connect, SpeechConfig},
    voice::get_voices_list,
};

/// Microsoft Edge TTS engine using neural voices
pub struct EdgeTts {
    voices_cache: Option<Vec<Voice>>,
}

impl EdgeTts {
    /// Create a new Edge TTS engine
    pub fn new() -> Self {
        Self { voices_cache: None }
    }
    
    /// Get a recommended English voice
    pub fn recommended_english_voice() -> Voice {
        Voice {
            id: "en-US-AriaNeural".to_string(),
            name: "Aria".to_string(),
            language: "en-US".to_string(),
            gender: VoiceGender::Female,
            is_neural: true,
        }
    }
    
    /// Get a recommended male English voice
    pub fn recommended_english_male_voice() -> Voice {
        Voice {
            id: "en-US-GuyNeural".to_string(),
            name: "Guy".to_string(),
            language: "en-US".to_string(),
            gender: VoiceGender::Male,
            is_neural: true,
        }
    }
    
    /// Check if the engine is available
    pub fn is_available(&self) -> bool {
        // Try to fetch voices to check availability
        get_voices_list().is_ok()
    }
    
    /// Get available voices
    pub fn voices(&mut self) -> Result<Vec<Voice>> {
        // Return cached voices if available
        if let Some(ref cached) = self.voices_cache {
            return Ok(cached.clone());
        }
        
        let voice_list = get_voices_list()
            .map_err(|e| AudioLearnError::Tts(format!("Failed to fetch Edge voices: {}", e)))?;
        
        let voices: Vec<Voice> = voice_list
            .into_iter()
            .filter_map(|v| {
                // short_name is required for synthesizing, skip voices without it
                let short_name = v.short_name.as_ref()?;
                Some(Voice {
                    id: short_name.clone(),
                    name: v.name.clone(),
                    language: v.locale.clone().unwrap_or_else(|| "unknown".to_string()),
                    gender: v.gender.as_ref()
                        .map(|g| match g.as_str() {
                            "Male" => VoiceGender::Male,
                            "Female" => VoiceGender::Female,
                            _ => VoiceGender::Neutral,
                        })
                        .unwrap_or(VoiceGender::Neutral),
                    is_neural: short_name.contains("Neural"),
                })
            })
            .collect();
        
        self.voices_cache = Some(voices.clone());
        Ok(voices)
    }
    
    /// Synthesize text to audio bytes
    pub fn synthesize(&self, text: &str, options: &SpeechOptions) -> Result<Vec<u8>> {
        // Get voice configuration
        let voice_name = options
            .voice
            .as_ref()
            .map(|v| v.id.clone())
            .unwrap_or_else(|| "en-US-AriaNeural".to_string());
        
        // Create speech config manually
        let config = SpeechConfig {
            voice_name,
            // Use MP3 format for better compatibility
            audio_format: "audio-24khz-48kbitrate-mono-mp3".to_string(),
            // Rate: percentage from -100 to 100 (0 = normal)
            rate: ((options.rate - 1.0) * 100.0) as i32,
            // Pitch: Hz adjustment from -50 to 50 (0 = normal)
            pitch: (options.pitch * 50.0) as i32,
            // Volume: percentage from 0 to 100
            volume: (options.volume * 100.0) as i32,
        };
        
        // Connect and synthesize
        let mut tts = connect()
            .map_err(|e| AudioLearnError::Tts(format!("Failed to connect to Edge TTS: {}", e)))?;
        
        let audio_data = tts
            .synthesize(text, &config)
            .map_err(|e| AudioLearnError::Tts(format!("Failed to synthesize: {}", e)))?;
        
        // SynthesizedAudio has an audio_bytes field directly
        Ok(audio_data.audio_bytes)
    }
    
    /// Get the engine name
    pub fn name(&self) -> &str {
        "Microsoft Edge Neural TTS"
    }
}

impl Default for EdgeTts {
    fn default() -> Self {
        Self::new()
    }
}

/// Synchronous wrapper for EdgeTts (already sync, but provides consistent interface)
pub struct EdgeTtsSync {
    inner: EdgeTts,
}

impl EdgeTtsSync {
    /// Create a new sync wrapper for Edge TTS
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: EdgeTts::new(),
        })
    }
    
    /// Try to create a new sync wrapper, returning None on failure
    pub fn try_new() -> Option<Self> {
        Self::new().ok()
    }
    
    /// Get the engine name
    pub fn name(&self) -> &str {
        self.inner.name()
    }
    
    /// Check if available
    pub fn is_available(&self) -> bool {
        self.inner.is_available()
    }
    
    /// Get available voices
    pub fn voices(&mut self) -> Result<Vec<Voice>> {
        self.inner.voices()
    }
    
    /// Synthesize text to audio bytes
    pub fn synthesize(&self, text: &str, options: &SpeechOptions) -> Result<Vec<u8>> {
        self.inner.synthesize(text, options)
    }
    
    /// Get a recommended English voice
    pub fn recommended_english_voice() -> Voice {
        EdgeTts::recommended_english_voice()
    }
}

impl Default for EdgeTtsSync {
    fn default() -> Self {
        Self::new().expect("Failed to create EdgeTtsSync")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_edge_tts_voices() {
        let mut edge = EdgeTts::new();
        
        if edge.is_available() {
            let voices = edge.voices();
            if let Ok(voices) = voices {
                println!("Found {} Edge voices", voices.len());
                
                // Show some English neural voices
                let english_neural: Vec<_> = voices
                    .iter()
                    .filter(|v| v.language.starts_with("en-") && v.is_neural)
                    .take(10)
                    .collect();
                
                for voice in english_neural {
                    println!("  - {} ({}, {:?})", voice.name, voice.language, voice.gender);
                }
            }
        } else {
            println!("Edge TTS not available (no network?)");
        }
    }
    
    #[test]
    fn test_edge_tts_synthesize() {
        let edge = EdgeTts::new();
        
        if edge.is_available() {
            let options = SpeechOptions::default();
            let result = edge.synthesize("Hello, this is a test.", &options);
            
            if let Ok(audio) = result {
                println!("Synthesized {} bytes of audio", audio.len());
                assert!(!audio.is_empty());
            }
        }
    }
}
