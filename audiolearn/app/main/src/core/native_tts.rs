//! Native TTS Implementation
//!
//! Uses the `tts` crate for cross-platform native text-to-speech.
//! Supports Windows (SAPI), macOS (AppKit), and Linux (Speech Dispatcher).

use crate::api::{AudioLearnError, Result};
use crate::spi::tts::{SpeechOptions, TtsEngine, Voice, VoiceGender};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Native TTS engine using system voices
pub struct NativeTts {
    inner: Option<tts::Tts>,
    speaking: Arc<AtomicBool>,
}

impl NativeTts {
    /// Create a new native TTS engine
    pub fn new() -> Result<Self> {
        let inner = tts::Tts::default()
            .map_err(|e| AudioLearnError::Tts(format!("Failed to initialize native TTS: {}", e)))?;
        
        Ok(Self {
            inner: Some(inner),
            speaking: Arc::new(AtomicBool::new(false)),
        })
    }
    
    /// Try to create a new native TTS engine, returning None if not available
    pub fn try_new() -> Option<Self> {
        Self::new().ok()
    }
}

impl Default for NativeTts {
    fn default() -> Self {
        Self {
            inner: tts::Tts::default().ok(),
            speaking: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl TtsEngine for NativeTts {
    fn name(&self) -> &str {
        "Native TTS"
    }
    
    fn is_available(&self) -> bool {
        self.inner.is_some()
    }
    
    fn voices(&self) -> Result<Vec<Voice>> {
        let tts = self.inner.as_ref()
            .ok_or_else(|| AudioLearnError::Tts("TTS not initialized".into()))?;
        
        let system_voices = tts.voices()
            .map_err(|e| AudioLearnError::Tts(format!("Failed to get voices: {}", e)))?;
        
        Ok(system_voices.into_iter().map(|v| Voice {
            id: v.id().to_string(),
            name: v.name().to_string(),
            language: v.language().to_string(),
            gender: match v.gender() {
                Some(tts::Gender::Male) => VoiceGender::Male,
                Some(tts::Gender::Female) => VoiceGender::Female,
                None => VoiceGender::Neutral,
            },
            is_neural: false, // Native voices are typically not neural
        }).collect())
    }
    
    fn speak(&mut self, text: &str, options: &SpeechOptions) -> Result<()> {
        let tts = self.inner.as_mut()
            .ok_or_else(|| AudioLearnError::Tts("TTS not initialized".into()))?;
        
        // Apply options
        if let Err(e) = tts.set_rate(options.rate) {
            eprintln!("Warning: Could not set rate: {}", e);
        }
        if let Err(e) = tts.set_pitch(options.pitch) {
            eprintln!("Warning: Could not set pitch: {}", e);
        }
        if let Err(e) = tts.set_volume(options.volume) {
            eprintln!("Warning: Could not set volume: {}", e);
        }
        
        self.speaking.store(true, Ordering::SeqCst);
        
        tts.speak(text, false)
            .map_err(|e| AudioLearnError::Tts(format!("Failed to speak: {}", e)))?;
        
        self.speaking.store(false, Ordering::SeqCst);
        Ok(())
    }
    
    fn stop(&mut self) -> Result<()> {
        if let Some(tts) = self.inner.as_mut() {
            tts.stop()
                .map_err(|e| AudioLearnError::Tts(format!("Failed to stop: {}", e)))?;
        }
        self.speaking.store(false, Ordering::SeqCst);
        Ok(())
    }
    
    fn synthesize(&self, _text: &str, _options: &SpeechOptions) -> Result<Vec<u8>> {
        // Native TTS doesn't support direct synthesis to bytes
        // This would require platform-specific code or file-based workarounds
        Err(AudioLearnError::Tts(
            "Native TTS does not support synthesis to bytes. Use speak() instead or use Edge TTS.".into()
        ))
    }
    
    fn is_speaking(&self) -> bool {
        self.speaking.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_native_tts_creation() {
        // This test may fail on systems without TTS
        let tts = NativeTts::try_new();
        if let Some(tts) = tts {
            assert!(tts.is_available());
            assert_eq!(tts.name(), "Native TTS");
        }
    }
    
    #[test]
    fn test_list_voices() {
        if let Some(tts) = NativeTts::try_new() {
            let voices = tts.voices();
            if let Ok(voices) = voices {
                println!("Found {} native voices", voices.len());
                for voice in voices.iter().take(5) {
                    println!("  - {} ({}, {:?})", voice.name, voice.language, voice.gender);
                }
            }
        }
    }
}
