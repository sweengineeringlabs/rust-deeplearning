//! TTS Manager with Fallback
//!
//! Provides a unified interface for text-to-speech with automatic
//! fallback between Edge TTS (neural) and Native TTS (system).

use crate::api::{AudioLearnError, Result};
use crate::core::{EdgeTtsSync, NativeTts};
use crate::spi::tts::{SpeechOptions, TtsEngine, Voice};
use std::cell::RefCell;
use std::io::Cursor;
use rodio::{Decoder, OutputStream, Sink};

/// TTS engine preference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsPreference {
    /// Use Edge TTS (neural) if available, fallback to native
    EdgeFirst,
    /// Use native TTS (system) if available, fallback to Edge
    NativeFirst,
    /// Only use Edge TTS
    EdgeOnly,
    /// Only use native TTS  
    NativeOnly,
}

impl Default for TtsPreference {
    fn default() -> Self {
        Self::EdgeFirst
    }
}

/// TTS Manager with automatic fallback
pub struct TtsManager {
    edge: Option<EdgeTtsSync>,
    native: Option<NativeTts>,
    preference: TtsPreference,
    last_engine_used: Option<String>,
}

impl TtsManager {
    /// Create a new TTS manager with default preference (Edge first)
    pub fn new() -> Self {
        Self::with_preference(TtsPreference::default())
    }
    
    /// Create a TTS manager with specific preference
    pub fn with_preference(preference: TtsPreference) -> Self {
        // Initialize engines based on preference
        let edge = match preference {
            TtsPreference::NativeOnly => None,
            _ => EdgeTtsSync::try_new(),
        };
        
        let native = match preference {
            TtsPreference::EdgeOnly => None,
            _ => NativeTts::try_new(),
        };
        
        Self {
            edge,
            native,
            preference,
            last_engine_used: None,
        }
    }
    
    /// Check if any TTS engine is available
    pub fn is_available(&self) -> bool {
        self.edge.as_ref().map(|e| e.is_available()).unwrap_or(false)
            || self.native.as_ref().map(|n| n.is_available()).unwrap_or(false)
    }
    
    /// Get the name of the last engine used
    pub fn last_engine(&self) -> Option<&str> {
        self.last_engine_used.as_deref()
    }
    
    /// Set TTS preference
    pub fn set_preference(&mut self, preference: TtsPreference) {
        self.preference = preference;
    }
    
    /// Get available voices from all engines
    pub fn voices(&mut self) -> Result<Vec<Voice>> {
        let mut all_voices = Vec::new();
        
        // Get Edge voices
        if let Some(ref mut edge) = self.edge {
            if let Ok(voices) = edge.voices() {
                all_voices.extend(voices);
            }
        }
        
        // Get native voices
        if let Some(ref native) = self.native {
            if let Ok(voices) = native.voices() {
                all_voices.extend(voices);
            }
        }
        
        if all_voices.is_empty() {
            return Err(AudioLearnError::Tts("No TTS voices available".into()));
        }
        
        Ok(all_voices)
    }
    
    /// Synthesize text to audio bytes using the preferred engine
    pub fn synthesize(&mut self, text: &str, options: &SpeechOptions) -> Result<Vec<u8>> {
        match self.preference {
            TtsPreference::EdgeFirst => {
                // Try Edge first
                if let Some(ref edge) = self.edge {
                    match edge.synthesize(text, options) {
                        Ok(audio) => {
                            self.last_engine_used = Some(edge.name().to_string());
                            return Ok(audio);
                        }
                        Err(e) => {
                            eprintln!("Edge TTS failed, falling back to native: {}", e);
                        }
                    }
                }
                
                // Fallback to native (which doesn't support synthesis to bytes)
                Err(AudioLearnError::Tts(
                    "Edge TTS failed and native TTS doesn't support synthesis to bytes".into()
                ))
            }
            TtsPreference::NativeFirst => {
                // Native doesn't support synthesis to bytes, try Edge
                if let Some(ref edge) = self.edge {
                    match edge.synthesize(text, options) {
                        Ok(audio) => {
                            self.last_engine_used = Some(edge.name().to_string());
                            return Ok(audio);
                        }
                        Err(e) => {
                            return Err(e);
                        }
                    }
                }
                
                Err(AudioLearnError::Tts("No synthesis-capable TTS available".into()))
            }
            TtsPreference::EdgeOnly => {
                self.edge
                    .as_ref()
                    .ok_or_else(|| AudioLearnError::Tts("Edge TTS not available".into()))?
                    .synthesize(text, options)
                    .map(|audio| {
                        self.last_engine_used = Some("Microsoft Edge Neural TTS".to_string());
                        audio
                    })
            }
            TtsPreference::NativeOnly => {
                Err(AudioLearnError::Tts(
                    "Native TTS doesn't support synthesis to bytes".into()
                ))
            }
        }
    }
    
    /// Speak text using the preferred engine
    pub fn speak(&mut self, text: &str, options: &SpeechOptions) -> Result<()> {
        match self.preference {
            TtsPreference::EdgeFirst => {
                // For Edge, synthesize then play
                if let Some(ref edge) = self.edge {
                    match edge.synthesize(text, options) {
                        Ok(audio) => {
                            self.last_engine_used = Some(edge.name().to_string());
                            return play_audio_bytes(&audio);
                        }
                        Err(e) => {
                            eprintln!("Edge TTS failed, falling back to native: {}", e);
                        }
                    }
                }
                
                // Fallback to native
                if let Some(ref mut native) = self.native {
                    self.last_engine_used = Some(native.name().to_string());
                    return native.speak(text, options);
                }
                
                Err(AudioLearnError::Tts("No TTS engine available".into()))
            }
            TtsPreference::NativeFirst => {
                // Try native first
                if let Some(ref mut native) = self.native {
                    if native.is_available() {
                        self.last_engine_used = Some(native.name().to_string());
                        return native.speak(text, options);
                    }
                }
                
                // Fallback to Edge
                if let Some(ref edge) = self.edge {
                    match edge.synthesize(text, options) {
                        Ok(audio) => {
                            self.last_engine_used = Some(edge.name().to_string());
                            return play_audio_bytes(&audio);
                        }
                        Err(e) => {
                            return Err(e);
                        }
                    }
                }
                
                Err(AudioLearnError::Tts("No TTS engine available".into()))
            }
            TtsPreference::EdgeOnly => {
                let audio = self.edge
                    .as_ref()
                    .ok_or_else(|| AudioLearnError::Tts("Edge TTS not available".into()))?
                    .synthesize(text, options)?;
                
                self.last_engine_used = Some("Microsoft Edge Neural TTS".to_string());
                play_audio_bytes(&audio)
            }
            TtsPreference::NativeOnly => {
                self.native
                    .as_mut()
                    .ok_or_else(|| AudioLearnError::Tts("Native TTS not available".into()))?
                    .speak(text, options)
                    .map(|_| {
                        self.last_engine_used = Some("Native TTS".to_string());
                    })
            }
        }
    }
    
    /// Stop any ongoing speech
    pub fn stop(&mut self) -> Result<()> {
        // Stop native TTS
        if let Some(ref mut native) = self.native {
            let _ = native.stop();
        }
        // Stop Edge TTS audio playback
        stop_edge_audio();
        Ok(())
    }
}

impl Default for TtsManager {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Shared Audio Playback for Edge TTS (stoppable)
// =============================================================================

use std::sync::atomic::{AtomicBool, Ordering};

// Global flag to signal stop
static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Stop any Edge TTS audio playback
fn stop_edge_audio() {
    STOP_REQUESTED.store(true, Ordering::SeqCst);
}

/// Play audio bytes through rodio with stoppable playback
fn play_audio_bytes(audio: &[u8]) -> Result<()> {
    // Clear stop flag
    STOP_REQUESTED.store(false, Ordering::SeqCst);
    
    // Create output stream
    let (_stream, handle) = OutputStream::try_default()
        .map_err(|e| AudioLearnError::Audio(format!("Failed to get output stream: {}", e)))?;
    
    // Create sink
    let sink = Sink::try_new(&handle)
        .map_err(|e| AudioLearnError::Audio(format!("Failed to create sink: {}", e)))?;
    
    // Decode audio
    let cursor = Cursor::new(audio.to_vec());
    let source = Decoder::new(cursor)
        .map_err(|e| AudioLearnError::Audio(format!("Failed to decode audio: {}", e)))?;
    
    sink.append(source);
    
    // Wait for playback to complete or stop signal
    while !sink.empty() {
        // Check if stop requested
        if STOP_REQUESTED.load(Ordering::SeqCst) {
            sink.stop();
            break;
        }
        
        // Small sleep to avoid busy waiting
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    
    Ok(())
}

// Thread-local TTS manager for global access
thread_local! {
    static TTS_MANAGER: RefCell<Option<TtsManager>> = const { RefCell::new(None) };
}

/// Initialize the global TTS manager
pub fn init_tts() {
    TTS_MANAGER.with(|mgr| {
        let mut mgr = mgr.borrow_mut();
        if mgr.is_none() {
            *mgr = Some(TtsManager::new());
        }
    });
}

/// Initialize TTS with specific preference
pub fn init_tts_with_preference(preference: TtsPreference) {
    TTS_MANAGER.with(|mgr| {
        *mgr.borrow_mut() = Some(TtsManager::with_preference(preference));
    });
}

/// Speak text using the global TTS manager
pub fn speak_text(text: &str) -> Result<()> {
    TTS_MANAGER.with(|mgr| {
        let mut mgr = mgr.borrow_mut();
        if mgr.is_none() {
            *mgr = Some(TtsManager::new());
        }
        mgr.as_mut().unwrap().speak(text, &SpeechOptions::default())
    })
}

/// Speak text with custom options
pub fn speak_text_with_options(text: &str, options: &SpeechOptions) -> Result<()> {
    TTS_MANAGER.with(|mgr| {
        let mut mgr = mgr.borrow_mut();
        if mgr.is_none() {
            *mgr = Some(TtsManager::new());
        }
        mgr.as_mut().unwrap().speak(text, options)
    })
}

/// Synthesize text to audio bytes
pub fn synthesize_text(text: &str) -> Result<Vec<u8>> {
    TTS_MANAGER.with(|mgr| {
        let mut mgr = mgr.borrow_mut();
        if mgr.is_none() {
            *mgr = Some(TtsManager::new());
        }
        mgr.as_mut().unwrap().synthesize(text, &SpeechOptions::default())
    })
}

/// Synthesize text with custom options
pub fn synthesize_text_with_options(text: &str, options: &SpeechOptions) -> Result<Vec<u8>> {
    TTS_MANAGER.with(|mgr| {
        let mut mgr = mgr.borrow_mut();
        if mgr.is_none() {
            *mgr = Some(TtsManager::new());
        }
        mgr.as_mut().unwrap().synthesize(text, options)
    })
}

/// Stop TTS playback
pub fn stop_tts() -> Result<()> {
    TTS_MANAGER.with(|mgr| {
        if let Some(ref mut manager) = *mgr.borrow_mut() {
            manager.stop()
        } else {
            Ok(())
        }
    })
}

/// Check if TTS is available
pub fn is_tts_available() -> bool {
    TTS_MANAGER.with(|mgr| {
        let mut mgr = mgr.borrow_mut();
        if mgr.is_none() {
            *mgr = Some(TtsManager::new());
        }
        mgr.as_ref().map(|m| m.is_available()).unwrap_or(false)
    })
}

/// Get available voices
pub fn get_tts_voices() -> Result<Vec<Voice>> {
    TTS_MANAGER.with(|mgr| {
        let mut mgr = mgr.borrow_mut();
        if mgr.is_none() {
            *mgr = Some(TtsManager::new());
        }
        mgr.as_mut()
            .ok_or_else(|| AudioLearnError::Tts("TTS not initialized".into()))?
            .voices()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tts_manager_creation() {
        let manager = TtsManager::new();
        println!("TTS available: {}", manager.is_available());
    }
    
    #[test]
    fn test_tts_voices() {
        let mut manager = TtsManager::new();
        if let Ok(voices) = manager.voices() {
            println!("Found {} total voices", voices.len());
            
            // Count neural voices
            let neural = voices.iter().filter(|v| v.is_neural).count();
            println!("  - {} neural voices", neural);
            println!("  - {} native voices", voices.len() - neural);
        }
    }
}
