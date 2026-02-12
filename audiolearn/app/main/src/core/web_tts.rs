//! Web TTS Implementation
//!
//! Uses the Web Speech API for text-to-speech in the browser.
//! This is only available when compiling for the web target.

use crate::api::{AudioLearnError, Result};
use crate::spi::tts::{SpeechOptions, TtsEngine, Voice, VoiceGender};

#[cfg(feature = "web")]
use wasm_bindgen::prelude::*;

/// Web TTS engine using the Web Speech API
pub struct WebTts {
    speaking: bool,
}

impl WebTts {
    /// Create a new Web TTS engine
    pub fn new() -> Result<Self> {
        Ok(Self { speaking: false })
    }
    
    /// Try to create a new Web TTS engine
    pub fn try_new() -> Option<Self> {
        Self::new().ok()
    }
}

impl Default for WebTts {
    fn default() -> Self {
        Self { speaking: false }
    }
}

impl TtsEngine for WebTts {
    fn name(&self) -> &str {
        "Web Speech API"
    }
    
    fn is_available(&self) -> bool {
        #[cfg(feature = "web")]
        {
            // Check if speechSynthesis is available
            if let Some(window) = web_sys::window() {
                return window.speech_synthesis().is_ok();
            }
            false
        }
        #[cfg(not(feature = "web"))]
        false
    }
    
    fn voices(&self) -> Result<Vec<Voice>> {
        #[cfg(feature = "web")]
        {
            use wasm_bindgen::JsCast;
            
            let window = web_sys::window()
                .ok_or_else(|| AudioLearnError::Tts("No window object".into()))?;
            let synth = window.speech_synthesis()
                .map_err(|_| AudioLearnError::Tts("SpeechSynthesis not available".into()))?;
            
            let js_voices = synth.get_voices();
            let mut voices = Vec::new();
            
            for i in 0..js_voices.length() {
                let voice_js = js_voices.get(i);
                if !voice_js.is_undefined() && !voice_js.is_null() {
                    if let Ok(voice) = voice_js.dyn_into::<web_sys::SpeechSynthesisVoice>() {
                        voices.push(Voice {
                            id: voice.voice_uri(),
                            name: voice.name(),
                            language: voice.lang(),
                            gender: VoiceGender::Neutral,
                            is_neural: false,
                        });
                    }
                }
            }
            
            Ok(voices)
        }
        #[cfg(not(feature = "web"))]
        Err(AudioLearnError::Tts("Web TTS not available on this platform".into()))
    }
    
    fn speak(&mut self, text: &str, options: &SpeechOptions) -> Result<()> {
        #[cfg(feature = "web")]
        {
            let window = web_sys::window()
                .ok_or_else(|| AudioLearnError::Tts("No window object".into()))?;
            let synth = window.speech_synthesis()
                .map_err(|_| AudioLearnError::Tts("SpeechSynthesis not available".into()))?;
            
            let utterance = web_sys::SpeechSynthesisUtterance::new_with_text(text)
                .map_err(|_| AudioLearnError::Tts("Failed to create utterance".into()))?;
            
            // Apply options
            utterance.set_rate(options.rate);
            utterance.set_pitch(options.pitch);
            utterance.set_volume(options.volume);
            
            synth.speak(&utterance);
            self.speaking = true;
            
            Ok(())
        }
        #[cfg(not(feature = "web"))]
        {
            let _ = (text, options);
            Err(AudioLearnError::Tts("Web TTS not available on this platform".into()))
        }
    }
    
    fn stop(&mut self) -> Result<()> {
        #[cfg(feature = "web")]
        {
            if let Some(window) = web_sys::window() {
                if let Ok(synth) = window.speech_synthesis() {
                    synth.cancel();
                }
            }
            self.speaking = false;
            Ok(())
        }
        #[cfg(not(feature = "web"))]
        {
            self.speaking = false;
            Ok(())
        }
    }
    
    fn synthesize(&self, _text: &str, _options: &SpeechOptions) -> Result<Vec<u8>> {
        Err(AudioLearnError::Tts(
            "Web Speech API does not support synthesis to bytes.".into()
        ))
    }
    
    fn is_speaking(&self) -> bool {
        self.speaking
    }
}

/// Speak text using Web Speech API (convenience function)
#[cfg(feature = "web")]
pub fn web_speak_text(text: &str) -> Result<()> {
    let mut tts = WebTts::new()?;
    tts.speak(text, &SpeechOptions::default())
}

/// Stop Web Speech API (convenience function)
#[cfg(feature = "web")]
pub fn web_stop_tts() -> Result<()> {
    let mut tts = WebTts::default();
    tts.stop()
}
