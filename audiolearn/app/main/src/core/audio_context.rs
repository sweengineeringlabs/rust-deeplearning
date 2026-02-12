//! Audio playback context using thread-local storage
//!
//! On Windows, rodio's OutputStream is not Send+Sync, so we use
//! thread-local storage to manage the audio state.

use std::cell::RefCell;
use std::io::Cursor;
use rodio::{Decoder, OutputStream, Sink};
use crate::core::generate_speech_pattern;

thread_local! {
    static AUDIO_STATE: RefCell<Option<AudioState>> = const { RefCell::new(None) };
}

struct AudioState {
    sink: Sink,
    _stream: OutputStream,
}

/// Initialize audio and load test content
pub fn init_audio() {
    AUDIO_STATE.with(|state| {
        let mut state = state.borrow_mut();
        
        if state.is_some() {
            return; // Already initialized
        }
        
        if let Ok((stream, handle)) = OutputStream::try_default() {
            if let Ok(sink) = Sink::try_new(&handle) {
                // Generate 60 seconds of test audio
                let audio_data = generate_speech_pattern(60.0);
                
                if let Ok(source) = Decoder::new(Cursor::new(audio_data)) {
                    sink.append(source);
                    sink.pause();
                    
                    *state = Some(AudioState {
                        sink,
                        _stream: stream,
                    });
                }
            }
        }
    });
}

/// Play audio
pub fn play_audio() {
    AUDIO_STATE.with(|state| {
        if let Some(s) = state.borrow().as_ref() {
            s.sink.play();
        }
    });
}

/// Pause audio
pub fn pause_audio() {
    AUDIO_STATE.with(|state| {
        if let Some(s) = state.borrow().as_ref() {
            s.sink.pause();
        }
    });
}

/// Check if audio is playing
pub fn is_audio_playing() -> bool {
    AUDIO_STATE.with(|state| {
        state.borrow()
            .as_ref()
            .map(|s| !s.sink.is_paused())
            .unwrap_or(false)
    })
}

/// Stop audio
pub fn stop_audio() {
    AUDIO_STATE.with(|state| {
        if let Some(s) = state.borrow().as_ref() {
            s.sink.stop();
        }
    });
}

/// Set playback speed
pub fn set_audio_speed(speed: f32) {
    AUDIO_STATE.with(|state| {
        if let Some(s) = state.borrow().as_ref() {
            s.sink.set_speed(speed);
        }
    });
}

/// Set volume (0.0 to 1.0)
pub fn set_audio_volume(volume: f32) {
    AUDIO_STATE.with(|state| {
        if let Some(s) = state.borrow().as_ref() {
            s.sink.set_volume(volume);
        }
    });
}
