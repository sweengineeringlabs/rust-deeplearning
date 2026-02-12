//! Facade re-exports and platform-dispatch functions

pub use crate::api::*;
pub use crate::core::app::*;
pub use crate::core::components::*;
pub use crate::core::pages::*;

/// Speak text using the appropriate TTS for the platform
pub fn speak_text(text: &str) -> crate::api::error::Result<()> {
    #[cfg(feature = "desktop")]
    {
        use crate::spi::tts::TtsEngine;
        let mut tts = crate::core::native_tts::NativeTts::default();
        tts.speak(text, &crate::spi::tts::SpeechOptions::default())
    }
    #[cfg(feature = "web")]
    {
        crate::core::web_tts::web_speak_text(text)
    }
    #[cfg(not(any(feature = "desktop", feature = "web")))]
    {
        let _ = text;
        Err(crate::api::error::AppError::Tts("No TTS available".into()))
    }
}

/// Stop TTS playback
pub fn stop_tts() -> crate::api::error::Result<()> {
    #[cfg(feature = "desktop")]
    {
        use crate::spi::tts::TtsEngine;
        let mut tts = crate::core::native_tts::NativeTts::default();
        tts.stop()
    }
    #[cfg(feature = "web")]
    {
        crate::core::web_tts::web_stop_tts()
    }
    #[cfg(not(any(feature = "desktop", feature = "web")))]
    {
        Ok(())
    }
}

/// Get available TTS voices
pub fn get_tts_voices() -> crate::api::error::Result<Vec<crate::spi::tts::Voice>> {
    #[cfg(feature = "desktop")]
    {
        crate::core::tts_manager::get_tts_voices()
    }
    #[cfg(feature = "web")]
    {
        use crate::spi::tts::TtsEngine;
        let tts = crate::core::web_tts::WebTts::default();
        tts.voices()
    }
    #[cfg(not(any(feature = "desktop", feature = "web")))]
    {
        Ok(Vec::new())
    }
}
