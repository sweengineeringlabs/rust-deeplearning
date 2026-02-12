//! SPI Layer - Service Provider Interfaces
//! 
//! Traits for external service implementations (audio player, storage, API, etc.)
//! These can be swapped for different platforms (web, desktop, mobile).

mod audio;
mod storage;
mod analytics;
pub mod tts;

pub use audio::*;
pub use storage::*;
pub use analytics::*;
pub use tts::*;
