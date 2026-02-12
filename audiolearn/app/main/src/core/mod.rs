//! Core Layer - Business logic implementations
//!
//! Contains implementations of the API services, UI components,
//! app entry point, and page definitions.

mod course_service;
mod sample_data;
mod playback_state;
mod search;
mod settings;

#[cfg(feature = "desktop")]
mod rodio_player;
#[cfg(feature = "desktop")]
mod audio_test;
#[cfg(feature = "desktop")]
mod audio_context;

// Desktop TTS implementations
#[cfg(feature = "desktop")]
pub(crate) mod native_tts;
#[cfg(feature = "desktop")]
mod edge_tts;
#[cfg(feature = "desktop")]
pub(crate) mod tts_manager;

// Web TTS implementation
#[cfg(feature = "web")]
pub(crate) mod web_tts;

// UI modules (moved from facade)
pub mod app;
pub mod components;
pub mod pages;

#[cfg(test)]
mod create_tests;

pub use course_service::*;
pub use sample_data::*;
pub use playback_state::*;
pub use search::*;
pub use settings::*;

#[cfg(feature = "desktop")]
pub use rodio_player::*;
#[cfg(feature = "desktop")]
pub use audio_test::*;
#[cfg(feature = "desktop")]
pub use audio_context::*;

// TTS exports - platform specific
#[cfg(feature = "desktop")]
pub use native_tts::*;
#[cfg(feature = "desktop")]
pub use edge_tts::*;
#[cfg(feature = "desktop")]
pub use tts_manager::*;

#[cfg(feature = "web")]
pub use web_tts::*;
