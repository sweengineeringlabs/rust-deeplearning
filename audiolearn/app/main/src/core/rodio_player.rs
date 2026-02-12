//! Rodio-based audio player (placeholder for platform-specific implementation)
//!
//! Note: rodio's OutputStream is not Send+Sync on Windows, so we use
//! the audio_context module with global state instead.

// This module is kept as documentation for the intended API design.
// The actual implementation is in audio_context.rs
