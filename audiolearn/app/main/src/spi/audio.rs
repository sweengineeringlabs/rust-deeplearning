//! Audio Player SPI

use crate::api::{PlaybackSpeed, PlaybackState, Result, Seconds};

/// Audio player interface - to be implemented by platform-specific code
pub trait AudioPlayer: Send + Sync {
    /// Load audio from URL
    fn load(&mut self, url: &str) -> Result<()>;
    
    /// Play audio
    fn play(&mut self) -> Result<()>;
    
    /// Pause audio
    fn pause(&mut self) -> Result<()>;
    
    /// Stop and unload audio
    fn stop(&mut self) -> Result<()>;
    
    /// Seek to position
    fn seek(&mut self, position: Seconds) -> Result<()>;
    
    /// Skip forward
    fn skip_forward(&mut self, seconds: Seconds) -> Result<()>;
    
    /// Skip backward
    fn skip_backward(&mut self, seconds: Seconds) -> Result<()>;
    
    /// Set playback speed
    fn set_speed(&mut self, speed: PlaybackSpeed) -> Result<()>;
    
    /// Get current position
    fn position(&self) -> Seconds;
    
    /// Get total duration
    fn duration(&self) -> Seconds;
    
    /// Get current state
    fn state(&self) -> PlaybackState;
    
    /// Get current speed
    fn speed(&self) -> PlaybackSpeed;
}

/// Audio download interface
pub trait AudioDownloader: Send + Sync {
    /// Download audio file for offline use
    fn download(&self, url: &str, path: &str) -> Result<()>;
    
    /// Check if file is downloaded
    fn is_downloaded(&self, path: &str) -> bool;
    
    /// Delete downloaded file
    fn delete(&self, path: &str) -> Result<()>;
    
    /// Get download progress (0.0 - 1.0)
    fn progress(&self, url: &str) -> Option<f32>;
}
