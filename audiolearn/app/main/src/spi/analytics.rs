//! Analytics SPI

use crate::api::{Id, Result, Seconds};

/// Analytics tracking interface
pub trait Analytics: Send + Sync {
    /// Track lesson started
    fn track_lesson_start(&self, user_id: &Id, lesson_id: &Id) -> Result<()>;
    
    /// Track lesson progress
    fn track_progress(&self, user_id: &Id, lesson_id: &Id, position: Seconds) -> Result<()>;
    
    /// Track lesson completed
    fn track_lesson_complete(&self, user_id: &Id, lesson_id: &Id) -> Result<()>;
    
    /// Track course completed
    fn track_course_complete(&self, user_id: &Id, course_id: &Id) -> Result<()>;
    
    /// Track quiz answered
    fn track_quiz_answer(&self, user_id: &Id, quiz_id: &Id, correct: bool) -> Result<()>;
    
    /// Track bookmark created
    fn track_bookmark(&self, user_id: &Id, lesson_id: &Id, timestamp: Seconds) -> Result<()>;
    
    /// Track time spent
    fn track_time_spent(&self, user_id: &Id, seconds: Seconds) -> Result<()>;
}

/// Noop analytics (for when analytics is disabled)
pub struct NoopAnalytics;

impl Analytics for NoopAnalytics {
    fn track_lesson_start(&self, _: &Id, _: &Id) -> Result<()> { Ok(()) }
    fn track_progress(&self, _: &Id, _: &Id, _: Seconds) -> Result<()> { Ok(()) }
    fn track_lesson_complete(&self, _: &Id, _: &Id) -> Result<()> { Ok(()) }
    fn track_course_complete(&self, _: &Id, _: &Id) -> Result<()> { Ok(()) }
    fn track_quiz_answer(&self, _: &Id, _: &Id, _: bool) -> Result<()> { Ok(()) }
    fn track_bookmark(&self, _: &Id, _: &Id, _: Seconds) -> Result<()> { Ok(()) }
    fn track_time_spent(&self, _: &Id, _: Seconds) -> Result<()> { Ok(()) }
}
