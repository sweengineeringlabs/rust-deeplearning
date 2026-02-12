//! Service interfaces

use crate::api::{Id, Result, Seconds};
use crate::api::{Course, CourseProgress, Lesson, User, Library};

/// Course service interface
pub trait CourseService: Send + Sync {
    /// Get all available courses
    fn list_courses(&self) -> Result<Vec<Course>>;
    
    /// Get course by ID
    fn get_course(&self, id: &Id) -> Result<Option<Course>>;
    
    /// Get recommended courses for user
    fn get_recommended(&self, user_id: &Id) -> Result<Vec<Course>>;
    
    /// Search courses
    fn search(&self, query: &str) -> Result<Vec<Course>>;
    
    /// Get course by category
    fn get_by_category(&self, category: &str) -> Result<Vec<Course>>;
}

/// Progress service interface
pub trait ProgressService: Send + Sync {
    /// Get user's progress for a course
    fn get_progress(&self, user_id: &Id, course_id: &Id) -> Result<CourseProgress>;
    
    /// Update lesson position
    fn update_position(&self, user_id: &Id, lesson_id: &Id, position: Seconds) -> Result<()>;
    
    /// Mark lesson as complete
    fn complete_lesson(&self, user_id: &Id, lesson_id: &Id) -> Result<()>;
    
    /// Get user's library
    fn get_library(&self, user_id: &Id) -> Result<Library>;
    
    /// Enroll in course
    fn enroll(&self, user_id: &Id, course_id: &Id) -> Result<()>;
}

/// User service interface
pub trait UserService: Send + Sync {
    /// Get current user
    fn get_current(&self) -> Result<User>;
    
    /// Update streak
    fn update_streak(&self, user_id: &Id) -> Result<()>;
    
    /// Add points
    fn add_points(&self, user_id: &Id, points: u32) -> Result<()>;
    
    /// Check and unlock achievements
    fn check_achievements(&self, user_id: &Id) -> Result<Vec<String>>;
}

/// Playback service interface
pub trait PlaybackService: Send + Sync {
    /// Start playing a lesson
    fn play_lesson(&mut self, lesson: &Lesson) -> Result<()>;
    
    /// Get current playback state
    fn get_state(&self) -> PlaybackServiceState;
    
    /// Control playback
    fn play(&mut self) -> Result<()>;
    fn pause(&mut self) -> Result<()>;
    fn stop(&mut self) -> Result<()>;
    fn seek(&mut self, position: Seconds) -> Result<()>;
    fn skip_forward(&mut self) -> Result<()>;
    fn skip_backward(&mut self) -> Result<()>;
}

/// Playback state for the service
#[derive(Clone, Debug)]
pub struct PlaybackServiceState {
    pub current_lesson: Option<Lesson>,
    pub position: Seconds,
    pub duration: Seconds,
    pub is_playing: bool,
}
