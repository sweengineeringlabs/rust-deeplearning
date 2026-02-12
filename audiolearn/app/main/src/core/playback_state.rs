//! Playback State Management
//!
//! Manages the global playback state including current position,
//! sleep timer, playback speed, and bookmarks.

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use crate::api::{Bookmark, Id, PlaybackSpeed, PlaybackState, Seconds, Timestamp};
use crate::api::{Course, Lesson};

/// Sleep timer options
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum SleepTimer {
    Off,
    Minutes5,
    Minutes10,
    Minutes15,
    Minutes30,
    Minutes45,
    Minutes60,
    EndOfLesson,
}

impl SleepTimer {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Minutes5 => "5 min",
            Self::Minutes10 => "10 min",
            Self::Minutes15 => "15 min",
            Self::Minutes30 => "30 min",
            Self::Minutes45 => "45 min",
            Self::Minutes60 => "60 min",
            Self::EndOfLesson => "End of lesson",
        }
    }
    
    pub fn as_seconds(&self) -> Option<u32> {
        match self {
            Self::Off => None,
            Self::Minutes5 => Some(5 * 60),
            Self::Minutes10 => Some(10 * 60),
            Self::Minutes15 => Some(15 * 60),
            Self::Minutes30 => Some(30 * 60),
            Self::Minutes45 => Some(45 * 60),
            Self::Minutes60 => Some(60 * 60),
            Self::EndOfLesson => None, // Handled specially
        }
    }
    
    pub fn all() -> Vec<Self> {
        vec![
            Self::Off,
            Self::Minutes5,
            Self::Minutes10,
            Self::Minutes15,
            Self::Minutes30,
            Self::Minutes45,
            Self::Minutes60,
            Self::EndOfLesson,
        ]
    }
}

impl Default for SleepTimer {
    fn default() -> Self {
        Self::Off
    }
}

/// Lesson playback progress
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LessonProgress {
    pub lesson_id: Id,
    pub course_id: Id,
    pub position: Seconds,
    pub duration: Seconds,
    pub completed: bool,
    pub last_played: DateTime<Utc>,
}

impl LessonProgress {
    pub fn new(lesson_id: Id, course_id: Id, duration: Seconds) -> Self {
        Self {
            lesson_id,
            course_id,
            position: 0,
            duration,
            completed: false,
            last_played: Utc::now(),
        }
    }
    
    pub fn percentage(&self) -> f32 {
        if self.duration == 0 {
            0.0
        } else {
            (self.position as f32 / self.duration as f32) * 100.0
        }
    }
    
    /// Mark as completed if position is within 10 seconds of duration
    pub fn update_position(&mut self, position: Seconds) {
        self.position = position;
        self.last_played = Utc::now();
        
        if position + 10 >= self.duration {
            self.completed = true;
        }
    }
}

/// Current playback context
#[derive(Clone, Debug, Default)]
pub struct PlaybackContext {
    pub state: PlaybackState,
    pub current_course: Option<Course>,
    pub current_lesson: Option<Lesson>,
    pub position: Seconds,
    pub speed: PlaybackSpeed,
    pub sleep_timer: SleepTimer,
    pub sleep_timer_remaining: Option<Seconds>,
}

impl PlaybackContext {
    pub fn play(&mut self, course: Course, lesson: Lesson) {
        self.current_course = Some(course);
        self.current_lesson = Some(lesson);
        self.state = PlaybackState::Playing;
        // Don't reset position - resume from where we left off
    }
    
    pub fn pause(&mut self) {
        self.state = PlaybackState::Paused;
    }
    
    pub fn resume(&mut self) {
        if self.current_lesson.is_some() {
            self.state = PlaybackState::Playing;
        }
    }
    
    pub fn stop(&mut self) {
        self.state = PlaybackState::Stopped;
    }
    
    pub fn seek(&mut self, position: Seconds) {
        self.position = position;
    }
    
    pub fn skip_forward(&mut self, seconds: Seconds) {
        if let Some(lesson) = &self.current_lesson {
            self.position = (self.position + seconds).min(lesson.duration);
        }
    }
    
    pub fn skip_backward(&mut self, seconds: Seconds) {
        self.position = self.position.saturating_sub(seconds);
    }
    
    pub fn set_speed(&mut self, speed: PlaybackSpeed) {
        self.speed = speed;
    }
    
    pub fn set_sleep_timer(&mut self, timer: SleepTimer) {
        self.sleep_timer = timer;
        self.sleep_timer_remaining = timer.as_seconds();
    }
    
    pub fn duration(&self) -> Seconds {
        self.current_lesson.as_ref().map(|l| l.duration).unwrap_or(0)
    }
    
    pub fn is_playing(&self) -> bool {
        matches!(self.state, PlaybackState::Playing)
    }
}

/// Persistent playback data
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PlaybackData {
    /// Progress per lesson
    pub lesson_progress: HashMap<Id, LessonProgress>,
    /// User bookmarks
    pub bookmarks: Vec<Bookmark>,
    /// Last played lesson ID (for resume)
    pub last_played_lesson: Option<Id>,
    /// Last played course ID
    pub last_played_course: Option<Id>,
    /// Preferred playback speed
    pub preferred_speed: PlaybackSpeed,
}

impl PlaybackData {
    pub fn get_progress(&self, lesson_id: &Id) -> Option<&LessonProgress> {
        self.lesson_progress.get(lesson_id)
    }
    
    pub fn update_progress(&mut self, lesson_id: Id, course_id: Id, position: Seconds, duration: Seconds) {
        let progress = self.lesson_progress
            .entry(lesson_id.clone())
            .or_insert_with(|| LessonProgress::new(lesson_id.clone(), course_id.clone(), duration));
        
        progress.update_position(position);
        self.last_played_lesson = Some(lesson_id);
        self.last_played_course = Some(course_id);
    }
    
    pub fn add_bookmark(&mut self, lesson_id: Id, timestamp: Timestamp, note: Option<String>) {
        let bookmark = Bookmark {
            id: format!("bm_{}", Utc::now().timestamp_millis()),
            lesson_id,
            timestamp,
            note,
            created_at: Utc::now(),
        };
        self.bookmarks.push(bookmark);
    }
    
    pub fn remove_bookmark(&mut self, bookmark_id: &Id) {
        self.bookmarks.retain(|b| &b.id != bookmark_id);
    }
    
    pub fn get_lesson_bookmarks(&self, lesson_id: &Id) -> Vec<&Bookmark> {
        self.bookmarks.iter().filter(|b| &b.lesson_id == lesson_id).collect()
    }
    
    pub fn completed_lesson_count(&self) -> usize {
        self.lesson_progress.values().filter(|p| p.completed).count()
    }
    
    pub fn total_listen_time(&self) -> Seconds {
        self.lesson_progress.values().map(|p| p.position).sum()
    }
    
    pub fn recently_played(&self, limit: usize) -> Vec<&LessonProgress> {
        let mut sorted: Vec<_> = self.lesson_progress.values().collect();
        sorted.sort_by(|a, b| b.last_played.cmp(&a.last_played));
        sorted.into_iter().take(limit).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lesson_progress_percentage() {
        let mut progress = LessonProgress::new("l1".into(), "c1".into(), 100);
        progress.update_position(50);
        assert_eq!(progress.percentage(), 50.0);
    }
    
    #[test]
    fn test_lesson_completion() {
        let mut progress = LessonProgress::new("l1".into(), "c1".into(), 100);
        assert!(!progress.completed);
        
        progress.update_position(95);  // Within 10 seconds of end
        assert!(progress.completed);
    }
    
    #[test]
    fn test_playback_data_bookmarks() {
        let mut data = PlaybackData::default();
        data.add_bookmark("l1".into(), Timestamp::new(60), Some("Important point".into()));
        
        let bookmarks = data.get_lesson_bookmarks(&"l1".into());
        assert_eq!(bookmarks.len(), 1);
        assert_eq!(bookmarks[0].note, Some("Important point".into()));
    }
    
    #[test]
    fn test_sleep_timer() {
        assert_eq!(SleepTimer::Minutes15.as_seconds(), Some(15 * 60));
        assert_eq!(SleepTimer::Off.as_seconds(), None);
    }
}
