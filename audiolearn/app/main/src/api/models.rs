//! Domain models

use serde::{Deserialize, Serialize};
use crate::api::{Achievement, Bookmark, Difficulty, Id, Progress, Seconds, Streak, Timestamp};

/// User profile
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct User {
    pub id: Id,
    pub name: String,
    pub email: String,
    pub avatar_url: Option<String>,
    pub streak: Streak,
    pub total_listen_time: Seconds,
    pub points: u32,
    pub achievements: Vec<Achievement>,
    pub enrolled_courses: Vec<Id>,
}

impl Default for User {
    fn default() -> Self {
        Self {
            id: "user_1".to_string(),
            name: "Learner".to_string(),
            email: "learner@example.com".to_string(),
            avatar_url: None,
            streak: Streak::default(),
            total_listen_time: 0,
            points: 0,
            achievements: vec![],
            enrolled_courses: vec![],
        }
    }
}

/// Course
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Course {
    pub id: Id,
    pub title: String,
    pub description: String,
    pub author: Author,
    pub cover_image: String,
    pub icon: String,
    pub difficulty: Difficulty,
    pub total_duration: Seconds,
    pub chapters: Vec<Chapter>,
    pub rating: f32,
    pub review_count: u32,
    pub tags: Vec<String>,
}

impl Course {
    pub fn lesson_count(&self) -> u32 {
        self.chapters.iter().map(|c| c.lessons.len() as u32).sum()
    }
    
    pub fn get_lesson(&self, lesson_id: &Id) -> Option<&Lesson> {
        for chapter in &self.chapters {
            if let Some(lesson) = chapter.lessons.iter().find(|l| &l.id == lesson_id) {
                return Some(lesson);
            }
        }
        None
    }
}

/// Course author
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Author {
    pub id: Id,
    pub name: String,
    pub bio: Option<String>,
    pub avatar_url: Option<String>,
}

/// Chapter within a course
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Chapter {
    pub id: Id,
    pub title: String,
    pub description: Option<String>,
    pub lessons: Vec<Lesson>,
}

impl Chapter {
    pub fn total_duration(&self) -> Seconds {
        self.lessons.iter().map(|l| l.duration).sum()
    }
}

/// Individual lesson
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Lesson {
    pub id: Id,
    pub chapter_id: Id,
    pub title: String,
    pub description: Option<String>,
    pub audio_url: String,
    pub duration: Seconds,
    pub transcript: Option<Vec<TranscriptSegment>>,
    pub quiz: Option<Quiz>,
    pub order: u32,
}

/// Transcript segment
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub start: Timestamp,
    pub end: Timestamp,
    pub text: String,
}

/// Quiz for a lesson
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Quiz {
    pub id: Id,
    pub questions: Vec<QuizQuestion>,
}

/// Quiz question
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuizQuestion {
    pub id: Id,
    pub audio_url: Option<String>, // Audio version of question
    pub text: String,
    pub options: Vec<QuizOption>,
    pub correct_index: usize,
    pub explanation: String,
}

/// Quiz option
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QuizOption {
    pub id: Id,
    pub audio_url: Option<String>, // Audio version of option
    pub text: String,
}

/// User's course progress
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CourseProgress {
    pub course_id: Id,
    pub user_id: Id,
    pub completed_lessons: Vec<Id>,
    pub current_lesson_id: Option<Id>,
    pub current_position: Seconds,
    pub bookmarks: Vec<Bookmark>,
}

impl CourseProgress {
    pub fn lesson_progress(&self, course: &Course) -> Progress {
        Progress::new(
            self.completed_lessons.len() as u32,
            course.lesson_count(),
        )
    }
}

/// Library - user's enrolled courses
#[derive(Clone, Debug, Default)]
pub struct Library {
    pub courses: Vec<Course>,
    pub progress: Vec<CourseProgress>,
}

impl Library {
    pub fn get_progress(&self, course_id: &Id) -> Option<&CourseProgress> {
        self.progress.iter().find(|p| &p.course_id == course_id)
    }
}
