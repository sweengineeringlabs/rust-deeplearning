//! Search functionality for courses and lessons

use crate::api::{Course, Lesson};
use crate::api::Id;

/// Search result item
#[derive(Clone, Debug, PartialEq)]
pub enum SearchResult {
    Course(CourseSearchResult),
    Lesson(LessonSearchResult),
}

#[derive(Clone, Debug, PartialEq)]
pub struct CourseSearchResult {
    pub course: Course,
    pub match_score: f32,
    pub matched_fields: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LessonSearchResult {
    pub lesson: Lesson,
    pub course_id: Id,
    pub course_title: String,
    pub match_score: f32,
    pub matched_fields: Vec<String>,
}

/// Search engine for courses and lessons
#[derive(Clone, PartialEq)]
pub struct SearchEngine {
    courses: Vec<Course>,
}

impl SearchEngine {
    pub fn new(courses: Vec<Course>) -> Self {
        Self { courses }
    }
    
    pub fn update_courses(&mut self, courses: Vec<Course>) {
        self.courses = courses;
    }
    
    /// Search all content with a query string
    pub fn search(&self, query: &str) -> Vec<SearchResult> {
        if query.trim().is_empty() {
            return Vec::new();
        }
        
        let query_lower = query.to_lowercase();
        let terms: Vec<&str> = query_lower.split_whitespace().collect();
        
        let mut results = Vec::new();
        
        for course in &self.courses {
            // Search course
            if let Some(result) = self.search_course(course, &terms) {
                results.push(SearchResult::Course(result));
            }
            
            // Search lessons within course
            for chapter in &course.chapters {
                for lesson in &chapter.lessons {
                    if let Some(result) = self.search_lesson(lesson, course, &terms) {
                        results.push(SearchResult::Lesson(result));
                    }
                }
            }
        }
        
        // Sort by score (descending)
        results.sort_by(|a, b| {
            let score_a = match a {
                SearchResult::Course(c) => c.match_score,
                SearchResult::Lesson(l) => l.match_score,
            };
            let score_b = match b {
                SearchResult::Course(c) => c.match_score,
                SearchResult::Lesson(l) => l.match_score,
            };
            score_b.partial_cmp(&score_a).unwrap()
        });
        
        results
    }
    
    fn search_course(&self, course: &Course, terms: &[&str]) -> Option<CourseSearchResult> {
        let mut score = 0.0;
        let mut matched_fields = Vec::new();
        
        let title_lower = course.title.to_lowercase();
        let desc_lower = course.description.to_lowercase();
        let author_lower = course.author.name.to_lowercase();
        
        for term in terms {
            // Title match (highest weight)
            if title_lower.contains(term) {
                score += 10.0;
                if !matched_fields.contains(&"title".to_string()) {
                    matched_fields.push("title".to_string());
                }
            }
            
            // Author match
            if author_lower.contains(term) {
                score += 5.0;
                if !matched_fields.contains(&"author".to_string()) {
                    matched_fields.push("author".to_string());
                }
            }
            
            // Description match
            if desc_lower.contains(term) {
                score += 3.0;
                if !matched_fields.contains(&"description".to_string()) {
                    matched_fields.push("description".to_string());
                }
            }
            
            // Tag match
            for tag in &course.tags {
                if tag.to_lowercase().contains(term) {
                    score += 7.0;
                    if !matched_fields.contains(&"tags".to_string()) {
                        matched_fields.push("tags".to_string());
                    }
                    break;
                }
            }
        }
        
        if score > 0.0 {
            Some(CourseSearchResult {
                course: course.clone(),
                match_score: score,
                matched_fields,
            })
        } else {
            None
        }
    }
    
    fn search_lesson(&self, lesson: &Lesson, course: &Course, terms: &[&str]) -> Option<LessonSearchResult> {
        let mut score = 0.0;
        let mut matched_fields = Vec::new();
        
        let title_lower = lesson.title.to_lowercase();
        let desc_lower = lesson.description.as_ref().map(|d| d.to_lowercase()).unwrap_or_default();
        
        for term in terms {
            // Title match
            if title_lower.contains(term) {
                score += 8.0;
                if !matched_fields.contains(&"title".to_string()) {
                    matched_fields.push("title".to_string());
                }
            }
            
            // Description match
            if desc_lower.contains(term) {
                score += 2.0;
                if !matched_fields.contains(&"description".to_string()) {
                    matched_fields.push("description".to_string());
                }
            }
            
            // Transcript match
            if let Some(transcript) = &lesson.transcript {
                for segment in transcript {
                    if segment.text.to_lowercase().contains(term) {
                        score += 1.0;
                        if !matched_fields.contains(&"transcript".to_string()) {
                            matched_fields.push("transcript".to_string());
                        }
                        break;
                    }
                }
            }
        }
        
        if score > 0.0 {
            Some(LessonSearchResult {
                lesson: lesson.clone(),
                course_id: course.id.clone(),
                course_title: course.title.clone(),
                match_score: score,
                matched_fields,
            })
        } else {
            None
        }
    }
    
    /// Get search suggestions based on partial query
    pub fn suggest(&self, query: &str, limit: usize) -> Vec<String> {
        if query.trim().is_empty() {
            return Vec::new();
        }
        
        let query_lower = query.to_lowercase();
        let mut suggestions = Vec::new();
        
        // Collect course titles that start with or contain the query
        for course in &self.courses {
            if course.title.to_lowercase().starts_with(&query_lower) {
                suggestions.push(course.title.clone());
            }
            
            // Add tags
            for tag in &course.tags {
                if tag.to_lowercase().starts_with(&query_lower) && !suggestions.contains(tag) {
                    suggestions.push(tag.clone());
                }
            }
        }
        
        suggestions.truncate(limit);
        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Author;
    use crate::api::Difficulty;
    
    fn create_test_course() -> Course {
        Course {
            id: "c1".into(),
            title: "Rust Programming Basics".into(),
            description: "Learn Rust from scratch".into(),
            author: Author {
                id: "a1".into(),
                name: "Test Author".into(),
                bio: None,
                avatar_url: None,
            },
            cover_image: "".into(),
            icon: "🦀".into(),
            difficulty: Difficulty::Beginner,
            total_duration: 3600,
            chapters: vec![],
            rating: 4.5,
            review_count: 100,
            tags: vec!["rust".into(), "programming".into(), "systems".into()],
        }
    }
    
    #[test]
    fn test_search_by_title() {
        let engine = SearchEngine::new(vec![create_test_course()]);
        let results = engine.search("rust");
        
        assert!(!results.is_empty());
        if let SearchResult::Course(result) = &results[0] {
            assert!(result.matched_fields.contains(&"title".to_string()));
        }
    }
    
    #[test]
    fn test_search_by_tag() {
        let engine = SearchEngine::new(vec![create_test_course()]);
        let results = engine.search("systems");
        
        assert!(!results.is_empty());
        if let SearchResult::Course(result) = &results[0] {
            assert!(result.matched_fields.contains(&"tags".to_string()));
        }
    }
    
    #[test]
    fn test_suggestions() {
        let engine = SearchEngine::new(vec![create_test_course()]);
        let suggestions = engine.suggest("Rust", 5);
        
        assert!(!suggestions.is_empty());
    }
}
