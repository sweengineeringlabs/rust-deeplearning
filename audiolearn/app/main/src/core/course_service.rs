//! In-memory course service implementation

use crate::api::*;
use crate::api::{Id, Result};
use crate::core::get_sample_courses;

/// In-memory course service
pub struct InMemoryCourseService {
    courses: Vec<Course>,
}

impl Default for InMemoryCourseService {
    fn default() -> Self {
        Self {
            courses: get_sample_courses(),
        }
    }
}

impl InMemoryCourseService {
    pub fn new() -> Self {
        Self::default()
    }
}

impl CourseService for InMemoryCourseService {
    fn list_courses(&self) -> Result<Vec<Course>> {
        Ok(self.courses.clone())
    }
    
    fn get_course(&self, id: &Id) -> Result<Option<Course>> {
        Ok(self.courses.iter().find(|c| &c.id == id).cloned())
    }
    
    fn get_recommended(&self, _user_id: &Id) -> Result<Vec<Course>> {
        Ok(self.courses.clone())
    }
    
    fn search(&self, query: &str) -> Result<Vec<Course>> {
        let q = query.to_lowercase();
        Ok(self.courses.iter()
            .filter(|c| c.title.to_lowercase().contains(&q))
            .cloned()
            .collect())
    }
    
    fn get_by_category(&self, category: &str) -> Result<Vec<Course>> {
        Ok(self.courses.iter()
            .filter(|c| c.tags.contains(&category.to_string()))
            .cloned()
            .collect())
    }
}
