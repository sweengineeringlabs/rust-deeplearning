//! Storage SPI

use crate::api::{Id, Result};
use serde::{de::DeserializeOwned, Serialize};

/// Key-value storage interface
pub trait Storage: Send + Sync {
    /// Get value by key
    fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>>;
    
    /// Set value by key
    fn set<T: Serialize>(&self, key: &str, value: &T) -> Result<()>;
    
    /// Delete value by key
    fn delete(&self, key: &str) -> Result<()>;
    
    /// Check if key exists
    fn exists(&self, key: &str) -> bool;
    
    /// Clear all storage
    fn clear(&self) -> Result<()>;
}

/// User data persistence
pub trait UserStorage: Send + Sync {
    /// Save user progress
    fn save_progress(&self, user_id: &Id, course_id: &Id, lesson_id: &Id, position: u32) -> Result<()>;
    
    /// Get user progress
    fn get_progress(&self, user_id: &Id, course_id: &Id, lesson_id: &Id) -> Result<Option<u32>>;
    
    /// Mark lesson complete
    fn mark_complete(&self, user_id: &Id, course_id: &Id, lesson_id: &Id) -> Result<()>;
    
    /// Get completed lessons
    fn get_completed(&self, user_id: &Id, course_id: &Id) -> Result<Vec<Id>>;
}
