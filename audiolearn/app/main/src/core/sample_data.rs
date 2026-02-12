//! Sample course data

use crate::api::*;
use crate::api::Difficulty;

/// Get sample courses
pub fn get_sample_courses() -> Vec<Course> {
    vec![
        create_rust_course(),
        create_python_course(),
        create_aws_course(),
    ]
}

fn create_rust_course() -> Course {
    Course {
        id: "rust-audio".to_string(),
        title: "Learn Rust by Ear".to_string(),
        description: "Master Rust programming through audio lessons.".to_string(),
        author: Author {
            id: "sarah".to_string(),
            name: "Sarah Chen".to_string(),
            bio: Some("Rust educator".to_string()),
            avatar_url: None,
        },
        cover_image: "rust_cover.png".to_string(),
        icon: "🦀".to_string(),
        difficulty: Difficulty::Beginner,
        total_duration: 31500,
        rating: 4.8,
        review_count: 2340,
        tags: vec!["rust".to_string(), "programming".to_string()],
        chapters: vec![
            Chapter {
                id: "ch1".to_string(),
                title: "Getting Started".to_string(),
                description: Some("Setup and first program".to_string()),
                lessons: vec![
                    create_lesson("l1.1", "ch1", "Welcome", 512, 1),
                    create_lesson("l1.2", "ch1", "Installing Rust", 765, 2),
                    create_lesson("l1.3", "ch1", "First Program", 920, 3),
                ],
            },
            Chapter {
                id: "ch2".to_string(),
                title: "Variables & Types".to_string(),
                description: None,
                lessons: vec![
                    create_lesson("l2.1", "ch2", "Variables", 1120, 1),
                    create_lesson("l2.2", "ch2", "Data Types", 1335, 2),
                ],
            },
            Chapter {
                id: "ch3".to_string(),
                title: "Ownership".to_string(),
                description: Some("Rust's unique feature".to_string()),
                lessons: vec![
                    create_lesson("l3.1", "ch3", "What is Ownership?", 1500, 1),
                    create_lesson("l3.2", "ch3", "Borrowing", 1725, 2),
                ],
            },
        ],
    }
}

fn create_python_course() -> Course {
    Course {
        id: "python-audio".to_string(),
        title: "Python Fundamentals".to_string(),
        description: "Learn Python through audio.".to_string(),
        author: Author {
            id: "alex".to_string(),
            name: "Alex Johnson".to_string(),
            bio: None,
            avatar_url: None,
        },
        cover_image: "python_cover.png".to_string(),
        icon: "🐍".to_string(),
        difficulty: Difficulty::Beginner,
        total_duration: 25200,
        rating: 4.6,
        review_count: 1850,
        tags: vec!["python".to_string()],
        chapters: vec![],
    }
}

fn create_aws_course() -> Course {
    Course {
        id: "aws-audio".to_string(),
        title: "AWS Cloud Essentials".to_string(),
        description: "Master AWS through audio.".to_string(),
        author: Author {
            id: "mike".to_string(),
            name: "Mike Roberts".to_string(),
            bio: None,
            avatar_url: None,
        },
        cover_image: "aws_cover.png".to_string(),
        icon: "☁️".to_string(),
        difficulty: Difficulty::Intermediate,
        total_duration: 36000,
        rating: 4.7,
        review_count: 980,
        tags: vec!["aws".to_string(), "cloud".to_string()],
        chapters: vec![],
    }
}

fn create_lesson(id: &str, chapter_id: &str, title: &str, duration: u32, order: u32) -> Lesson {
    Lesson {
        id: id.to_string(),
        chapter_id: chapter_id.to_string(),
        title: title.to_string(),
        description: None,
        audio_url: format!("https://cdn.audiolearn.dev/{}.mp3", id),
        duration,
        order,
        transcript: None,
        quiz: None,
    }
}
