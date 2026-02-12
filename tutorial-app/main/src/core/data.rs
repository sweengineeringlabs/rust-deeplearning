//! Lesson and quiz data structures

use serde::{Deserialize, Serialize};

/// A learning module containing multiple lessons
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Module {
    pub id: String,
    pub name: String,
    pub description: String,
    pub icon: String,
    pub lessons: Vec<Lesson>,
    pub required_level: u32,
}

/// A single lesson
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Lesson {
    pub id: String,
    pub title: String,
    pub description: String,
    pub content: Vec<ContentBlock>,
    pub quiz: Vec<QuizQuestion>,
    pub xp_reward: u32,
}

/// Content block types
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ContentBlock {
    Text(String),
    Code { language: String, code: String },
    Tip(String),
    Warning(String),
    Image { src: String, alt: String },
}

/// Quiz question types
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum QuizQuestion {
    MultipleChoice {
        id: String,
        question: String,
        options: Vec<String>,
        correct_index: usize,
        explanation: String,
    },
    FillInBlank {
        id: String,
        question: String,
        blank_text: String,  // Text with ___ for blanks
        answer: String,
        explanation: String,
    },
    TrueFalse {
        id: String,
        statement: String,
        is_true: bool,
        explanation: String,
    },
    CodeComplete {
        id: String,
        question: String,
        code_template: String,
        correct_answer: String,
        explanation: String,
    },
}

impl QuizQuestion {
    pub fn id(&self) -> &str {
        match self {
            QuizQuestion::MultipleChoice { id, .. } => id,
            QuizQuestion::FillInBlank { id, .. } => id,
            QuizQuestion::TrueFalse { id, .. } => id,
            QuizQuestion::CodeComplete { id, .. } => id,
        }
    }
}

/// Sample course data
pub fn get_rust_course() -> Vec<Module> {
    vec![
        Module {
            id: "basics".to_string(),
            name: "Rust Basics".to_string(),
            description: "Learn the fundamentals of Rust".to_string(),
            icon: "📚".to_string(),
            required_level: 1,
            lessons: vec![
                Lesson {
                    id: "variables".to_string(),
                    title: "Variables & Mutability".to_string(),
                    description: "Learn how to declare variables in Rust".to_string(),
                    xp_reward: 50,
                    content: vec![
                        ContentBlock::Text("In Rust, variables are **immutable by default**. This is one of Rust's key features for safety.".to_string()),
                        ContentBlock::Code {
                            language: "rust".to_string(),
                            code: "let x = 5; // immutable\nlet mut y = 10; // mutable - can be changed".to_string(),
                        },
                        ContentBlock::Tip("Use `mut` only when you need to change a variable's value.".to_string()),
                    ],
                    quiz: vec![
                        QuizQuestion::MultipleChoice {
                            id: "q1".to_string(),
                            question: "What keyword makes a variable mutable?".to_string(),
                            options: vec!["let".to_string(), "mut".to_string(), "var".to_string(), "const".to_string()],
                            correct_index: 1,
                            explanation: "The `mut` keyword makes a variable mutable.".to_string(),
                        },
                        QuizQuestion::TrueFalse {
                            id: "q2".to_string(),
                            statement: "Variables in Rust are mutable by default.".to_string(),
                            is_true: false,
                            explanation: "Variables are immutable by default in Rust. Use `mut` to make them mutable.".to_string(),
                        },
                    ],
                },
                Lesson {
                    id: "types".to_string(),
                    title: "Data Types".to_string(),
                    description: "Explore Rust's type system".to_string(),
                    xp_reward: 60,
                    content: vec![
                        ContentBlock::Text("Rust is a **statically typed** language. Every variable has a type.".to_string()),
                        ContentBlock::Code {
                            language: "rust".to_string(),
                            code: "let number: i32 = 42;\nlet pi: f64 = 3.14159;\nlet active: bool = true;\nlet letter: char = 'A';".to_string(),
                        },
                        ContentBlock::Tip("Rust can often infer types, but explicit annotations help readability.".to_string()),
                    ],
                    quiz: vec![
                        QuizQuestion::MultipleChoice {
                            id: "q1".to_string(),
                            question: "What type represents a 32-bit signed integer?".to_string(),
                            options: vec!["u32".to_string(), "i32".to_string(), "int".to_string(), "num".to_string()],
                            correct_index: 1,
                            explanation: "`i32` is a 32-bit signed integer. `u32` is unsigned.".to_string(),
                        },
                    ],
                },
            ],
        },
        Module {
            id: "ownership".to_string(),
            name: "Ownership".to_string(),
            description: "Master Rust's unique ownership system".to_string(),
            icon: "🔐".to_string(),
            required_level: 3,
            lessons: vec![
                Lesson {
                    id: "ownership_basics".to_string(),
                    title: "What is Ownership?".to_string(),
                    description: "Understanding Rust's memory management".to_string(),
                    xp_reward: 75,
                    content: vec![
                        ContentBlock::Text("Ownership is Rust's most unique feature. It enables memory safety without garbage collection.".to_string()),
                        ContentBlock::Code {
                            language: "rust".to_string(),
                            code: "let s1 = String::from(\"hello\");\nlet s2 = s1; // s1 is MOVED to s2\n// println!(\"{}\", s1); // ERROR: s1 is no longer valid".to_string(),
                        },
                        ContentBlock::Warning("After a move, the original variable is invalid!".to_string()),
                    ],
                    quiz: vec![
                        QuizQuestion::TrueFalse {
                            id: "q1".to_string(),
                            statement: "After moving a String to another variable, both variables are valid.".to_string(),
                            is_true: false,
                            explanation: "After a move, only the new owner is valid. The original variable is invalidated.".to_string(),
                        },
                    ],
                },
            ],
        },
        Module {
            id: "structs".to_string(),
            name: "Structs & Enums".to_string(),
            description: "Create custom data types".to_string(),
            icon: "🏗️".to_string(),
            required_level: 5,
            lessons: vec![
                Lesson {
                    id: "struct_basics".to_string(),
                    title: "Defining Structs".to_string(),
                    description: "Create your own data structures".to_string(),
                    xp_reward: 80,
                    content: vec![
                        ContentBlock::Text("Structs let you create custom types that group related data.".to_string()),
                        ContentBlock::Code {
                            language: "rust".to_string(),
                            code: "struct User {\n    name: String,\n    email: String,\n    age: u32,\n}\n\nlet user = User {\n    name: String::from(\"Alice\"),\n    email: String::from(\"alice@example.com\"),\n    age: 30,\n};".to_string(),
                        },
                    ],
                    quiz: vec![
                        QuizQuestion::MultipleChoice {
                            id: "q1".to_string(),
                            question: "What keyword is used to define a struct?".to_string(),
                            options: vec!["class".to_string(), "struct".to_string(), "type".to_string(), "data".to_string()],
                            correct_index: 1,
                            explanation: "The `struct` keyword defines a new structure type.".to_string(),
                        },
                    ],
                },
            ],
        },
    ]
}
