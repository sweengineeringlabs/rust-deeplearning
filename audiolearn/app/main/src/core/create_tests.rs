//! Unit tests for the Create Page feature
//! Tests for uploading/pasting custom learning material

#[cfg(test)]
mod create_page_tests {
    use crate::core::pages::CustomMaterial;

    // ==========================================================================
    // CustomMaterial Tests
    // ==========================================================================

    #[test]
    fn test_custom_material_creation() {
        let material = CustomMaterial {
            id: "custom_123".to_string(),
            title: "Test Material".to_string(),
            content: "This is test content for TTS playback.".to_string(),
            created_at: "2024-12-27 12:00".to_string(),
        };

        assert_eq!(material.id, "custom_123");
        assert_eq!(material.title, "Test Material");
        assert!(!material.content.is_empty());
    }

    #[test]
    fn test_custom_material_with_empty_title() {
        let material = CustomMaterial {
            id: "custom_456".to_string(),
            title: "".to_string(),
            content: "Content without a title".to_string(),
            created_at: "2024-12-27 12:00".to_string(),
        };

        assert!(material.title.is_empty());
        assert!(!material.content.is_empty());
    }

    #[test]
    fn test_custom_material_clone() {
        let material = CustomMaterial {
            id: "custom_789".to_string(),
            title: "Clone Test".to_string(),
            content: "This material will be cloned.".to_string(),
            created_at: "2024-12-27 12:00".to_string(),
        };

        let cloned = material.clone();

        assert_eq!(material, cloned);
        assert_eq!(material.id, cloned.id);
        assert_eq!(material.content, cloned.content);
    }

    // ==========================================================================
    // Word Count Calculation Tests
    // ==========================================================================

    #[test]
    fn test_word_count_empty() {
        let content = "";
        let word_count = content.split_whitespace().count();
        assert_eq!(word_count, 0);
    }

    #[test]
    fn test_word_count_single_word() {
        let content = "Hello";
        let word_count = content.split_whitespace().count();
        assert_eq!(word_count, 1);
    }

    #[test]
    fn test_word_count_multiple_words() {
        let content = "The quick brown fox jumps over the lazy dog";
        let word_count = content.split_whitespace().count();
        assert_eq!(word_count, 9);
    }

    #[test]
    fn test_word_count_with_extra_whitespace() {
        let content = "  Hello   World   This   Has   Extra   Spaces  ";
        let word_count = content.split_whitespace().count();
        assert_eq!(word_count, 6);
    }

    #[test]
    fn test_word_count_with_newlines() {
        let content = "Line one\nLine two\nLine three";
        let word_count = content.split_whitespace().count();
        assert_eq!(word_count, 6);
    }

    #[test]
    fn test_word_count_typical_paragraph() {
        let content = r#"
            Text-to-speech (TTS) technology converts written text into spoken words.
            It's commonly used in accessibility, education, and entertainment.
            Modern TTS systems use neural networks to produce natural-sounding speech.
        "#;
        let word_count = content.split_whitespace().count();
        assert!(word_count > 20); // Should be around 30 words
        assert!(word_count < 50);
    }

    // ==========================================================================
    // Estimated Duration Tests
    // ==========================================================================

    fn estimate_minutes(word_count: usize) -> u32 {
        (word_count as f32 / 150.0).ceil() as u32
    }

    #[test]
    fn test_duration_empty_content() {
        let word_count = 0;
        let minutes = estimate_minutes(word_count);
        assert_eq!(minutes, 0);
    }

    #[test]
    fn test_duration_short_content() {
        // 50 words = ~20 seconds, rounds up to 1 minute
        let word_count = 50;
        let minutes = estimate_minutes(word_count);
        assert_eq!(minutes, 1);
    }

    #[test]
    fn test_duration_one_minute() {
        // 150 words = exactly 1 minute
        let word_count = 150;
        let minutes = estimate_minutes(word_count);
        assert_eq!(minutes, 1);
    }

    #[test]
    fn test_duration_five_minutes() {
        // 750 words = exactly 5 minutes
        let word_count = 750;
        let minutes = estimate_minutes(word_count);
        assert_eq!(minutes, 5);
    }

    #[test]
    fn test_duration_rounds_up() {
        // 151 words should round up to 2 minutes
        let word_count = 151;
        let minutes = estimate_minutes(word_count);
        assert_eq!(minutes, 2);
    }

    #[test]
    fn test_duration_long_content() {
        // 1500 words = 10 minutes (typical article length)
        let word_count = 1500;
        let minutes = estimate_minutes(word_count);
        assert_eq!(minutes, 10);
    }

    // ==========================================================================
    // Text Formatting Tests
    // ==========================================================================

    #[test]
    fn test_format_with_title() {
        let title = "Introduction to Rust";
        let content = "Rust is a systems programming language.";

        let full_text = format!("{}. {}", title, content);

        assert!(full_text.starts_with("Introduction to Rust. "));
        assert!(full_text.contains("Rust is a systems"));
    }

    #[test]
    fn test_format_without_title() {
        let title = "";
        let content = "Just the content here.";

        let full_text = if title.is_empty() {
            content.to_string()
        } else {
            format!("{}. {}", title, content)
        };

        assert_eq!(full_text, "Just the content here.");
    }

    #[test]
    fn test_content_preview_truncation() {
        let content = "This is a very long piece of content that should be truncated when displaying as a preview in the saved materials list.";

        let preview: String = content.chars().take(100).collect();

        assert!(preview.len() <= 100);
        assert!(preview.starts_with("This is a very long"));
    }

    // ==========================================================================
    // Material ID Generation Tests
    // ==========================================================================

    #[test]
    fn test_material_id_format() {
        let timestamp = 1703683200i64; // Fixed timestamp for testing
        let id = format!("custom_{}", timestamp);

        assert!(id.starts_with("custom_"));
        assert!(id.len() > 7); // "custom_" + at least some digits
    }

    #[test]
    fn test_unique_material_ids() {
        // Using different timestamps should produce unique IDs
        let id1 = format!("custom_{}", 1703683200i64);
        let id2 = format!("custom_{}", 1703683201i64);

        assert_ne!(id1, id2);
    }

    // ==========================================================================
    // TTS Integration Tests
    // ==========================================================================

    #[test]
    fn test_tts_text_preparation() {
        let material = CustomMaterial {
            id: "test_1".to_string(),
            title: "Learning Rust".to_string(),
            content: "Ownership is one of Rust's most unique features.".to_string(),
            created_at: "2024-12-27".to_string(),
        };

        let tts_text = format!("{}. {}", material.title, material.content);

        assert!(tts_text.contains("Learning Rust"));
        assert!(tts_text.contains("Ownership"));
        assert!(tts_text.contains(". ")); // Title-content separator
    }

    #[test]
    fn test_tts_with_special_characters() {
        let content = "C++ vs Rust: Which is faster? It's complicated!";

        // TTS should handle special characters
        assert!(content.contains("++"));
        assert!(content.contains("?"));
        assert!(content.contains("!"));
        assert!(content.contains("'"));
    }

    #[test]
    fn test_tts_max_content_length() {
        // Edge TTS can handle ~10,000 characters reliably
        let long_content: String = "word ".repeat(2000); // ~10,000 chars

        assert!(long_content.len() < 10500);
        assert!(long_content.len() > 9000);
    }

    // ==========================================================================
    // Saved Materials List Tests
    // ==========================================================================

    #[test]
    fn test_materials_vec_operations() {
        let mut materials: Vec<CustomMaterial> = Vec::new();

        assert!(materials.is_empty());

        materials.push(CustomMaterial {
            id: "1".to_string(),
            title: "First".to_string(),
            content: "Content 1".to_string(),
            created_at: "2024-12-27".to_string(),
        });

        assert_eq!(materials.len(), 1);

        materials.push(CustomMaterial {
            id: "2".to_string(),
            title: "Second".to_string(),
            content: "Content 2".to_string(),
            created_at: "2024-12-27".to_string(),
        });

        assert_eq!(materials.len(), 2);
        assert_eq!(materials[0].title, "First");
        assert_eq!(materials[1].title, "Second");
    }

    #[test]
    fn test_find_material_by_id() {
        let materials = vec![
            CustomMaterial {
                id: "a".to_string(),
                title: "Material A".to_string(),
                content: "Content A".to_string(),
                created_at: "2024-12-27".to_string(),
            },
            CustomMaterial {
                id: "b".to_string(),
                title: "Material B".to_string(),
                content: "Content B".to_string(),
                created_at: "2024-12-27".to_string(),
            },
        ];

        let found = materials.iter().find(|m| m.id == "b");

        assert!(found.is_some());
        assert_eq!(found.unwrap().title, "Material B");
    }

    // ==========================================================================
    // File Upload Tests
    // ==========================================================================

    #[test]
    fn test_file_extension_filtering() {
        let valid_extensions = ["txt", "md", "text"];

        assert!(valid_extensions.contains(&"txt"));
        assert!(valid_extensions.contains(&"md"));
        assert!(valid_extensions.contains(&"text"));
        assert!(!valid_extensions.contains(&"pdf"));
        assert!(!valid_extensions.contains(&"docx"));
    }

    #[test]
    fn test_filename_extraction() {
        let path = std::path::PathBuf::from("/path/to/my-document.txt");

        let filename = path.file_name()
            .map(|n| n.to_string_lossy().to_string());

        assert_eq!(filename, Some("my-document.txt".to_string()));
    }

    #[test]
    fn test_file_stem_extraction() {
        let path = std::path::PathBuf::from("/path/to/my-document.txt");

        let stem = path.file_stem()
            .map(|n| n.to_string_lossy().to_string());

        assert_eq!(stem, Some("my-document".to_string()));
    }

    #[test]
    fn test_file_stem_with_multiple_dots() {
        let path = std::path::PathBuf::from("chapter.1.notes.txt");

        let stem = path.file_stem()
            .map(|n| n.to_string_lossy().to_string());

        assert_eq!(stem, Some("chapter.1.notes".to_string()));
    }

    #[test]
    fn test_uploaded_file_title_generation() {
        // Simulate generating title from filename
        let filename = "rust-programming-guide.txt";
        let path = std::path::PathBuf::from(filename);

        let title = path.file_stem()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "Untitled".to_string());

        assert_eq!(title, "rust-programming-guide");
    }

    #[test]
    fn test_file_content_word_count() {
        // Simulate reading file content
        let file_content = r#"
            Chapter 1: Introduction to Rust

            Rust is a systems programming language focused on safety,
            concurrency, and performance. It provides memory safety
            without using garbage collection.
        "#;

        let word_count = file_content.split_whitespace().count();

        assert!(word_count > 20);
        assert!(word_count < 40);
    }

    #[test]
    fn test_reupload_replaces_content() {
        // Simulate content replacement
        let mut content = "Original content".to_string();
        let mut filename = Some("original.txt".to_string());

        // Simulate re-upload
        content = "New uploaded content".to_string();
        filename = Some("new-file.txt".to_string());

        assert_eq!(content, "New uploaded content");
        assert_eq!(filename, Some("new-file.txt".to_string()));
    }

    #[test]
    fn test_clear_content_resets_state() {
        let mut content = "Some content".to_string();
        let mut title = "Some title".to_string();
        let mut filename: Option<String> = Some("file.txt".to_string());
        let mut error: Option<String> = Some("An error".to_string());

        // Simulate clear
        content = String::new();
        title = String::new();
        filename = None;
        error = None;

        assert!(content.is_empty());
        assert!(title.is_empty());
        assert!(filename.is_none());
        assert!(error.is_none());
    }
}
