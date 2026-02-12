//! Page components

use dioxus::prelude::*;
use ::components::prelude::*;
use crate::api::*;
use crate::api::Timestamp;
use crate::core::components::*;

/// Home page
#[derive(Props, Clone, PartialEq)]
pub struct HomePageProps {
    pub courses: Vec<Course>,
    pub on_course_click: EventHandler<String>,
}

#[component]
pub fn HomePage(props: HomePageProps) -> Element {
    rsx! {
        div { class: "home-page",
            section { class: "section",
                h2 { "Continue Learning" }
                // Continue card would go here
            }

            section { class: "section",
                h2 { "Your Courses" }
                div { class: "course-grid",
                    for course in props.courses.iter() {
                        CourseCard {
                            course: course.clone(),
                            progress: 40.0,
                            on_click: props.on_course_click.clone(),
                        }
                    }
                }
            }

            section { class: "section",
                h2 { "Recommended" }
            }
        }
    }
}

/// Course detail page
#[derive(Props, Clone, PartialEq)]
pub struct CoursePageProps {
    pub course: Course,
    pub completed_lessons: Vec<String>,
    pub current_lesson_id: Option<String>,
    pub on_lesson_click: EventHandler<String>,
    pub on_back: EventHandler<()>,
}

#[component]
pub fn CoursePage(props: CoursePageProps) -> Element {
    let course = &props.course;
    let total_duration = Timestamp::new(course.total_duration);

    rsx! {
        div { class: "course-page",
            header { class: "course-header",
                button { class: "back-btn",
                    onclick: move |_| props.on_back.call(()),
                    Icon { name: IconName::ArrowLeft }
                    "Back"
                }
            }

            div { class: "course-cover",
                div { class: "cover-icon", "{course.icon}" }
            }

            div { class: "course-info",
                h1 { "{course.title}" }
                p { class: "author", "By {course.author.name}" }
                p { class: "meta",
                    "⭐ {course.rating} ({course.review_count} reviews) | "
                    "🎧 {course.lesson_count()} lessons | "
                    "⏱️ {total_duration.format_long()} | "
                    "{course.difficulty.icon()} {course.difficulty.label()}"
                }

                div { class: "course-actions",
                    Button {
                        variant: Variant::Primary,
                        size: Size::Lg,
                        Icon { name: IconName::Play }
                        "Play Course"
                    }

                    // Read course title aloud
                    ReadAloudButton {
                        text: format!("{} by {}. {}", course.title, course.author.name, course.description),
                        label: Some("Read Description".to_string()),
                        size: Size::Lg,
                    }
                }
            }

            div { class: "course-description",
                p { "{course.description}" }
                QuickTtsButton {
                    text: course.description.clone(),
                    tooltip: Some("Read description aloud".to_string()),
                }
            }

            div { class: "chapters",
                for chapter in course.chapters.iter() {
                    div { class: "chapter",
                        div { class: "chapter-header",
                            h3 { "{chapter.title}" }
                            QuickTtsButton {
                                text: chapter.title.clone(),
                                tooltip: Some("Read chapter title".to_string()),
                            }
                        }

                        div { class: "lessons",
                            for lesson in chapter.lessons.iter() {
                                LessonRow {
                                    number: format!("{}.{}", chapter.id.replace("ch", ""), lesson.order),
                                    title: lesson.title.clone(),
                                    duration: Timestamp::new(lesson.duration).format(),
                                    completed: props.completed_lessons.contains(&lesson.id),
                                    playing: props.current_lesson_id.as_ref() == Some(&lesson.id),
                                    on_click: {
                                        let id = lesson.id.clone();
                                        move |_| props.on_lesson_click.call(id.clone())
                                    },
                                    on_read_aloud: move |text: String| {
                                        #[cfg(feature = "desktop")]
                                        spawn(async move {
                                            let t = text.clone();
                                            let _ = tokio::task::spawn_blocking(move || {
                                                // Stop any existing speech first
                                                let _ = crate::saf::stop_tts();
                                                crate::saf::speak_text(&t)
                                            }).await;
                                        });
                                        #[cfg(feature = "web")]
                                        {
                                            let _ = crate::saf::stop_tts();
                                            let _ = crate::saf::speak_text(&text);
                                        }
                                    },
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Profile page
#[derive(Props, Clone, PartialEq)]
pub struct ProfilePageProps {
    pub user: User,
}

#[component]
pub fn ProfilePage(props: ProfilePageProps) -> Element {
    let user = &props.user;
    let listen_hours = user.total_listen_time / 3600;

    // TTS settings state
    let mut selected_voice = use_signal(|| Option::<String>::None);
    let mut speech_rate = use_signal(|| 1.0f32);

    rsx! {
        div { class: "profile-page",
            div { class: "profile-header",
                Avatar {
                    fallback: user.name.chars().next().unwrap_or('U').to_string(),
                    size: Size::Xl,
                }
                h1 { "{user.name}" }
                p { "Learning since 2024" }
            }

            div { class: "stats-grid",
                StatsCard { icon: "🔥", value: format!("{}", user.streak.current_days), label: "Day Streak" }
                StatsCard { icon: "⏱️", value: format!("{}h", listen_hours), label: "Listened" }
                StatsCard { icon: "📚", value: format!("{}", user.enrolled_courses.len()), label: "Courses" }
                StatsCard { icon: "⭐", value: format!("{}", user.points), label: "Points" }
            }

            // TTS Settings Section
            TtsSettingsPanel {
                selected_voice_id: selected_voice.read().clone(),
                on_voice_change: move |voice_id: String| {
                    if voice_id.is_empty() {
                        selected_voice.set(None);
                    } else {
                        selected_voice.set(Some(voice_id));
                    }
                },
                rate: *speech_rate.read(),
                on_rate_change: move |rate: f32| {
                    speech_rate.set(rate);
                },
            }

            section { class: "section",
                h2 { "Achievements" }
                div { class: "achievements-grid",
                    for achievement in user.achievements.iter() {
                        div { class: "achievement",
                            class: if achievement.is_unlocked() { "unlocked" } else { "locked" },
                            span { class: "achievement-icon", "{achievement.icon}" }
                            span { class: "achievement-name", "{achievement.name}" }
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// Create Page - Upload/Paste Learning Material
// =============================================================================

/// Custom learning material
#[derive(Clone, Debug, PartialEq)]
pub struct CustomMaterial {
    pub id: String,
    pub title: String,
    pub content: String,
    pub created_at: String,
}

/// Create page props
#[derive(Props, Clone, PartialEq)]
pub struct CreatePageProps {
    pub on_back: EventHandler<()>,
    pub on_play: EventHandler<CustomMaterial>,
}

#[component]
pub fn CreatePage(props: CreatePageProps) -> Element {
    let mut title = use_signal(|| String::new());
    let mut content = use_signal(|| String::new());
    let mut is_playing = use_signal(|| false);
    let mut saved_materials = use_signal(|| Vec::<CustomMaterial>::new());
    let mut show_saved = use_signal(|| false);
    let mut uploaded_filename = use_signal(|| Option::<String>::None);
    let mut upload_error = use_signal(|| Option::<String>::None);

    // Calculate word count and estimated duration
    let word_count = content.read().split_whitespace().count();
    let estimated_minutes = (word_count as f32 / 150.0).ceil() as u32; // ~150 words/min

    // File upload handler - platform specific
    #[cfg(feature = "desktop")]
    let upload_file = {
        let content = content.clone();
        let uploaded_filename = uploaded_filename.clone();
        let upload_error = upload_error.clone();
        let title = title.clone();

        move |_| {
            // Clone signals for the closure
            let mut content = content.clone();
            let mut uploaded_filename = uploaded_filename.clone();
            let mut upload_error = upload_error.clone();
            let mut title = title.clone();

            // Open file dialog synchronously on main thread using pollster
            // (rfd needs to run on a thread with a message loop on Windows)
            let result = rfd::FileDialog::new()
                .add_filter("Text files", &["txt", "md", "text"])
                .add_filter("All files", &["*"])
                .set_title("Select Learning Material")
                .pick_file();

            if let Some(path) = result {
                let filename = path.file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "Unknown file".to_string());

                let file_stem = path.file_stem()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| filename.clone());

                match std::fs::read_to_string(&path) {
                    Ok(file_content) => {
                        content.set(file_content);
                        uploaded_filename.set(Some(filename));
                        upload_error.set(None);

                        if title.read().is_empty() {
                            title.set(file_stem);
                        }
                    }
                    Err(e) => {
                        upload_error.set(Some(format!("Failed to read file: {}", e)));
                    }
                }
            }
        }
    };

    // Web version - file upload not supported yet
    #[cfg(feature = "web")]
    let upload_file = {
        let upload_error = upload_error.clone();
        move |_| {
            let mut upload_error = upload_error.clone();
            upload_error.set(Some("File upload is only available in the desktop app.".to_string()));
        }
    };

    // Clear uploaded file and content
    let clear_content = move |_| {
        content.set(String::new());
        title.set(String::new());
        uploaded_filename.set(None);
        upload_error.set(None);
    };

    let play_content = move |_| {
        let text = content.read().clone();
        let material_title = title.read().clone();

        if text.is_empty() {
            return;
        }

        is_playing.set(true);

        // Format the text with title
        let full_text = if material_title.is_empty() {
            text.clone()
        } else {
            format!("{}. {}", material_title, text)
        };

        // Limit text length for TTS
        let text_to_speak = if full_text.len() > 5000 {
            full_text.chars().take(5000).collect::<String>()
        } else {
            full_text
        };

        // Platform-specific TTS
        #[cfg(feature = "desktop")]
        std::thread::spawn(move || {
            let _ = crate::saf::stop_tts();
            let _ = crate::saf::speak_text(&text_to_speak);
        });
        #[cfg(feature = "web")]
        {
            let _ = crate::saf::stop_tts();
            let _ = crate::saf::speak_text(&text_to_speak);
        }
    };

    let stop_playback = move |_| {
        let _ = crate::saf::stop_tts();
        is_playing.set(false);
    };

    let save_material = move |_| {
        let material_title = title.read().clone();
        let material_content = content.read().clone();

        if material_content.is_empty() {
            return;
        }

        let material = CustomMaterial {
            id: format!("custom_{}", chrono::Utc::now().timestamp()),
            title: if material_title.is_empty() {
                "Untitled".to_string()
            } else {
                material_title
            },
            content: material_content,
            created_at: chrono::Utc::now().format("%Y-%m-%d %H:%M").to_string(),
        };

        saved_materials.write().push(material);

        // Clear form
        title.set(String::new());
        content.set(String::new());
    };

    let mut play_saved_material = move |material: CustomMaterial| {
        is_playing.set(true);

        let full_text = format!("{}. {}", material.title, material.content);

        #[cfg(feature = "desktop")]
        spawn(async move {
            let _ = tokio::task::spawn_blocking(move || {
                let _ = crate::saf::stop_tts();
                crate::saf::speak_text(&full_text)
            }).await;
            is_playing.set(false);
        });
        #[cfg(feature = "web")]
        {
            let _ = crate::saf::stop_tts();
            let _ = crate::saf::speak_text(&full_text);
            is_playing.set(false);
        }
    };

    rsx! {
        div { class: "create-page",
            // Header
            header { class: "create-header",
                button {
                    class: "back-btn",
                    onclick: move |_| props.on_back.call(()),
                    Icon { name: IconName::ArrowLeft }
                    "Back"
                }
                h1 { "Create Learning Material" }
            }

            // Tabs
            div { class: "create-tabs",
                button {
                    class: if !*show_saved.read() { "tab active" } else { "tab" },
                    onclick: move |_| show_saved.set(false),
                    Icon { name: IconName::Edit }
                    "New"
                }
                button {
                    class: if *show_saved.read() { "tab active" } else { "tab" },
                    onclick: move |_| show_saved.set(true),
                    Icon { name: IconName::Bookmark }
                    "Saved ({saved_materials.read().len()})"
                }
            }

            if !*show_saved.read() {
                // New material form
                div { class: "create-form",
                    // Upload section
                    div { class: "upload-section",
                        Button {
                            variant: Variant::Ghost,
                            size: Size::Md,
                            onclick: upload_file,
                            Icon { name: IconName::Upload }
                            "Upload File"
                        }

                        if let Some(filename) = uploaded_filename.read().as_ref() {
                            div { class: "uploaded-file",
                                Icon { name: IconName::File, size: Size::Sm }
                                span { "{filename}" }
                                button {
                                    class: "clear-btn",
                                    onclick: clear_content,
                                    title: "Clear and re-upload",
                                    Icon { name: IconName::X, size: Size::Sm }
                                }
                            }
                        }

                        if let Some(error) = upload_error.read().as_ref() {
                            div { class: "upload-error",
                                Icon { name: IconName::Warning, size: Size::Sm }
                                "{error}"
                            }
                        }
                    }

                    // Title input
                    div { class: "form-group",
                        label { "Title (optional)" }
                        input {
                            r#type: "text",
                            class: "title-input",
                            placeholder: "Enter a title for your material...",
                            value: "{title}",
                            oninput: move |e| title.set(e.value()),
                        }
                    }

                    // Content textarea
                    div { class: "form-group",
                        label { "Learning Content" }
                        textarea {
                            class: "content-textarea",
                            placeholder: "Paste or type your learning material here...\n\nOr click 'Upload File' to load a text file.\n\nSupported formats: .txt, .md, .text",
                            value: "{content}",
                            oninput: move |e| content.set(e.value()),
                            rows: 12,
                        }
                    }

                    // Stats bar
                    div { class: "content-stats",
                        span { class: "stat",
                            Icon { name: IconName::FileText, size: Size::Sm }
                            "{word_count} words"
                        }
                        span { class: "stat",
                            Icon { name: IconName::Clock, size: Size::Sm }
                            "~{estimated_minutes} min"
                        }
                    }

                    // Action buttons
                    div { class: "create-actions",
                        if *is_playing.read() {
                            Button {
                                variant: Variant::Danger,
                                size: Size::Lg,
                                onclick: stop_playback,
                                Icon { name: IconName::X }
                                "Stop"
                            }
                        } else {
                            Button {
                                variant: Variant::Primary,
                                size: Size::Lg,
                                disabled: content.read().is_empty(),
                                onclick: play_content,
                                Icon { name: IconName::Play }
                                "Play Now"
                            }
                        }

                        Button {
                            variant: Variant::Ghost,
                            size: Size::Lg,
                            disabled: content.read().is_empty(),
                            onclick: save_material,
                            Icon { name: IconName::Bookmark }
                            "Save"
                        }
                    }
                }
            } else {
                // Saved materials list
                div { class: "saved-materials",
                    if saved_materials.read().is_empty() {
                        div { class: "empty-state",
                            Icon { name: IconName::Inbox, size: Size::Xl }
                            h3 { "No saved materials" }
                            p { "Create and save learning materials to access them here." }
                        }
                    } else {
                        for material in saved_materials.read().iter() {
                            {
                                let mat = material.clone();
                                let mat_for_play = material.clone();
                                rsx! {
                                    div { class: "saved-material-card",
                                        div { class: "material-info",
                                            h3 { "{mat.title}" }
                                            p { class: "preview",
                                                "{mat.content.chars().take(100).collect::<String>()}..."
                                            }
                                            span { class: "meta",
                                                Icon { name: IconName::Clock, size: Size::Sm }
                                                "{mat.created_at}"
                                            }
                                        }
                                        div { class: "material-actions",
                                            button {
                                                class: "play-btn",
                                                onclick: move |_| play_saved_material(mat_for_play.clone()),
                                                Icon { name: IconName::Play }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
