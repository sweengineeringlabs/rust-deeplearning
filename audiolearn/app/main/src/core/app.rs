//! Main AudioLearn Application

use dioxus::prelude::*;
use ::components::prelude::*;
use crate::api::*;
use crate::core::{get_sample_courses, components::*, pages::*};

/// Navigation routes
#[derive(Clone, PartialEq)]
pub enum Route {
    Home,
    Library,
    Streak,
    Profile,
    Create,
    Course(String),
    Player,
}

/// Embedded CSS styles
const STYLES: &str = include_str!("../../assets/styles.css");

/// Main app component
#[component]
pub fn AudioLearnApp() -> Element {
    // State
    let mut route = use_signal(|| Route::Home);
    let user = use_signal(User::default);
    let courses = use_signal(get_sample_courses);
    let mut current_course = use_signal(|| Option::<Course>::None);
    let mut current_lesson = use_signal(|| Option::<Lesson>::None);
    let mut is_playing = use_signal(|| false);
    let position = use_signal(|| 0u32);
    let mut show_player = use_signal(|| false);


    rsx! {
        // Inject CSS
        style { "{STYLES}" }

        div { class: "audiolearn-app",
            // Header
            header { class: "app-header",
                div { class: "logo",
                    span { class: "logo-icon", "🎧" }
                    span { class: "logo-text", "AudioLearn" }
                }

                div { class: "header-actions",
                    button {
                        class: "create-btn",
                        onclick: move |_| route.set(Route::Create),
                        title: "Create custom material",
                        Icon { name: IconName::Plus }
                    }
                    button { class: "search-btn",
                        Icon { name: IconName::Search }
                    }
                    Avatar { fallback: "U".to_string(), size: Size::Sm }
                }
            }

            // Main content
            main { class: "app-main",
                match route.read().clone() {
                    Route::Home | Route::Library => rsx! {
                        HomePage {
                            courses: courses.read().clone(),
                            on_course_click: move |id: String| {
                                if let Some(c) = courses.read().iter().find(|c| c.id == id) {
                                    current_course.set(Some(c.clone()));
                                    route.set(Route::Course(id));
                                }
                            },
                        }
                    },
                    Route::Course(_id) => {
                        if let Some(course) = current_course.read().clone() {
                            rsx! {
                                CoursePage {
                                    course: course.clone(),
                                    completed_lessons: vec![],
                                    current_lesson_id: current_lesson.read().as_ref().map(|l| l.id.clone()),
                                    on_lesson_click: {
                                        let course_clone = course.clone();
                                        move |lesson_id: String| {
                                            if let Some(lesson) = course_clone.get_lesson(&lesson_id) {
                                                current_lesson.set(Some(lesson.clone()));
                                                show_player.set(true);
                                                is_playing.set(true);

                                                // Use TTS to read the lesson content
                                                // Extract transcript text or use description
                                                let content = lesson.transcript
                                                    .as_ref()
                                                    .map(|segments| {
                                                        segments.iter()
                                                            .map(|s| s.text.as_str())
                                                            .collect::<Vec<_>>()
                                                            .join(" ")
                                                    })
                                                    .or_else(|| lesson.description.clone())
                                                    .unwrap_or_else(|| "This lesson content will be available soon.".to_string());

                                                let lesson_text = format!(
                                                    "Now playing: {}. {}",
                                                    lesson.title,
                                                    content
                                                );
                                                // Platform-specific TTS handling
                                                #[cfg(feature = "desktop")]
                                                spawn(async move {
                                                    let _ = tokio::task::spawn_blocking(move || {
                                                        // Stop any existing speech first
                                                        let _ = crate::saf::stop_tts();
                                                        crate::saf::speak_text(&lesson_text)
                                                    }).await;
                                                });
                                                #[cfg(feature = "web")]
                                                {
                                                    let _ = crate::saf::stop_tts();
                                                    let _ = crate::saf::speak_text(&lesson_text);
                                                }
                                            }
                                        }
                                    },
                                    on_back: move |_| route.set(Route::Home),
                                }
                            }
                        } else {
                            rsx! { div { "Course not found" } }
                        }
                    },
                    Route::Profile => rsx! {
                        ProfilePage { user: user.read().clone() }
                    },
                    Route::Streak => rsx! {
                        div { class: "streak-page",
                            h1 { "🔥 Streak" }
                            p { "{user.read().streak.current_days} day streak" }
                        }
                    },
                    Route::Create => rsx! {
                        CreatePage {
                            on_back: move |_| route.set(Route::Home),
                            on_play: move |_material| {
                                // Material is already played in CreatePage
                            },
                        }
                    },
                    Route::Player => rsx! {
                        div { "Full player view" }
                    },
                }
            }

            // Mini player (when audio is playing)
            if *show_player.read() {
                if let Some(lesson) = current_lesson.read().clone() {
                    {
                        let lesson_title = lesson.title.clone();
                        rsx! {
                            MiniPlayer {
                                title: lesson.title,
                                subtitle: current_course.read().as_ref().map(|c| c.title.clone()).unwrap_or_default(),
                                icon: current_course.read().as_ref().map(|c| c.icon.clone()).unwrap_or("🎧".to_string()),
                                position: *position.read(),
                                duration: lesson.duration,
                                is_playing: *is_playing.read(),
                                on_play: {
                                    let title = lesson_title.clone();
                                    move |_| {
                                        is_playing.set(true);
                                        let text = format!("Resuming: {}", title);
                                        #[cfg(feature = "desktop")]
                                        spawn(async move {
                                            let _ = tokio::task::spawn_blocking(move || {
                                                let _ = crate::saf::stop_tts();
                                                crate::saf::speak_text(&text)
                                            }).await;
                                        });
                                        #[cfg(feature = "web")]
                                        {
                                            let _ = crate::saf::stop_tts();
                                            let _ = crate::saf::speak_text(&text);
                                        }
                                    }
                                },
                                on_pause: move |_| {
                                    is_playing.set(false);
                                    let _ = crate::saf::stop_tts();
                                },
                                on_expand: move |_| route.set(Route::Player),
                            }
                        }
                    }
                }
            }

            // Bottom navigation
            nav { class: "bottom-nav",
                button { class: if matches!(*route.read(), Route::Home) { "active" } else { "" },
                    onclick: move |_| route.set(Route::Home),
                    Icon { name: IconName::Home }
                    span { "Home" }
                }
                button { class: if matches!(*route.read(), Route::Library) { "active" } else { "" },
                    onclick: move |_| route.set(Route::Library),
                    Icon { name: IconName::Book }
                    span { "Library" }
                }
                button { class: if matches!(*route.read(), Route::Streak) { "active" } else { "" },
                    onclick: move |_| route.set(Route::Streak),
                    Icon { name: IconName::Zap }
                    span { "Streak" }
                }
                button { class: if matches!(*route.read(), Route::Profile) { "active" } else { "" },
                    onclick: move |_| route.set(Route::Profile),
                    Icon { name: IconName::User }
                    span { "Profile" }
                }
            }
        }
    }
}
