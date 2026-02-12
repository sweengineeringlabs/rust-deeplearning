//! Main Tutorial Application

use dioxus::prelude::*;
use components::*;
use crate::core::components::*;
use crate::core::data::{get_rust_course, Module, Lesson};
use crate::core::state::{User, LevelUpEvent, Achievement};

/// Main app component
#[component]
pub fn TutorialApp() -> Element {
    // State
    let mut user = use_signal(|| User::default());
    let mut current_module = use_signal(|| 0usize);
    let mut current_lesson = use_signal(|| Option::<usize>::None);
    let mut current_quiz = use_signal(|| 0usize);
    let mut show_content = use_signal(|| true);
    let mut level_up_event = use_signal(|| Option::<LevelUpEvent>::None);
    let mut xp_gained = use_signal(|| Option::<u32>::None);
    let mut mistakes = use_signal(|| 0u32);

    let course = get_rust_course();

    // Handle quiz answer
    let handle_answer = move |is_correct: bool| {
        if is_correct {
            // Correct answer - award XP
            let xp = 10;
            xp_gained.set(Some(xp));

            if let Some(event) = user.write().add_xp(xp) {
                level_up_event.set(Some(event));
            }
        } else {
            // Wrong answer - lose heart
            mistakes.set(*mistakes.read() + 1);
            user.write().lose_heart();
        }
    };

    // Complete lesson
    let complete_lesson = move |lesson_id: String, xp_reward: u32| {
        user.write().complete_lesson(&lesson_id);

        // Award XP
        let bonus = if *mistakes.read() == 0 { xp_reward / 2 } else { 0 }; // Perfect bonus
        let total_xp = xp_reward + bonus;

        if let Some(event) = user.write().add_xp(total_xp) {
            level_up_event.set(Some(event));
        }

        // Check achievements
        if user.read().completed_lessons.len() == 1 {
            user.write().unlock_achievement(Achievement::first_lesson());
        }
        if *mistakes.read() == 0 {
            user.write().unlock_achievement(Achievement::perfect_lesson());
        }

        // Reset and go back to path
        mistakes.set(0);
        current_lesson.set(None);
        current_quiz.set(0);
        show_content.set(true);
    };

    rsx! {
        div { class: "tutorial-app",
            // Header
            header { class: "app-header",
                div { class: "logo",
                    span { class: "logo-icon", "🦀" }
                    span { class: "logo-text", "RustLingo" }
                }

                div { class: "user-stats",
                    Hearts { count: user.read().hearts, max: 5 }
                    Streak { days: user.read().streak }
                    Gems { count: user.read().gems }
                }

                div { class: "user-profile",
                    div { class: "level-badge",
                        "Lv.{user.read().level}"
                    }
                    Progress {
                        value: user.read().xp as f32,
                        max: user.read().xp_to_next_level as f32,
                        variant: Variant::Success,
                        size: Size::Sm,
                    }
                }
            }

            // Main content
            main { class: "app-main",
                if current_lesson.read().is_none() {
                    // Learning path view
                    div { class: "learning-path",
                        h2 { "Your Learning Path" }

                        div { class: "modules-list",
                            for (mod_idx, module) in course.iter().enumerate() {
                                div { class: "module-section",
                                    div { class: "module-header",
                                        span { class: "module-icon", "{module.icon}" }
                                        h3 { "{module.name}" }
                                        if module.required_level > user.read().level {
                                            Badge { variant: Variant::Secondary,
                                                Icon { name: IconName::Lock, size: Size::Sm }
                                                " Level {module.required_level}"
                                            }
                                        }
                                    }

                                    p { class: "module-desc", "{module.description}" }

                                    div { class: "lessons-path",
                                        for (lesson_idx, lesson) in module.lessons.iter().enumerate() {
                                            LessonNode {
                                                title: lesson.title.clone(),
                                                icon: (if user.read().is_lesson_completed(&lesson.id) { "✓" } else { &module.icon }).to_string(),
                                                completed: user.read().is_lesson_completed(&lesson.id),
                                                locked: module.required_level > user.read().level,
                                                current: mod_idx == *current_module.read() && !user.read().is_lesson_completed(&lesson.id),
                                                on_click: move |_| {
                                                    if module.required_level <= user.read().level {
                                                        current_module.set(mod_idx);
                                                        current_lesson.set(Some(lesson_idx));
                                                    }
                                                },
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    // Lesson view
                    {
                        let mod_idx = *current_module.read();
                        let lesson_idx = current_lesson.read().unwrap_or(0);
                        let module = &course[mod_idx];
                        let lesson = &module.lessons[lesson_idx];

                        rsx! {
                            div { class: "lesson-view",
                                // Lesson header
                                div { class: "lesson-header",
                                    button {
                                        class: "back-btn",
                                        onclick: move |_| {
                                            current_lesson.set(None);
                                            current_quiz.set(0);
                                            show_content.set(true);
                                            mistakes.set(0);
                                        },
                                        Icon { name: IconName::ArrowLeft }
                                    }

                                    div { class: "lesson-progress",
                                        Progress {
                                            value: if *show_content.read() { 0.0 } else { (*current_quiz.read() + 1) as f32 },
                                            max: (lesson.quiz.len() + 1) as f32,
                                            variant: Variant::Primary,
                                        }
                                    }

                                    Hearts { count: user.read().hearts, max: 5 }
                                }

                                h2 { class: "lesson-title", "{lesson.title}" }

                                if *show_content.read() {
                                    // Show lesson content
                                    div { class: "lesson-content-section",
                                        ContentRenderer { blocks: lesson.content.clone() }

                                        Button {
                                            variant: Variant::Primary,
                                            size: Size::Lg,
                                            full_width: true,
                                            onclick: move |_| {
                                                show_content.set(false);
                                            },
                                            "Start Quiz"
                                            Icon { name: IconName::ChevronRight }
                                        }
                                    }
                                } else {
                                    // Show quiz
                                    div { class: "quiz-section",
                                        {
                                            let quiz_idx = *current_quiz.read();

                                            if quiz_idx < lesson.quiz.len() {
                                                let question = lesson.quiz[quiz_idx].clone();

                                                rsx! {
                                                    Quiz {
                                                        question: question,
                                                        on_answer: move |is_correct| {
                                                            handle_answer(is_correct);

                                                            // Move to next question after delay
                                                            // In real app, use timeout
                                                        },
                                                    }

                                                    Button {
                                                        variant: Variant::Secondary,
                                                        onclick: move |_| {
                                                            let next = quiz_idx + 1;
                                                            if next < lesson.quiz.len() {
                                                                current_quiz.set(next);
                                                            } else {
                                                                // Lesson complete
                                                                complete_lesson(lesson.id.clone(), lesson.xp_reward);
                                                            }
                                                        },
                                                        if quiz_idx + 1 < lesson.quiz.len() {
                                                            "Next Question"
                                                        } else {
                                                            "Complete Lesson"
                                                        }
                                                    }
                                                }
                                            } else {
                                                rsx! { div { "Quiz complete!" } }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // XP gained animation
            if let Some(xp) = *xp_gained.read() {
                XpGained { amount: xp, visible: true }
            }

            // Level up modal
            if let Some(event) = level_up_event.read().clone() {
                LevelUpModal {
                    level: event.new_level,
                    gems_earned: event.gems_earned,
                    on_close: move |_| {
                        level_up_event.set(None);
                    },
                }
            }
        }
    }
}
