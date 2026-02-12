//! Custom components for the tutorial app

use dioxus::prelude::*;
use components::*;
use crate::core::data::{ContentBlock, QuizQuestion};

/// XP gained animation
#[derive(Props, Clone, PartialEq)]
pub struct XpGainedProps {
    pub amount: u32,
    pub visible: bool,
}

#[component]
pub fn XpGained(props: XpGainedProps) -> Element {
    if !props.visible {
        return rsx! {};
    }

    rsx! {
        div { class: "xp-gained animate-float",
            Icon { name: IconName::Star, color: "gold" }
            span { "+{props.amount} XP" }
        }
    }
}

/// Hearts display
#[derive(Props, Clone, PartialEq)]
pub struct HeartsProps {
    pub count: u32,
    pub max: u32,
}

#[component]
pub fn Hearts(props: HeartsProps) -> Element {
    rsx! {
        div { class: "hearts-container",
            for i in 0..props.max {
                if i < props.count {
                    Icon { name: IconName::HeartFilled, color: "#ff4b4b" }
                } else {
                    Icon { name: IconName::Heart, color: "#ccc" }
                }
            }
        }
    }
}

/// Streak flame display
#[derive(Props, Clone, PartialEq)]
pub struct StreakProps {
    pub days: u32,
}

#[component]
pub fn Streak(props: StreakProps) -> Element {
    let color = if props.days >= 7 { "#ff9500" } else { "#ffcc00" };

    rsx! {
        div { class: "streak-container",
            Icon { name: IconName::Zap, color: "{color}" }
            span { class: "streak-count", "{props.days}" }
        }
    }
}

/// Gems display
#[derive(Props, Clone, PartialEq)]
pub struct GemsProps {
    pub count: u32,
}

#[component]
pub fn Gems(props: GemsProps) -> Element {
    rsx! {
        div { class: "gems-container",
            span { class: "gem-icon", "💎" }
            span { class: "gem-count", "{props.count}" }
        }
    }
}

/// Lesson card on the path
#[derive(Props, Clone, PartialEq)]
pub struct LessonNodeProps {
    pub title: String,
    pub icon: String,
    pub completed: bool,
    pub locked: bool,
    pub current: bool,
    pub on_click: EventHandler<()>,
}

#[component]
pub fn LessonNode(props: LessonNodeProps) -> Element {
    let class = format!(
        "lesson-node {} {} {}",
        if props.completed { "completed" } else { "" },
        if props.locked { "locked" } else { "" },
        if props.current { "current pulse" } else { "" },
    );

    rsx! {
        button {
            class: "{class}",
            disabled: props.locked,
            onclick: move |_| props.on_click.call(()),

            div { class: "node-icon", "{props.icon}" }

            if props.completed {
                div { class: "check-overlay",
                    Icon { name: IconName::Check, color: "white" }
                }
            }

            if props.locked {
                div { class: "lock-overlay",
                    Icon { name: IconName::Lock, color: "#888" }
                }
            }
        }

        span { class: "node-title", "{props.title}" }
    }
}

/// Content renderer
#[derive(Props, Clone, PartialEq)]
pub struct ContentRendererProps {
    pub blocks: Vec<ContentBlock>,
}

#[component]
pub fn ContentRenderer(props: ContentRendererProps) -> Element {
    rsx! {
        div { class: "lesson-content",
            for block in props.blocks.iter() {
                match block {
                    ContentBlock::Text(text) => rsx! {
                        p { class: "content-text", "{text}" }
                    },
                    ContentBlock::Code { language, code } => rsx! {
                        div { class: "code-block",
                            div { class: "code-header",
                                span { class: "language-tag", "{language}" }
                                button { class: "copy-btn",
                                    Icon { name: IconName::Copy, size: Size::Sm }
                                }
                            }
                            pre { code { class: "language-{language}", "{code}" } }
                        }
                    },
                    ContentBlock::Tip(text) => rsx! {
                        Alert { variant: Variant::Primary,
                            div { class: "tip",
                                Icon { name: IconName::Info }
                                span { "{text}" }
                            }
                        }
                    },
                    ContentBlock::Warning(text) => rsx! {
                        Alert { variant: Variant::Warning,
                            div { class: "warning",
                                Icon { name: IconName::Warning }
                                span { "{text}" }
                            }
                        }
                    },
                    ContentBlock::Image { src, alt } => rsx! {
                        img { class: "content-image", src: "{src}", alt: "{alt}" }
                    },
                }
            }
        }
    }
}

/// Quiz component
#[derive(Props, Clone, PartialEq)]
pub struct QuizProps {
    pub question: QuizQuestion,
    pub on_answer: EventHandler<bool>,
}

#[component]
pub fn Quiz(props: QuizProps) -> Element {
    let mut selected = use_signal(|| Option::<usize>::None);
    let mut answered = use_signal(|| false);
    let mut is_correct = use_signal(|| false);

    match &props.question {
        QuizQuestion::MultipleChoice { question, options, correct_index, explanation, .. } => {
            let correct = *correct_index;

            rsx! {
                div { class: "quiz-container",
                    h3 { class: "quiz-question", "{question}" }

                    div { class: "quiz-options",
                        for (i, option) in options.iter().enumerate() {
                            button {
                                class: "quiz-option",
                                class: if *answered.read() && i == correct { "correct" } else { "" },
                                class: if *answered.read() && *selected.read() == Some(i) && i != correct { "incorrect" } else { "" },
                                class: if !*answered.read() && *selected.read() == Some(i) { "selected" } else { "" },
                                disabled: *answered.read(),
                                onclick: move |_| {
                                    selected.set(Some(i));
                                },
                                "{option}"
                            }
                        }
                    }

                    if !*answered.read() && selected.read().is_some() {
                        Button {
                            variant: Variant::Primary,
                            full_width: true,
                            onclick: move |_| {
                                let is_right = *selected.read() == Some(correct);
                                is_correct.set(is_right);
                                answered.set(true);
                                props.on_answer.call(is_right);
                            },
                            "Check Answer"
                        }
                    }

                    if *answered.read() {
                        div { class: "quiz-feedback",
                            class: if *is_correct.read() { "correct" } else { "incorrect" },

                            if *is_correct.read() {
                                div { class: "feedback-header",
                                    Icon { name: IconName::Check, color: "#4caf50" }
                                    span { "Correct!" }
                                }
                            } else {
                                div { class: "feedback-header",
                                    Icon { name: IconName::X, color: "#f44336" }
                                    span { "Not quite!" }
                                }
                            }

                            p { class: "explanation", "{explanation}" }
                        }
                    }
                }
            }
        }
        QuizQuestion::TrueFalse { statement, is_true, explanation, .. } => {
            let correct = *is_true;

            rsx! {
                div { class: "quiz-container",
                    h3 { class: "quiz-question", "True or False?" }
                    p { class: "quiz-statement", "{statement}" }

                    div { class: "quiz-options true-false",
                        button {
                            class: "quiz-option",
                            class: if *answered.read() && correct { "correct" } else { "" },
                            class: if *answered.read() && *selected.read() == Some(0) && !correct { "incorrect" } else { "" },
                            class: if !*answered.read() && *selected.read() == Some(0) { "selected" } else { "" },
                            disabled: *answered.read(),
                            onclick: move |_| selected.set(Some(0)),
                            "True"
                        }
                        button {
                            class: "quiz-option",
                            class: if *answered.read() && !correct { "correct" } else { "" },
                            class: if *answered.read() && *selected.read() == Some(1) && correct { "incorrect" } else { "" },
                            class: if !*answered.read() && *selected.read() == Some(1) { "selected" } else { "" },
                            disabled: *answered.read(),
                            onclick: move |_| selected.set(Some(1)),
                            "False"
                        }
                    }

                    if !*answered.read() && selected.read().is_some() {
                        Button {
                            variant: Variant::Primary,
                            full_width: true,
                            onclick: move |_| {
                                let user_answer = *selected.read() == Some(0);
                                let is_right = user_answer == correct;
                                is_correct.set(is_right);
                                answered.set(true);
                                props.on_answer.call(is_right);
                            },
                            "Check Answer"
                        }
                    }

                    if *answered.read() {
                        div { class: "quiz-feedback",
                            class: if *is_correct.read() { "correct" } else { "incorrect" },

                            if *is_correct.read() {
                                div { class: "feedback-header correct",
                                    Icon { name: IconName::Check }
                                    span { "Correct!" }
                                }
                            } else {
                                div { class: "feedback-header incorrect",
                                    Icon { name: IconName::X }
                                    span { "Not quite!" }
                                }
                            }

                            p { class: "explanation", "{explanation}" }
                        }
                    }
                }
            }
        }
        _ => rsx! { div { "Quiz type not implemented yet" } }
    }
}

/// Level up celebration modal
#[derive(Props, Clone, PartialEq)]
pub struct LevelUpModalProps {
    pub level: u32,
    pub gems_earned: u32,
    pub on_close: EventHandler<()>,
}

#[component]
pub fn LevelUpModal(props: LevelUpModalProps) -> Element {
    rsx! {
        Modal {
            open: true,
            on_close: move |_| props.on_close.call(()),

            div { class: "level-up-content",
                div { class: "confetti" }

                h1 { class: "level-up-title", "🎉 Level Up! 🎉" }

                div { class: "new-level",
                    span { class: "level-badge", "{props.level}" }
                }

                p { "You've reached Level {props.level}!" }

                div { class: "rewards",
                    div { class: "reward",
                        span { "💎" }
                        span { "+{props.gems_earned} Gems" }
                    }
                }

                Button {
                    variant: Variant::Primary,
                    size: Size::Lg,
                    onclick: move |_| props.on_close.call(()),
                    "Continue"
                }
            }
        }
    }
}

/// Achievement unlocked toast
#[derive(Props, Clone, PartialEq)]
pub struct AchievementToastProps {
    pub name: String,
    pub icon: String,
    pub description: String,
}

#[component]
pub fn AchievementToast(props: AchievementToastProps) -> Element {
    rsx! {
        div { class: "achievement-toast",
            div { class: "achievement-icon", "{props.icon}" }
            div { class: "achievement-info",
                span { class: "achievement-title", "Achievement Unlocked!" }
                span { class: "achievement-name", "{props.name}" }
            }
        }
    }
}
