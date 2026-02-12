//! Custom UI components for AudioLearn

use dioxus::prelude::*;
use ::components::prelude::*;
use crate::api::{Seconds, Timestamp};
use crate::api::Course;

/// Audio player controls
#[derive(Props, Clone, PartialEq)]
pub struct PlayerControlsProps {
    pub is_playing: bool,
    pub on_play: EventHandler<()>,
    pub on_pause: EventHandler<()>,
    pub on_skip_back: EventHandler<()>,
    pub on_skip_forward: EventHandler<()>,
}

#[component]
pub fn PlayerControls(props: PlayerControlsProps) -> Element {
    rsx! {
        div { class: "player-controls",
            button { class: "skip-btn",
                onclick: move |_| props.on_skip_back.call(()),
                "-15s"
            }
            button { class: "prev-btn",
                Icon { name: IconName::SkipBack }
            }
            button { class: "play-btn",
                onclick: move |_| {
                    if props.is_playing {
                        props.on_pause.call(());
                    } else {
                        props.on_play.call(());
                    }
                },
                if props.is_playing {
                    Icon { name: IconName::Pause, size: Size::Lg }
                } else {
                    Icon { name: IconName::Play, size: Size::Lg }
                }
            }
            button { class: "next-btn",
                Icon { name: IconName::SkipForward }
            }
            button { class: "skip-btn",
                onclick: move |_| props.on_skip_forward.call(()),
                "+15s"
            }
        }
    }
}

/// Progress bar for audio
#[derive(Props, Clone, PartialEq)]
pub struct AudioProgressProps {
    pub position: Seconds,
    pub duration: Seconds,
    pub on_seek: EventHandler<Seconds>,
}

#[component]
pub fn AudioProgress(props: AudioProgressProps) -> Element {
    let progress = if props.duration > 0 {
        (props.position as f32 / props.duration as f32) * 100.0
    } else {
        0.0
    };

    let pos = Timestamp::new(props.position);
    let dur = Timestamp::new(props.duration);

    rsx! {
        div { class: "audio-progress",
            span { class: "time current", "{pos.format()}" }
            div { class: "progress-bar",
                div { class: "progress-fill", style: "width: {progress}%" }
            }
            span { class: "time total", "{dur.format()}" }
        }
    }
}

/// Course card for library
#[derive(Props, Clone, PartialEq)]
pub struct CourseCardProps {
    pub course: Course,
    pub progress: f32,
    pub on_click: EventHandler<String>,
}

#[component]
pub fn CourseCard(props: CourseCardProps) -> Element {
    let course_id = props.course.id.clone();

    rsx! {
        div { class: "course-card",
            onclick: move |_| props.on_click.call(course_id.clone()),

            div { class: "course-icon", "{props.course.icon}" }
            div { class: "course-info",
                h4 { "{props.course.title}" }
                Progress {
                    value: props.progress,
                    max: 100.0,
                    size: Size::Sm,
                }
            }
        }
    }
}

/// Mini player bar
#[derive(Props, Clone, PartialEq)]
pub struct MiniPlayerProps {
    pub title: String,
    pub subtitle: String,
    pub icon: String,
    pub position: Seconds,
    pub duration: Seconds,
    pub is_playing: bool,
    pub on_play: EventHandler<()>,
    pub on_pause: EventHandler<()>,
    pub on_expand: EventHandler<()>,
}

#[component]
pub fn MiniPlayer(props: MiniPlayerProps) -> Element {
    let progress = if props.duration > 0 {
        (props.position as f32 / props.duration as f32) * 100.0
    } else {
        0.0
    };

    rsx! {
        div { class: "mini-player",
            div { class: "mini-progress", style: "width: {progress}%" }

            div { class: "mini-content",
                onclick: move |_| props.on_expand.call(()),

                div { class: "mini-icon", "{props.icon}" }
                div { class: "mini-info",
                    span { class: "mini-title", "{props.title}" }
                    span { class: "mini-subtitle", "{props.subtitle}" }
                }

                button { class: "mini-prev",
                    Icon { name: IconName::SkipBack }
                }
                button { class: "mini-play",
                    onclick: move |e| {
                        e.stop_propagation();
                        if props.is_playing {
                            props.on_pause.call(());
                        } else {
                            props.on_play.call(());
                        }
                    },
                    if props.is_playing {
                        Icon { name: IconName::Pause }
                    } else {
                        Icon { name: IconName::Play }
                    }
                }
                button { class: "mini-next",
                    Icon { name: IconName::SkipForward }
                }

                button { class: "mini-expand",
                    Icon { name: IconName::ChevronUp }
                }
            }
        }
    }
}

/// Stats card
#[derive(Props, Clone, PartialEq)]
pub struct StatsCardProps {
    pub icon: String,
    pub value: String,
    pub label: String,
}

#[component]
pub fn StatsCard(props: StatsCardProps) -> Element {
    rsx! {
        div { class: "stats-card",
            span { class: "stats-icon", "{props.icon}" }
            span { class: "stats-value", "{props.value}" }
            span { class: "stats-label", "{props.label}" }
        }
    }
}

/// Lesson row in chapter list
#[derive(Props, Clone, PartialEq)]
pub struct LessonRowProps {
    pub number: String,
    pub title: String,
    pub duration: String,
    pub completed: bool,
    pub playing: bool,
    pub on_click: EventHandler<()>,
    /// Optional handler for reading lesson title aloud
    #[props(default)]
    pub on_read_aloud: Option<EventHandler<String>>,
}

#[component]
pub fn LessonRow(props: LessonRowProps) -> Element {
    let class = format!(
        "lesson-row {} {}",
        if props.completed { "completed" } else { "" },
        if props.playing { "playing" } else { "" },
    );

    let title_for_tts = props.title.clone();

    rsx! {
        div { class: "{class}",
            onclick: move |_| props.on_click.call(()),

            span { class: "lesson-number", "{props.number}" }
            span { class: "lesson-title", "{props.title}" }

            if props.completed {
                Icon { name: IconName::Check, color: "var(--success)" }
            } else if props.playing {
                Icon { name: IconName::Play, color: "var(--primary)" }
            }

            span { class: "lesson-duration", "{props.duration}" }

            // Read Aloud button
            if let Some(on_read) = props.on_read_aloud.clone() {
                button {
                    class: "read-aloud-btn",
                    title: "Read title aloud",
                    onclick: move |e| {
                        e.stop_propagation();
                        on_read.call(title_for_tts.clone());
                    },
                    Icon { name: IconName::Headphones, size: Size::Sm }
                }
            }

            button { class: "download-btn",
                Icon { name: IconName::Download, size: Size::Sm }
            }
        }
    }
}

// =============================================================================
// TTS Components
// =============================================================================

/// TTS Status for tracking reading state
#[derive(Clone, Copy, PartialEq, Default)]
pub enum TtsStatus {
    #[default]
    Idle,
    Speaking,
    Loading,
}

/// Read Aloud Button - Simple button to speak text
#[derive(Props, Clone, PartialEq)]
pub struct ReadAloudButtonProps {
    /// Text content to read aloud
    pub text: String,
    /// Optional button label (defaults to icon only)
    #[props(default)]
    pub label: Option<String>,
    /// Button size
    #[props(default = Size::Md)]
    pub size: Size,
    /// Full width button
    #[props(default = false)]
    pub full_width: bool,
}

#[component]
pub fn ReadAloudButton(props: ReadAloudButtonProps) -> Element {
    let mut status = use_signal(|| TtsStatus::Idle);
    let text = props.text.clone();

    let onclick = move |_| {
        let text = text.clone();
        status.set(TtsStatus::Loading);

        // Platform-specific TTS task
        #[cfg(feature = "desktop")]
        spawn(async move {
            status.set(TtsStatus::Speaking);

            // Use blocking spawn for TTS since it's synchronous
            let result = tokio::task::spawn_blocking(move || {
                // Stop any existing speech first
                let _ = crate::saf::stop_tts();
                crate::saf::speak_text(&text)
            }).await;

            match result {
                Ok(Ok(_)) => {
                    status.set(TtsStatus::Idle);
                }
                Ok(Err(e)) => {
                    eprintln!("TTS Error: {}", e);
                    status.set(TtsStatus::Idle);
                }
                Err(e) => {
                    eprintln!("Task Error: {}", e);
                    status.set(TtsStatus::Idle);
                }
            }
        });
        #[cfg(feature = "web")]
        {
            status.set(TtsStatus::Speaking);
            let _ = crate::saf::stop_tts();
            let result = crate::saf::speak_text(&text);
            if let Err(e) = result {
                eprintln!("TTS Error: {}", e);
            }
            status.set(TtsStatus::Idle);
        }
    };

    let icon = match *status.read() {
        TtsStatus::Idle => IconName::Headphones,
        TtsStatus::Speaking => IconName::Speaker,
        TtsStatus::Loading => IconName::Refresh,
    };

    let is_active = *status.read() != TtsStatus::Idle;
    let class = format!(
        "read-aloud-button {}",
        if is_active { "active" } else { "" }
    );

    rsx! {
        Button {
            class: "{class}",
            variant: if is_active { Variant::Primary } else { Variant::Ghost },
            size: props.size.clone(),
            full_width: props.full_width,
            disabled: is_active,
            onclick: onclick,

            Icon { name: icon }
            if let Some(label) = &props.label {
                span { "{label}" }
            }
        }
    }
}

/// TTS Control Bar - Shows during TTS playback with stop button
#[derive(Props, Clone, PartialEq)]
pub struct TtsControlBarProps {
    /// Text being read
    pub text: String,
    /// Whether currently speaking
    pub is_speaking: bool,
    /// Handler for stop button
    pub on_stop: EventHandler<()>,
}

#[component]
pub fn TtsControlBar(props: TtsControlBarProps) -> Element {
    if !props.is_speaking {
        return rsx! {};
    }

    // Truncate long text for display
    let display_text = if props.text.len() > 100 {
        format!("{}...", &props.text[..100])
    } else {
        props.text.clone()
    };

    rsx! {
        div { class: "tts-control-bar",
            div { class: "tts-indicator",
                Icon { name: IconName::Speaker, size: Size::Sm }
                span { class: "tts-pulse" }
            }

            div { class: "tts-text",
                span { "{display_text}" }
            }

            button {
                class: "tts-stop-btn",
                onclick: move |_| props.on_stop.call(()),
                Icon { name: IconName::X }
                "Stop"
            }
        }
    }
}

/// TTS Settings Panel - For voice selection and options
#[derive(Props, Clone, PartialEq)]
pub struct TtsSettingsPanelProps {
    /// Currently selected voice ID
    #[props(default)]
    pub selected_voice_id: Option<String>,
    /// Handler for voice selection
    pub on_voice_change: EventHandler<String>,
    /// Current speech rate (0.5 - 2.0)
    #[props(default = 1.0)]
    pub rate: f32,
    /// Handler for rate change
    pub on_rate_change: EventHandler<f32>,
}

#[component]
pub fn TtsSettingsPanel(props: TtsSettingsPanelProps) -> Element {
    // Get available voices
    let voices = use_signal(|| {
        crate::saf::get_tts_voices().unwrap_or_default()
    });

    // Filter to English neural voices for better UX
    let english_voices: Vec<_> = voices.read()
        .iter()
        .filter(|v| v.language.starts_with("en-") && v.is_neural)
        .cloned()
        .collect();

    let rate_percent = ((props.rate - 0.5) / 1.5 * 100.0) as i32;

    rsx! {
        div { class: "tts-settings-panel",
            h3 {
                Icon { name: IconName::Headphones }
                "Text-to-Speech Settings"
            }

            div { class: "setting-group",
                label { "Voice" }
                select {
                    class: "voice-select",
                    onchange: move |e| props.on_voice_change.call(e.value().clone()),

                    option { value: "", "Default (Aria)" }

                    for voice in english_voices.iter() {
                        option {
                            value: "{voice.id}",
                            selected: props.selected_voice_id.as_ref() == Some(&voice.id),
                            "{voice.name} ({voice.language})"
                        }
                    }
                }
                span { class: "setting-hint",
                    "Total: {voices.read().len()} voices ({english_voices.len()} English neural)"
                }
            }

            div { class: "setting-group",
                label { "Speed: {props.rate:.1}x" }
                input {
                    r#type: "range",
                    class: "rate-slider",
                    min: "50",
                    max: "200",
                    value: "{rate_percent + 50}",
                    onchange: move |e| {
                        if let Ok(val) = e.value().parse::<f32>() {
                            let rate = (val - 50.0) / 100.0 * 1.5 + 0.5;
                            props.on_rate_change.call(rate);
                        }
                    },
                }
                div { class: "rate-labels",
                    span { "0.5x" }
                    span { "1.0x" }
                    span { "1.5x" }
                    span { "2.0x" }
                }
            }

            div { class: "setting-group",
                label { "Test Voice" }
                ReadAloudButton {
                    text: "Welcome to AudioLearn. This is a sample of the selected voice.".to_string(),
                    label: Some("Play Sample".to_string()),
                    full_width: true,
                }
            }
        }
    }
}

/// Quick TTS Button - Inline button for any text
#[derive(Props, Clone, PartialEq)]
pub struct QuickTtsButtonProps {
    /// Text to speak
    pub text: String,
    /// Optional tooltip
    #[props(default)]
    pub tooltip: Option<String>,
}

#[component]
pub fn QuickTtsButton(props: QuickTtsButtonProps) -> Element {
    let mut is_speaking = use_signal(|| false);
    let text = props.text.clone();
    let tooltip = props.tooltip.clone().unwrap_or_else(|| "Read aloud".to_string());

    let onclick = move |e: MouseEvent| {
        e.stop_propagation();
        let text = text.clone();
        is_speaking.set(true);

        #[cfg(feature = "desktop")]
        spawn(async move {
            let result = tokio::task::spawn_blocking(move || {
                // Stop any existing speech first
                let _ = crate::saf::stop_tts();
                crate::saf::speak_text(&text)
            }).await;

            if let Err(e) = result {
                eprintln!("TTS error: {}", e);
            }
            is_speaking.set(false);
        });
        #[cfg(feature = "web")]
        {
            let _ = crate::saf::stop_tts();
            let result = crate::saf::speak_text(&text);
            if let Err(e) = result {
                eprintln!("TTS error: {}", e);
            }
            is_speaking.set(false);
        }
    };

    rsx! {
        button {
            class: "quick-tts-btn",
            class: if *is_speaking.read() { "speaking" } else { "" },
            title: "{tooltip}",
            disabled: *is_speaking.read(),
            onclick: onclick,

            if *is_speaking.read() {
                Icon { name: IconName::Speaker, size: Size::Sm }
            } else {
                Icon { name: IconName::Headphones, size: Size::Sm }
            }
        }
    }
}

// =============================================================================
// New Feature Components
// =============================================================================

use crate::api::{Bookmark, PlaybackSpeed};
use crate::core::{SleepTimer, SearchResult, SearchEngine, LessonProgress};

/// Full player view with all controls
#[derive(Props, Clone, PartialEq)]
pub struct FullPlayerProps {
    pub title: String,
    pub subtitle: String,
    pub icon: String,
    pub position: Seconds,
    pub duration: Seconds,
    pub is_playing: bool,
    pub speed: PlaybackSpeed,
    pub sleep_timer: SleepTimer,
    pub bookmarks: Vec<Bookmark>,
    pub on_play: EventHandler<()>,
    pub on_pause: EventHandler<()>,
    pub on_seek: EventHandler<Seconds>,
    pub on_skip_back: EventHandler<()>,
    pub on_skip_forward: EventHandler<()>,
    pub on_speed_change: EventHandler<PlaybackSpeed>,
    pub on_sleep_timer_change: EventHandler<SleepTimer>,
    pub on_add_bookmark: EventHandler<()>,
    pub on_close: EventHandler<()>,
}

#[component]
pub fn FullPlayer(props: FullPlayerProps) -> Element {
    let mut show_speed_picker = use_signal(|| false);
    let mut show_sleep_picker = use_signal(|| false);
    let mut show_bookmarks = use_signal(|| false);

    let progress = if props.duration > 0 {
        (props.position as f32 / props.duration as f32) * 100.0
    } else {
        0.0
    };

    let pos = Timestamp::new(props.position);
    let dur = Timestamp::new(props.duration);

    rsx! {
        div { class: "full-player",
            // Header
            header { class: "full-player-header",
                button {
                    class: "close-btn",
                    onclick: move |_| props.on_close.call(()),
                    Icon { name: IconName::ChevronDown }
                }
                h2 { "Now Playing" }
                button {
                    class: "more-btn",
                    Icon { name: IconName::MoreVertical }
                }
            }

            // Album art / icon
            div { class: "full-player-artwork",
                div { class: "artwork-icon", "{props.icon}" }
            }

            // Track info
            div { class: "full-player-info",
                h1 { "{props.title}" }
                p { "{props.subtitle}" }
            }

            // Progress bar with seek
            div { class: "full-player-progress",
                span { class: "time", "{pos.format()}" }
                div { class: "progress-track",
                    div {
                        class: "progress-fill",
                        style: "width: {progress}%",
                    }
                }
                span { class: "time", "{dur.format()}" }
            }

            // Main controls
            div { class: "full-player-controls",
                button {
                    class: "control-btn secondary",
                    onclick: move |_| props.on_skip_back.call(()),
                    "-15s"
                }
                button { class: "control-btn secondary",
                    Icon { name: IconName::SkipBack, size: Size::Lg }
                }
                button {
                    class: "control-btn primary",
                    onclick: move |_| {
                        if props.is_playing {
                            props.on_pause.call(());
                        } else {
                            props.on_play.call(());
                        }
                    },
                    if props.is_playing {
                        Icon { name: IconName::Pause, size: Size::Xl }
                    } else {
                        Icon { name: IconName::Play, size: Size::Xl }
                    }
                }
                button { class: "control-btn secondary",
                    Icon { name: IconName::SkipForward, size: Size::Lg }
                }
                button {
                    class: "control-btn secondary",
                    onclick: move |_| props.on_skip_forward.call(()),
                    "+15s"
                }
            }

            // Secondary controls
            div { class: "full-player-extras",
                button {
                    class: "extra-btn",
                    onclick: move |_| show_speed_picker.set(true),
                    Icon { name: IconName::Sliders }
                    span { "{props.speed.label()}" }
                }
                button {
                    class: "extra-btn",
                    onclick: move |_| show_sleep_picker.set(true),
                    Icon { name: IconName::Moon }
                    span { "{props.sleep_timer.label()}" }
                }
                button {
                    class: "extra-btn",
                    onclick: move |_| props.on_add_bookmark.call(()),
                    Icon { name: IconName::Bookmark }
                    span { "Bookmark" }
                }
                button {
                    class: "extra-btn",
                    onclick: move |_| show_bookmarks.set(true),
                    Icon { name: IconName::List }
                    if !props.bookmarks.is_empty() {
                        span { class: "badge", "{props.bookmarks.len()}" }
                    }
                }
            }

            // Speed picker modal
            if *show_speed_picker.read() {
                PlaybackSpeedPicker {
                    current: props.speed.clone(),
                    on_select: move |speed| {
                        props.on_speed_change.call(speed);
                        show_speed_picker.set(false);
                    },
                    on_close: move |_| show_speed_picker.set(false),
                }
            }

            // Sleep timer picker modal
            if *show_sleep_picker.read() {
                SleepTimerPicker {
                    current: props.sleep_timer.clone(),
                    on_select: move |timer| {
                        props.on_sleep_timer_change.call(timer);
                        show_sleep_picker.set(false);
                    },
                    on_close: move |_| show_sleep_picker.set(false),
                }
            }
        }
    }
}

/// Playback speed picker
#[derive(Props, Clone, PartialEq)]
pub struct PlaybackSpeedPickerProps {
    pub current: PlaybackSpeed,
    pub on_select: EventHandler<PlaybackSpeed>,
    pub on_close: EventHandler<()>,
}

#[component]
pub fn PlaybackSpeedPicker(props: PlaybackSpeedPickerProps) -> Element {
    let speeds = vec![
        PlaybackSpeed::Half,
        PlaybackSpeed::ThreeQuarter,
        PlaybackSpeed::Normal,
        PlaybackSpeed::OneQuarter,
        PlaybackSpeed::OneHalf,
        PlaybackSpeed::OneThreeQuarter,
        PlaybackSpeed::Double,
    ];

    rsx! {
        div {
            class: "picker-overlay",
            onclick: move |_| props.on_close.call(()),

            div {
                class: "picker-modal",
                onclick: move |e| e.stop_propagation(),

                h3 { "Playback Speed" }

                div { class: "picker-options",
                    for speed in speeds.iter() {
                        button {
                            class: if *speed == props.current { "option active" } else { "option" },
                            onclick: {
                                let s = speed.clone();
                                move |_| props.on_select.call(s.clone())
                            },
                            span { class: "option-label", "{speed.label()}" }
                            if *speed == props.current {
                                Icon { name: IconName::Check }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Sleep timer picker
#[derive(Props, Clone, PartialEq)]
pub struct SleepTimerPickerProps {
    pub current: SleepTimer,
    pub on_select: EventHandler<SleepTimer>,
    pub on_close: EventHandler<()>,
}

#[component]
pub fn SleepTimerPicker(props: SleepTimerPickerProps) -> Element {
    let timers = SleepTimer::all();

    rsx! {
        div {
            class: "picker-overlay",
            onclick: move |_| props.on_close.call(()),

            div {
                class: "picker-modal",
                onclick: move |e| e.stop_propagation(),

                h3 {
                    Icon { name: IconName::Moon }
                    "Sleep Timer"
                }

                div { class: "picker-options",
                    for timer in timers.iter() {
                        button {
                            class: if *timer == props.current { "option active" } else { "option" },
                            onclick: {
                                let t = timer.clone();
                                move |_| props.on_select.call(t.clone())
                            },
                            span { class: "option-label", "{timer.label()}" }
                            if *timer == props.current {
                                Icon { name: IconName::Check }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Search modal
#[derive(Props, Clone, PartialEq)]
pub struct SearchModalProps {
    pub courses: Vec<Course>,
    pub on_course_select: EventHandler<String>,
    pub on_lesson_select: EventHandler<(String, String)>, // (course_id, lesson_id)
    pub on_close: EventHandler<()>,
}

#[component]
pub fn SearchModal(props: SearchModalProps) -> Element {
    let mut query = use_signal(|| String::new());
    let search_engine = use_memo(move || {
        SearchEngine::new(props.courses.clone())
    });

    let results = use_memo(move || {
        let q = query.read().clone();
        if q.len() < 2 {
            Vec::new()
        } else {
            search_engine.read().search(&q)
        }
    });

    rsx! {
        div {
            class: "search-overlay",
            onclick: move |_| props.on_close.call(()),

            div {
                class: "search-modal",
                onclick: move |e| e.stop_propagation(),

                // Search input
                div { class: "search-input-wrapper",
                    Icon { name: IconName::Search }
                    input {
                        class: "search-input",
                        r#type: "text",
                        placeholder: "Search courses, lessons...",
                        value: "{query}",
                        oninput: move |e| query.set(e.value()),
                        autofocus: true,
                    }
                    if !query.read().is_empty() {
                        button {
                            class: "clear-btn",
                            onclick: move |_| query.set(String::new()),
                            Icon { name: IconName::X }
                        }
                    }
                }

                // Results
                div { class: "search-results",
                    if results.read().is_empty() && query.read().len() >= 2 {
                        div { class: "no-results",
                            Icon { name: IconName::Search, size: Size::Xl }
                            p { "No results found for \"{query.read()}\"" }
                        }
                    } else if results.read().is_empty() && query.read().len() < 2 {
                        div { class: "search-hint",
                            p { "Type at least 2 characters to search" }
                        }
                    } else {
                        for result in results.read().iter() {
                            match result {
                                SearchResult::Course(course_result) => rsx! {
                                    button {
                                        class: "search-result course-result",
                                        onclick: {
                                            let id = course_result.course.id.clone();
                                            move |_| {
                                                props.on_course_select.call(id.clone());
                                                props.on_close.call(());
                                            }
                                        },
                                        span { class: "result-icon", "{course_result.course.icon}" }
                                        div { class: "result-info",
                                            span { class: "result-title", "{course_result.course.title}" }
                                            span { class: "result-meta",
                                                "Course • {course_result.course.lesson_count()} lessons"
                                            }
                                        }
                                    }
                                },
                                SearchResult::Lesson(lesson_result) => rsx! {
                                    button {
                                        class: "search-result lesson-result",
                                        onclick: {
                                            let cid = lesson_result.course_id.clone();
                                            let lid = lesson_result.lesson.id.clone();
                                            move |_| {
                                                props.on_lesson_select.call((cid.clone(), lid.clone()));
                                                props.on_close.call(());
                                            }
                                        },
                                        Icon { name: IconName::Headphones }
                                        div { class: "result-info",
                                            span { class: "result-title", "{lesson_result.lesson.title}" }
                                            span { class: "result-meta",
                                                "Lesson • {lesson_result.course_title}"
                                            }
                                        }
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

/// Bookmark list component
#[derive(Props, Clone, PartialEq)]
pub struct BookmarkListProps {
    pub bookmarks: Vec<Bookmark>,
    pub on_select: EventHandler<Bookmark>,
    pub on_delete: EventHandler<String>,
}

#[component]
pub fn BookmarkList(props: BookmarkListProps) -> Element {
    if props.bookmarks.is_empty() {
        return rsx! {
            div { class: "bookmark-empty",
                Icon { name: IconName::Bookmark, size: Size::Xl }
                h3 { "No bookmarks yet" }
                p { "Tap the bookmark button while listening to save your place" }
            }
        };
    }

    rsx! {
        div { class: "bookmark-list",
            for bookmark in props.bookmarks.iter() {
                div {
                    class: "bookmark-item",
                    onclick: {
                        let b = bookmark.clone();
                        move |_| props.on_select.call(b.clone())
                    },

                    div { class: "bookmark-time",
                        Icon { name: IconName::Bookmark }
                        span { "{bookmark.timestamp.format()}" }
                    }

                    if let Some(note) = &bookmark.note {
                        p { class: "bookmark-note", "{note}" }
                    }

                    {
                        let date_str = bookmark.created_at.format("%b %d, %H:%M").to_string();
                        rsx! { span { class: "bookmark-date", "{date_str}" } }
                    }

                    button {
                        class: "bookmark-delete",
                        onclick: {
                            let id = bookmark.id.clone();
                            move |e: Event<MouseData>| {
                                e.stop_propagation();
                                props.on_delete.call(id.clone());
                            }
                        },
                        Icon { name: IconName::Trash, size: Size::Sm }
                    }
                }
            }
        }
    }
}

/// Continue learning card
#[derive(Props, Clone, PartialEq)]
pub struct ContinueLearningCardProps {
    pub lesson_title: String,
    pub course_title: String,
    pub icon: String,
    pub progress: LessonProgress,
    pub on_click: EventHandler<()>,
}

#[component]
pub fn ContinueLearningCard(props: ContinueLearningCardProps) -> Element {
    let progress_pct = props.progress.percentage();
    let remaining = props.progress.duration.saturating_sub(props.progress.position);
    let remaining_str = Timestamp::new(remaining).format_long();

    rsx! {
        div {
            class: "continue-card",
            onclick: move |_| props.on_click.call(()),

            div { class: "continue-icon", "{props.icon}" }

            div { class: "continue-info",
                span { class: "continue-lesson", "{props.lesson_title}" }
                span { class: "continue-course", "{props.course_title}" }

                div { class: "continue-progress",
                    div { class: "progress-bar",
                        div {
                            class: "progress-fill",
                            style: "width: {progress_pct}%",
                        }
                    }
                    span { class: "remaining", "{remaining_str} left" }
                }
            }

            button { class: "continue-play",
                Icon { name: IconName::Play, size: Size::Lg }
            }
        }
    }
}
