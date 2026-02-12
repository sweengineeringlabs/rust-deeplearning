//! Progress bar component

use dioxus::prelude::*;
use crate::api::types::{Variant, Size};

/// Progress bar props
#[derive(Props, Clone, PartialEq)]
pub struct ProgressProps {
    /// Current value (0-100)
    pub value: f32,

    /// Maximum value
    #[props(default = 100.0)]
    pub max: f32,

    /// Visual variant
    #[props(default)]
    pub variant: Variant,

    /// Size
    #[props(default)]
    pub size: Size,

    /// Show percentage label
    #[props(default = false)]
    pub show_label: bool,

    /// Custom label format
    #[props(default)]
    pub label: Option<String>,

    /// Striped style
    #[props(default = false)]
    pub striped: bool,

    /// Animated stripes
    #[props(default = false)]
    pub animated: bool,

    /// Indeterminate (loading) state
    #[props(default = false)]
    pub indeterminate: bool,
}

/// Progress bar component
#[component]
pub fn Progress(props: ProgressProps) -> Element {
    let percentage = ((props.value / props.max) * 100.0).clamp(0.0, 100.0);

    let class = format!(
        "rust-ui-progress {} {} {} {}",
        props.variant.class(),
        props.size.class(),
        if props.striped { "striped" } else { "" },
        if props.animated { "animated" } else { "" },
    );

    let label = props.label.clone().unwrap_or_else(|| format!("{:.0}%", percentage));

    rsx! {
        div { class: "{class}",
            div {
                class: "progress-track",
                role: "progressbar",
                aria_valuenow: "{props.value}",
                aria_valuemin: "0",
                aria_valuemax: "{props.max}",

                div {
                    class: "progress-bar",
                    class: if props.indeterminate { "indeterminate" } else { "" },
                    style: if !props.indeterminate { format!("width: {}%", percentage) } else { String::new() },

                    if props.show_label && !props.indeterminate {
                        span { class: "progress-label", "{label}" }
                    }
                }
            }

            if props.show_label && props.indeterminate {
                span { class: "progress-label-outside", "Loading..." }
            }
        }
    }
}

/// Circular progress props
#[derive(Props, Clone, PartialEq)]
pub struct CircularProgressProps {
    /// Current value (0-100)
    #[props(default = 0.0)]
    pub value: f32,

    /// Size in pixels
    #[props(default = 40)]
    pub size: u32,

    /// Stroke width
    #[props(default = 4)]
    pub stroke_width: u32,

    /// Variant
    #[props(default)]
    pub variant: Variant,

    /// Show percentage
    #[props(default = false)]
    pub show_label: bool,

    /// Indeterminate
    #[props(default = false)]
    pub indeterminate: bool,
}

/// Circular progress component
#[component]
pub fn CircularProgress(props: CircularProgressProps) -> Element {
    let radius = (props.size as f32 / 2.0) - (props.stroke_width as f32 / 2.0);
    let circumference = 2.0 * std::f32::consts::PI * radius;
    let offset = circumference - (props.value / 100.0) * circumference;

    rsx! {
        div {
            class: "rust-ui-circular-progress {props.variant.class()}",
            class: if props.indeterminate { "indeterminate" } else { "" },
            style: "width: {props.size}px; height: {props.size}px",

            svg {
                width: "{props.size}",
                height: "{props.size}",
                view_box: "0 0 {props.size} {props.size}",

                // Background circle
                circle {
                    class: "progress-bg",
                    cx: "{props.size / 2}",
                    cy: "{props.size / 2}",
                    r: "{radius}",
                    stroke_width: "{props.stroke_width}",
                    fill: "none",
                }

                // Progress circle
                circle {
                    class: "progress-bar",
                    cx: "{props.size / 2}",
                    cy: "{props.size / 2}",
                    r: "{radius}",
                    stroke_width: "{props.stroke_width}",
                    fill: "none",
                    stroke_dasharray: "{circumference}",
                    stroke_dashoffset: if props.indeterminate { "0" } else { "{offset}" },
                    transform: "rotate(-90 {props.size / 2} {props.size / 2})",
                }
            }

            if props.show_label && !props.indeterminate {
                span { class: "progress-label", "{props.value:.0}%" }
            }
        }
    }
}
