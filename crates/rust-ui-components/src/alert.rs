//! Alert component

use dioxus::prelude::*;
use crate::Variant;

/// Alert component props
#[derive(Props, Clone, PartialEq)]
pub struct AlertProps {
    /// Alert content
    children: Element,

    /// Alert title
    #[props(default)]
    pub title: Option<String>,

    /// Visual variant
    #[props(default = Variant::Default)]
    pub variant: Variant,

    /// Dismissible
    #[props(default = false)]
    pub dismissible: bool,

    /// Show icon
    #[props(default = true)]
    pub show_icon: bool,

    /// Dismiss handler
    #[props(default)]
    pub ondismiss: Option<EventHandler<()>>,
}

/// Alert component
///
/// # Example
/// ```rust
/// rsx! {
///     Alert { variant: Variant::Success, "Operation completed!" }
///     Alert {
///         variant: Variant::Danger,
///         title: "Error",
///         dismissible: true,
///         "Something went wrong."
///     }
/// }
/// ```
#[component]
pub fn Alert(props: AlertProps) -> Element {
    let class = format!("rust-ui-alert {}", props.variant.class());

    let icon = match props.variant {
        Variant::Success => rsx! {
            svg { width: "20", height: "20", view_box: "0 0 24 24", fill: "none", stroke: "currentColor", stroke_width: "2",
                path { d: "M22 11.08V12a10 10 0 1 1-5.93-9.14" }
                polyline { points: "22 4 12 14.01 9 11.01" }
            }
        },
        Variant::Warning => rsx! {
            svg { width: "20", height: "20", view_box: "0 0 24 24", fill: "none", stroke: "currentColor", stroke_width: "2",
                path { d: "M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" }
                line { x1: "12", y1: "9", x2: "12", y2: "13" }
                line { x1: "12", y1: "17", x2: "12.01", y2: "17" }
            }
        },
        Variant::Danger => rsx! {
            svg { width: "20", height: "20", view_box: "0 0 24 24", fill: "none", stroke: "currentColor", stroke_width: "2",
                circle { cx: "12", cy: "12", r: "10" }
                line { x1: "15", y1: "9", x2: "9", y2: "15" }
                line { x1: "9", y1: "9", x2: "15", y2: "15" }
            }
        },
        _ => rsx! {
            svg { width: "20", height: "20", view_box: "0 0 24 24", fill: "none", stroke: "currentColor", stroke_width: "2",
                circle { cx: "12", cy: "12", r: "10" }
                line { x1: "12", y1: "16", x2: "12", y2: "12" }
                line { x1: "12", y1: "8", x2: "12.01", y2: "8" }
            }
        },
    };

    rsx! {
        div { class: "{class}", role: "alert",
            if props.show_icon {
                span { class: "alert-icon", {icon} }
            }

            div { class: "alert-content",
                if let Some(title) = &props.title {
                    h4 { class: "alert-title", "{title}" }
                }
                div { class: "alert-message", {props.children} }
            }

            if props.dismissible {
                button {
                    class: "alert-dismiss",
                    onclick: move |_| {
                        if let Some(handler) = &props.ondismiss {
                            handler.call(());
                        }
                    },
                    "×"
                }
            }
        }
    }
}
