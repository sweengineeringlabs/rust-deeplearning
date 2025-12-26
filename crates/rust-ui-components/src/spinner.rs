//! Spinner/loading components

use dioxus::prelude::*;
use crate::Size;

/// Spinner component props
#[derive(Props, Clone, PartialEq)]
pub struct SpinnerProps {
    /// Spinner size
    #[props(default)]
    pub size: Size,

    /// Custom color
    #[props(default)]
    pub color: Option<String>,

    /// Accessibility label
    #[props(default = "Loading")]
    pub label: &'static str,
}

/// Spinner component
///
/// # Example
/// ```rust
/// rsx! {
///     Spinner {}
///     Spinner { size: Size::Lg }
///     Spinner { color: "#3b82f6" }
/// }
/// ```
#[component]
pub fn Spinner(props: SpinnerProps) -> Element {
    let size_px = match props.size {
        Size::Xs => 12,
        Size::Sm => 16,
        Size::Md => 24,
        Size::Lg => 32,
        Size::Xl => 48,
    };

    let color = props.color.as_deref().unwrap_or("currentColor");

    rsx! {
        svg {
            class: "rust-ui-spinner",
            width: "{size_px}",
            height: "{size_px}",
            view_box: "0 0 24 24",
            "aria-label": "{props.label}",

            circle {
                cx: "12",
                cy: "12",
                r: "10",
                stroke: "{color}",
                stroke_width: "3",
                fill: "none",
                opacity: "0.25",
            }
            path {
                d: "M12 2a10 10 0 0 1 10 10",
                stroke: "{color}",
                stroke_width: "3",
                fill: "none",
                stroke_linecap: "round",
            }
        }
    }
}

/// Skeleton loading placeholder
#[derive(Props, Clone, PartialEq)]
pub struct SkeletonProps {
    /// Width (CSS value)
    #[props(default = "100%")]
    pub width: &'static str,

    /// Height (CSS value)
    #[props(default = "1rem")]
    pub height: &'static str,

    /// Rounded corners
    #[props(default = false)]
    pub rounded: bool,

    /// Circle shape
    #[props(default = false)]
    pub circle: bool,
}

#[component]
pub fn Skeleton(props: SkeletonProps) -> Element {
    let class = format!(
        "rust-ui-skeleton {} {}",
        if props.rounded { "rounded" } else { "" },
        if props.circle { "circle" } else { "" },
    );

    let style = if props.circle {
        format!("width: {}; height: {};", props.width, props.width)
    } else {
        format!("width: {}; height: {};", props.width, props.height)
    };

    rsx! {
        div { class: "{class}", style: "{style}" }
    }
}

/// Loading overlay
#[derive(Props, Clone, PartialEq)]
pub struct LoadingOverlayProps {
    /// Loading state
    pub loading: bool,

    /// Content to overlay
    children: Element,

    /// Loading text
    #[props(default)]
    pub text: Option<String>,
}

#[component]
pub fn LoadingOverlay(props: LoadingOverlayProps) -> Element {
    rsx! {
        div { class: "rust-ui-loading-overlay-container",
            {props.children}

            if props.loading {
                div { class: "loading-overlay",
                    div { class: "loading-content",
                        Spinner { size: Size::Lg }
                        if let Some(text) = &props.text {
                            p { class: "loading-text", "{text}" }
                        }
                    }
                }
            }
        }
    }
}
