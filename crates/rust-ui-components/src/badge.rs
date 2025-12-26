//! Badge component

use dioxus::prelude::*;
use crate::{Variant, Size};

/// Badge component props
#[derive(Props, Clone, PartialEq)]
pub struct BadgeProps {
    /// Badge content
    children: Element,

    /// Visual variant
    #[props(default)]
    pub variant: Variant,

    /// Badge size
    #[props(default)]
    pub size: Size,

    /// Pill style (fully rounded)
    #[props(default = false)]
    pub pill: bool,

    /// Outline style
    #[props(default = false)]
    pub outline: bool,

    /// Dot indicator (no text)
    #[props(default = false)]
    pub dot: bool,
}

/// Badge component
///
/// # Example
/// ```rust
/// rsx! {
///     Badge { "New" }
///     Badge { variant: Variant::Success, "Active" }
///     Badge { variant: Variant::Danger, pill: true, "3" }
///     Badge { variant: Variant::Warning, outline: true, "Pending" }
/// }
/// ```
#[component]
pub fn Badge(props: BadgeProps) -> Element {
    let class = format!(
        "rust-ui-badge {} {} {} {}",
        props.variant.class(),
        props.size.class(),
        if props.pill { "pill" } else { "" },
        if props.outline { "outline" } else { "" },
    );

    if props.dot {
        rsx! {
            span { class: "rust-ui-badge-dot {}", class: props.variant.class() }
        }
    } else {
        rsx! {
            span { class: "{class}", {props.children} }
        }
    }
}

/// Status badge with dot indicator
#[derive(Props, Clone, PartialEq)]
pub struct StatusBadgeProps {
    /// Status label
    pub label: String,

    /// Status variant
    #[props(default)]
    pub variant: Variant,
}

#[component]
pub fn StatusBadge(props: StatusBadgeProps) -> Element {
    rsx! {
        span { class: "rust-ui-status-badge",
            span { class: "status-dot {}", class: props.variant.class() }
            span { class: "status-label", "{props.label}" }
        }
    }
}
