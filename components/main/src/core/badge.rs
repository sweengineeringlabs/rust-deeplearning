//! Badge component

use dioxus::prelude::*;
use crate::api::types::{Variant, Size};

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
#[component]
pub fn Badge(props: BadgeProps) -> Element {
    if props.dot {
        let dot_class = format!("rust-ui-badge-dot {}", props.variant.class());
        rsx! {
            span { class: "{dot_class}" }
        }
    } else {
        let class = format!(
            "rust-ui-badge {} {} {} {}",
            props.variant.class(),
            props.size.class(),
            if props.pill { "pill" } else { "" },
            if props.outline { "outline" } else { "" },
        );
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
    let dot_class = format!("status-dot {}", props.variant.class());
    rsx! {
        span { class: "rust-ui-status-badge",
            span { class: "{dot_class}" }
            span { class: "status-label", "{props.label}" }
        }
    }
}
