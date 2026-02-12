//! Avatar component

use dioxus::prelude::*;
use crate::api::types::Size;

/// Avatar shape
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum AvatarShape {
    #[default]
    Circle,
    Square,
    Rounded,
}

impl AvatarShape {
    pub fn class(&self) -> &'static str {
        match self {
            AvatarShape::Circle => "shape-circle",
            AvatarShape::Square => "shape-square",
            AvatarShape::Rounded => "shape-rounded",
        }
    }
}

/// Avatar status
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AvatarStatus {
    Online,
    Away,
    Busy,
    Offline,
}

impl AvatarStatus {
    pub fn class(&self) -> &'static str {
        match self {
            AvatarStatus::Online => "status-online",
            AvatarStatus::Away => "status-away",
            AvatarStatus::Busy => "status-busy",
            AvatarStatus::Offline => "status-offline",
        }
    }
}

/// Avatar props
#[derive(Props, Clone, PartialEq)]
pub struct AvatarProps {
    /// Image source URL
    #[props(default)]
    pub src: Option<String>,

    /// Alt text
    #[props(default)]
    pub alt: Option<String>,

    /// Fallback text (initials)
    #[props(default)]
    pub fallback: Option<String>,

    /// Size
    #[props(default)]
    pub size: Size,

    /// Shape
    #[props(default)]
    pub shape: AvatarShape,

    /// Status indicator
    #[props(default)]
    pub status: Option<AvatarStatus>,

    /// Border
    #[props(default = false)]
    pub bordered: bool,
}

/// Avatar component
#[component]
pub fn Avatar(props: AvatarProps) -> Element {
    let class = format!(
        "rust-ui-avatar {} {} {}",
        props.size.class(),
        props.shape.class(),
        if props.bordered { "bordered" } else { "" },
    );

    let initials = props.fallback.clone().unwrap_or_else(|| {
        props.alt.as_ref()
            .map(|s| s.chars().take(2).collect::<String>().to_uppercase())
            .unwrap_or_else(|| "?".to_string())
    });

    rsx! {
        div { class: "{class}",
            if let Some(src) = &props.src {
                img {
                    class: "avatar-image",
                    src: "{src}",
                    alt: props.alt.as_deref().unwrap_or("Avatar"),
                }
            } else {
                span { class: "avatar-fallback", "{initials}" }
            }

            if let Some(status) = &props.status {
                span { class: "avatar-status {status.class()}" }
            }
        }
    }
}

/// Avatar group props
#[derive(Props, Clone, PartialEq)]
pub struct AvatarGroupProps {
    children: Element,

    /// Maximum visible avatars
    #[props(default = 4)]
    pub max: usize,

    /// Size for all avatars
    #[props(default)]
    pub size: Size,

    /// Total count (for +N indicator)
    #[props(default)]
    pub total: Option<usize>,
}

/// Avatar group component
#[component]
pub fn AvatarGroup(props: AvatarGroupProps) -> Element {
    rsx! {
        div { class: "rust-ui-avatar-group {props.size.class()}",
            {props.children}

            if let Some(total) = props.total {
                if total > props.max {
                    div { class: "avatar-overflow",
                        "+{total - props.max}"
                    }
                }
            }
        }
    }
}
