//! Tooltip component

use dioxus::prelude::*;

/// Tooltip position
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum TooltipPosition {
    #[default]
    Top,
    Bottom,
    Left,
    Right,
}

impl TooltipPosition {
    pub fn class(&self) -> &'static str {
        match self {
            TooltipPosition::Top => "tooltip-top",
            TooltipPosition::Bottom => "tooltip-bottom",
            TooltipPosition::Left => "tooltip-left",
            TooltipPosition::Right => "tooltip-right",
        }
    }
}

/// Tooltip props
#[derive(Props, Clone, PartialEq)]
pub struct TooltipProps {
    /// Trigger element
    children: Element,

    /// Tooltip content
    pub content: String,

    /// Position
    #[props(default)]
    pub position: TooltipPosition,

    /// Show delay (ms)
    #[props(default = 200)]
    pub delay: u32,

    /// Disabled
    #[props(default = false)]
    pub disabled: bool,
}

/// Tooltip component
#[component]
pub fn Tooltip(props: TooltipProps) -> Element {
    let mut visible = use_signal(|| false);

    if props.disabled {
        return rsx! { {props.children} };
    }

    rsx! {
        div {
            class: "rust-ui-tooltip-wrapper",
            onmouseenter: move |_| visible.set(true),
            onmouseleave: move |_| visible.set(false),

            {props.children}

            if *visible.read() {
                div {
                    class: "rust-ui-tooltip {props.position.class()}",
                    role: "tooltip",
                    "{props.content}"
                }
            }
        }
    }
}
