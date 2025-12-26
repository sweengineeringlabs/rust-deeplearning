//! Button component

use dioxus::prelude::*;
use crate::{Variant, Size};

/// Button component props
#[derive(Props, Clone, PartialEq)]
pub struct ButtonProps {
    /// Button content
    children: Element,

    /// Visual variant
    #[props(default)]
    pub variant: Variant,

    /// Button size
    #[props(default)]
    pub size: Size,

    /// Disabled state
    #[props(default = false)]
    pub disabled: bool,

    /// Loading state
    #[props(default = false)]
    pub loading: bool,

    /// Full width
    #[props(default = false)]
    pub full_width: bool,

    /// Icon (left side)
    #[props(default)]
    pub icon: Option<Element>,

    /// Icon position
    #[props(default)]
    pub icon_right: bool,

    /// Button type
    #[props(default = "button")]
    pub r#type: &'static str,

    /// Click handler
    #[props(default)]
    pub onclick: Option<EventHandler<MouseEvent>>,

    /// Additional CSS class
    #[props(default)]
    pub class: Option<String>,
}

/// Button component
///
/// # Example
/// ```rust
/// rsx! {
///     Button { variant: Variant::Primary, "Click me" }
///     Button { variant: Variant::Danger, disabled: true, "Delete" }
///     Button { loading: true, "Saving..." }
/// }
/// ```
#[component]
pub fn Button(props: ButtonProps) -> Element {
    let class = format!(
        "rust-ui-button {} {} {} {} {}",
        props.variant.class(),
        props.size.class(),
        if props.full_width { "full-width" } else { "" },
        if props.loading { "loading" } else { "" },
        props.class.as_deref().unwrap_or(""),
    );

    let disabled = props.disabled || props.loading;

    rsx! {
        button {
            class: "{class}",
            r#type: "{props.r#type}",
            disabled: disabled,
            onclick: move |evt| {
                if let Some(handler) = &props.onclick {
                    handler.call(evt);
                }
            },

            if props.loading {
                span { class: "button-spinner",
                    // Spinner SVG
                    svg {
                        class: "animate-spin",
                        width: "16",
                        height: "16",
                        view_box: "0 0 24 24",
                        circle {
                            cx: "12",
                            cy: "12",
                            r: "10",
                            stroke: "currentColor",
                            stroke_width: "3",
                            fill: "none",
                            opacity: "0.25",
                        }
                        path {
                            d: "M12 2a10 10 0 0 1 10 10",
                            stroke: "currentColor",
                            stroke_width: "3",
                            fill: "none",
                            stroke_linecap: "round",
                        }
                    }
                }
            }

            if !props.icon_right {
                if let Some(icon) = &props.icon {
                    span { class: "button-icon left", {icon} }
                }
            }

            span { class: "button-content", {props.children} }

            if props.icon_right {
                if let Some(icon) = &props.icon {
                    span { class: "button-icon right", {icon} }
                }
            }
        }
    }
}

/// Icon-only button
#[derive(Props, Clone, PartialEq)]
pub struct IconButtonProps {
    /// Icon element
    icon: Element,

    /// Visual variant
    #[props(default)]
    pub variant: Variant,

    /// Button size
    #[props(default)]
    pub size: Size,

    /// Disabled state
    #[props(default = false)]
    pub disabled: bool,

    /// Accessibility label
    #[props(default)]
    pub aria_label: Option<String>,

    /// Click handler
    #[props(default)]
    pub onclick: Option<EventHandler<MouseEvent>>,
}

#[component]
pub fn IconButton(props: IconButtonProps) -> Element {
    let class = format!(
        "rust-ui-icon-button {} {}",
        props.variant.class(),
        props.size.class(),
    );

    rsx! {
        button {
            class: "{class}",
            disabled: props.disabled,
            aria_label: props.aria_label,
            onclick: move |evt| {
                if let Some(handler) = &props.onclick {
                    handler.call(evt);
                }
            },
            {props.icon}
        }
    }
}
