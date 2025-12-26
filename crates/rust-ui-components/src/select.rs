//! Select component

use dioxus::prelude::*;
use crate::Size;

/// Select option
#[derive(Clone, PartialEq)]
pub struct SelectOption {
    pub value: String,
    pub label: String,
    pub disabled: bool,
}

impl SelectOption {
    pub fn new(value: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            label: label.into(),
            disabled: false,
        }
    }
}

impl<V: Into<String>, L: Into<String>> From<(V, L)> for SelectOption {
    fn from((value, label): (V, L)) -> Self {
        Self::new(value, label)
    }
}

/// Select component props
#[derive(Props, Clone, PartialEq)]
pub struct SelectProps {
    /// Current value
    #[props(default)]
    pub value: String,

    /// Options list
    pub options: Vec<SelectOption>,

    /// Label text
    #[props(default)]
    pub label: Option<String>,

    /// Placeholder text
    #[props(default)]
    pub placeholder: Option<String>,

    /// Error message
    #[props(default)]
    pub error: Option<String>,

    /// Select size
    #[props(default)]
    pub size: Size,

    /// Disabled state
    #[props(default = false)]
    pub disabled: bool,

    /// Required field
    #[props(default = false)]
    pub required: bool,

    /// Change handler
    #[props(default)]
    pub onchange: Option<EventHandler<String>>,

    /// Additional CSS class
    #[props(default)]
    pub class: Option<String>,
}

/// Select component
///
/// # Example
/// ```rust
/// rsx! {
///     Select {
///         label: "Region",
///         value: region,
///         onchange: move |v| region.set(v),
///         options: vec![
///             SelectOption::new("us-east-1", "US East"),
///             SelectOption::new("us-west-2", "US West"),
///             SelectOption::new("eu-west-1", "EU Ireland"),
///         ],
///     }
/// }
/// ```
#[component]
pub fn Select(props: SelectProps) -> Element {
    let has_error = props.error.is_some();

    let wrapper_class = format!(
        "rust-ui-select-wrapper {} {}",
        props.size.class(),
        if has_error { "has-error" } else { "" },
    );

    rsx! {
        div { class: "{wrapper_class}",
            if let Some(label) = &props.label {
                label { class: "select-label",
                    "{label}"
                    if props.required {
                        span { class: "required-mark", " *" }
                    }
                }
            }

            div { class: "select-container",
                select {
                    class: "rust-ui-select {props.class.as_deref().unwrap_or(\"\")}",
                    disabled: props.disabled,
                    required: props.required,
                    onchange: move |evt| {
                        if let Some(handler) = &props.onchange {
                            handler.call(evt.value());
                        }
                    },

                    // Placeholder option
                    if let Some(placeholder) = &props.placeholder {
                        option {
                            value: "",
                            disabled: true,
                            selected: props.value.is_empty(),
                            "{placeholder}"
                        }
                    }

                    // Options
                    for opt in &props.options {
                        option {
                            value: "{opt.value}",
                            disabled: opt.disabled,
                            selected: props.value == opt.value,
                            "{opt.label}"
                        }
                    }
                }

                // Dropdown arrow
                span { class: "select-arrow",
                    svg {
                        width: "16",
                        height: "16",
                        view_box: "0 0 24 24",
                        fill: "none",
                        stroke: "currentColor",
                        stroke_width: "2",
                        polyline { points: "6 9 12 15 18 9" }
                    }
                }
            }

            if let Some(error) = &props.error {
                p { class: "select-error", "{error}" }
            }
        }
    }
}
