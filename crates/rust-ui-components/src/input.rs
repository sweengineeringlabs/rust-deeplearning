//! Input component

use dioxus::prelude::*;
use crate::Size;

/// Input types
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum InputType {
    #[default]
    Text,
    Password,
    Email,
    Number,
    Tel,
    Url,
    Search,
    Date,
    Time,
    DateTime,
}

impl InputType {
    pub fn as_str(&self) -> &'static str {
        match self {
            InputType::Text => "text",
            InputType::Password => "password",
            InputType::Email => "email",
            InputType::Number => "number",
            InputType::Tel => "tel",
            InputType::Url => "url",
            InputType::Search => "search",
            InputType::Date => "date",
            InputType::Time => "time",
            InputType::DateTime => "datetime-local",
        }
    }
}

/// Input component props
#[derive(Props, Clone, PartialEq)]
pub struct InputProps {
    /// Input value
    #[props(default)]
    pub value: String,

    /// Placeholder text
    #[props(default)]
    pub placeholder: Option<String>,

    /// Label text
    #[props(default)]
    pub label: Option<String>,

    /// Helper text
    #[props(default)]
    pub helper: Option<String>,

    /// Error message
    #[props(default)]
    pub error: Option<String>,

    /// Input type
    #[props(default)]
    pub input_type: InputType,

    /// Input size
    #[props(default)]
    pub size: Size,

    /// Disabled state
    #[props(default = false)]
    pub disabled: bool,

    /// Required field
    #[props(default = false)]
    pub required: bool,

    /// Read-only field
    #[props(default = false)]
    pub readonly: bool,

    /// Show clear button
    #[props(default = false)]
    pub clearable: bool,

    /// Left icon
    #[props(default)]
    pub icon: Option<Element>,

    /// Right icon/element
    #[props(default)]
    pub suffix: Option<Element>,

    /// Change handler
    #[props(default)]
    pub onchange: Option<EventHandler<String>>,

    /// Input handler (on every keystroke)
    #[props(default)]
    pub oninput: Option<EventHandler<String>>,

    /// Focus handler
    #[props(default)]
    pub onfocus: Option<EventHandler<FocusEvent>>,

    /// Blur handler
    #[props(default)]
    pub onblur: Option<EventHandler<FocusEvent>>,

    /// Additional CSS class
    #[props(default)]
    pub class: Option<String>,
}

/// Input component
///
/// # Example
/// ```rust
/// rsx! {
///     Input {
///         label: "Email",
///         placeholder: "you@example.com",
///         input_type: InputType::Email,
///         value: email,
///         onchange: move |v| email.set(v),
///     }
/// }
/// ```
#[component]
pub fn Input(props: InputProps) -> Element {
    let has_error = props.error.is_some();

    let wrapper_class = format!(
        "rust-ui-input-wrapper {} {}",
        props.size.class(),
        if has_error { "has-error" } else { "" },
    );

    let input_class = format!(
        "rust-ui-input {} {}",
        if props.icon.is_some() { "has-icon" } else { "" },
        props.class.as_deref().unwrap_or(""),
    );

    rsx! {
        div { class: "{wrapper_class}",
            // Label
            if let Some(label) = &props.label {
                label { class: "input-label",
                    "{label}"
                    if props.required {
                        span { class: "required-mark", " *" }
                    }
                }
            }

            // Input container
            div { class: "input-container",
                // Left icon
                if let Some(icon) = &props.icon {
                    span { class: "input-icon left", {icon} }
                }

                // Input element
                input {
                    class: "{input_class}",
                    r#type: "{props.input_type.as_str()}",
                    value: "{props.value}",
                    placeholder: props.placeholder.as_deref().unwrap_or(""),
                    disabled: props.disabled,
                    readonly: props.readonly,
                    required: props.required,
                    oninput: move |evt| {
                        if let Some(handler) = &props.oninput {
                            handler.call(evt.value());
                        }
                    },
                    onchange: move |evt| {
                        if let Some(handler) = &props.onchange {
                            handler.call(evt.value());
                        }
                    },
                    onfocus: move |evt| {
                        if let Some(handler) = &props.onfocus {
                            handler.call(evt);
                        }
                    },
                    onblur: move |evt| {
                        if let Some(handler) = &props.onblur {
                            handler.call(evt);
                        }
                    },
                }

                // Clear button
                if props.clearable && !props.value.is_empty() {
                    button {
                        class: "input-clear",
                        r#type: "button",
                        onclick: move |_| {
                            if let Some(handler) = &props.onchange {
                                handler.call(String::new());
                            }
                        },
                        "×"
                    }
                }

                // Right suffix
                if let Some(suffix) = &props.suffix {
                    span { class: "input-suffix", {suffix} }
                }
            }

            // Helper or error text
            if let Some(error) = &props.error {
                p { class: "input-error", "{error}" }
            } else if let Some(helper) = &props.helper {
                p { class: "input-helper", "{helper}" }
            }
        }
    }
}

/// Textarea component props
#[derive(Props, Clone, PartialEq)]
pub struct TextareaProps {
    /// Textarea value
    #[props(default)]
    pub value: String,

    /// Placeholder text
    #[props(default)]
    pub placeholder: Option<String>,

    /// Label text
    #[props(default)]
    pub label: Option<String>,

    /// Error message
    #[props(default)]
    pub error: Option<String>,

    /// Number of rows
    #[props(default = 4)]
    pub rows: u32,

    /// Disabled state
    #[props(default = false)]
    pub disabled: bool,

    /// Auto-resize
    #[props(default = false)]
    pub auto_resize: bool,

    /// Change handler
    #[props(default)]
    pub onchange: Option<EventHandler<String>>,
}

#[component]
pub fn Textarea(props: TextareaProps) -> Element {
    let has_error = props.error.is_some();

    rsx! {
        div { class: "rust-ui-textarea-wrapper",
            if let Some(label) = &props.label {
                label { class: "input-label", "{label}" }
            }

            textarea {
                class: "rust-ui-textarea",
                class: if has_error { "has-error" } else { "" },
                value: "{props.value}",
                placeholder: props.placeholder.as_deref().unwrap_or(""),
                rows: "{props.rows}",
                disabled: props.disabled,
                oninput: move |evt| {
                    if let Some(handler) = &props.onchange {
                        handler.call(evt.value());
                    }
                },
            }

            if let Some(error) = &props.error {
                p { class: "input-error", "{error}" }
            }
        }
    }
}
