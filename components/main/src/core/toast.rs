//! Toast/notification component

use dioxus::prelude::*;
use crate::api::types::Variant;

/// Toast position
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum ToastPosition {
    TopLeft,
    TopCenter,
    #[default]
    TopRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

impl ToastPosition {
    pub fn class(&self) -> &'static str {
        match self {
            ToastPosition::TopLeft => "top-left",
            ToastPosition::TopCenter => "top-center",
            ToastPosition::TopRight => "top-right",
            ToastPosition::BottomLeft => "bottom-left",
            ToastPosition::BottomCenter => "bottom-center",
            ToastPosition::BottomRight => "bottom-right",
        }
    }
}

/// Toast data
#[derive(Clone, PartialEq)]
pub struct ToastData {
    pub id: String,
    pub message: String,
    pub variant: Variant,
    pub duration_ms: Option<u64>,
    pub dismissable: bool,
}

impl ToastData {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            message: message.into(),
            variant: Variant::Default,
            duration_ms: Some(5000),
            dismissable: true,
        }
    }

    pub fn success(message: impl Into<String>) -> Self {
        Self::new(message).variant(Variant::Success)
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self::new(message).variant(Variant::Danger)
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(message).variant(Variant::Warning)
    }

    pub fn info(message: impl Into<String>) -> Self {
        Self::new(message).variant(Variant::Primary)
    }

    pub fn variant(mut self, variant: Variant) -> Self {
        self.variant = variant;
        self
    }

    pub fn duration(mut self, ms: u64) -> Self {
        self.duration_ms = Some(ms);
        self
    }

    pub fn persistent(mut self) -> Self {
        self.duration_ms = None;
        self
    }
}

/// Toast container props
#[derive(Props, Clone, PartialEq)]
pub struct ToastContainerProps {
    /// Toasts to display
    pub toasts: Vec<ToastData>,

    /// Position
    #[props(default)]
    pub position: ToastPosition,

    /// Dismiss handler
    #[props(default)]
    pub on_dismiss: Option<EventHandler<String>>,
}

/// Toast container component
#[component]
pub fn ToastContainer(props: ToastContainerProps) -> Element {
    rsx! {
        div {
            class: "rust-ui-toast-container {props.position.class()}",
            for toast in props.toasts.iter() {
                Toast {
                    key: "{toast.id}",
                    toast: toast.clone(),
                    on_dismiss: props.on_dismiss.clone(),
                }
            }
        }
    }
}

/// Single toast props
#[derive(Props, Clone, PartialEq)]
pub struct ToastProps {
    pub toast: ToastData,

    #[props(default)]
    pub on_dismiss: Option<EventHandler<String>>,
}

/// Single toast component
#[component]
pub fn Toast(props: ToastProps) -> Element {
    let variant_icon = match props.toast.variant {
        Variant::Success => "\u{2713}",
        Variant::Danger => "\u{2715}",
        Variant::Warning => "\u{26a0}",
        Variant::Primary => "\u{2139}",
        _ => "",
    };

    rsx! {
        div {
            class: "rust-ui-toast {props.toast.variant.class()}",
            role: "alert",

            if !variant_icon.is_empty() {
                span { class: "toast-icon", "{variant_icon}" }
            }

            span { class: "toast-message", "{props.toast.message}" }

            if props.toast.dismissable {
                button {
                    class: "toast-dismiss",
                    onclick: {
                        let id = props.toast.id.clone();
                        move |_| {
                            if let Some(handler) = &props.on_dismiss {
                                handler.call(id.clone());
                            }
                        }
                    },
                    "\u{00d7}"
                }
            }
        }
    }
}
