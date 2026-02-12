//! Dropdown menu component

use dioxus::prelude::*;

/// Menu item type
#[derive(Clone, PartialEq)]
pub enum MenuItem {
    Item {
        id: String,
        label: String,
        icon: Option<Element>,
        disabled: bool,
    },
    Divider,
    Header(String),
}

impl MenuItem {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        MenuItem::Item {
            id: id.into(),
            label: label.into(),
            icon: None,
            disabled: false,
        }
    }

    pub fn icon(self, icon: Element) -> Self {
        match self {
            MenuItem::Item { id, label, disabled, .. } => {
                MenuItem::Item { id, label, icon: Some(icon), disabled }
            }
            other => other,
        }
    }

    pub fn disabled(self) -> Self {
        match self {
            MenuItem::Item { id, label, icon, .. } => {
                MenuItem::Item { id, label, icon, disabled: true }
            }
            other => other,
        }
    }

    pub fn divider() -> Self {
        MenuItem::Divider
    }

    pub fn header(text: impl Into<String>) -> Self {
        MenuItem::Header(text.into())
    }
}

/// Dropdown props
#[derive(Props, Clone, PartialEq)]
pub struct DropdownProps {
    /// Trigger element
    trigger: Element,

    /// Menu items
    pub items: Vec<MenuItem>,

    /// Align to right
    #[props(default = false)]
    pub align_right: bool,

    /// Item click handler
    #[props(default)]
    pub on_select: Option<EventHandler<String>>,
}

/// Dropdown component
#[component]
pub fn Dropdown(props: DropdownProps) -> Element {
    let mut open = use_signal(|| false);

    rsx! {
        div { class: "rust-ui-dropdown",
            // Trigger
            div {
                class: "dropdown-trigger",
                onclick: move |_| {
                    let current = *open.read();
                    open.set(!current);
                },
                {props.trigger}
            }

            // Menu
            if *open.read() {
                div {
                    class: "dropdown-menu",
                    class: if props.align_right { "align-right" } else { "" },

                    for item in props.items.iter() {
                        match item {
                            MenuItem::Item { id, label, icon, disabled } => {
                                rsx! {
                                    button {
                                        class: "dropdown-item",
                                        class: if *disabled { "disabled" } else { "" },
                                        disabled: *disabled,
                                        onclick: {
                                            let id = id.clone();
                                            move |_| {
                                                open.set(false);
                                                if let Some(handler) = &props.on_select {
                                                    handler.call(id.clone());
                                                }
                                            }
                                        },
                                        if let Some(icon) = icon {
                                            span { class: "item-icon", {icon} }
                                        }
                                        span { class: "item-label", "{label}" }
                                    }
                                }
                            }
                            MenuItem::Divider => {
                                rsx! { hr { class: "dropdown-divider" } }
                            }
                            MenuItem::Header(text) => {
                                rsx! {
                                    div { class: "dropdown-header", "{text}" }
                                }
                            }
                        }
                    }
                }

                // Click outside to close
                div {
                    class: "dropdown-backdrop",
                    onclick: move |_| open.set(false),
                }
            }
        }
    }
}
