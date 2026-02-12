//! Tabs component

use dioxus::prelude::*;

/// Tab item
#[derive(Clone, PartialEq)]
pub struct Tab {
    pub id: String,
    pub label: String,
    pub icon: Option<Element>,
    pub disabled: bool,
}

impl Tab {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            icon: None,
            disabled: false,
        }
    }

    pub fn icon(mut self, icon: Element) -> Self {
        self.icon = Some(icon);
        self
    }

    pub fn disabled(mut self) -> Self {
        self.disabled = true;
        self
    }
}

/// Tab variant
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum TabVariant {
    #[default]
    Line,
    Pills,
    Boxed,
}

/// Tabs props
#[derive(Props, Clone, PartialEq)]
pub struct TabsProps {
    /// Tab definitions
    pub tabs: Vec<Tab>,

    /// Active tab ID
    pub active: String,

    /// Tab content
    children: Element,

    /// Variant
    #[props(default)]
    pub variant: TabVariant,

    /// Full width tabs
    #[props(default = false)]
    pub full_width: bool,

    /// Tab change handler
    #[props(default)]
    pub on_change: Option<EventHandler<String>>,
}

/// Tabs component
#[component]
pub fn Tabs(props: TabsProps) -> Element {
    let variant_class = match props.variant {
        TabVariant::Line => "tabs-line",
        TabVariant::Pills => "tabs-pills",
        TabVariant::Boxed => "tabs-boxed",
    };

    rsx! {
        div { class: "rust-ui-tabs",
            // Tab headers
            div {
                class: "tabs-list {variant_class}",
                class: if props.full_width { "full-width" } else { "" },
                role: "tablist",

                for tab in props.tabs.iter() {
                    button {
                        class: "tab-item",
                        class: if tab.id == props.active { "active" } else { "" },
                        class: if tab.disabled { "disabled" } else { "" },
                        role: "tab",
                        disabled: tab.disabled,
                        onclick: {
                            let id = tab.id.clone();
                            move |_| {
                                if let Some(handler) = &props.on_change {
                                    handler.call(id.clone());
                                }
                            }
                        },

                        if let Some(icon) = &tab.icon {
                            span { class: "tab-icon", {icon} }
                        }
                        span { class: "tab-label", "{tab.label}" }
                    }
                }
            }

            // Tab content
            div { class: "tabs-content", role: "tabpanel",
                {props.children}
            }
        }
    }
}

/// Tab panel (content for a single tab)
#[derive(Props, Clone, PartialEq)]
pub struct TabPanelProps {
    pub id: String,
    pub active: String,
    children: Element,
}

#[component]
pub fn TabPanel(props: TabPanelProps) -> Element {
    if props.id != props.active {
        return rsx! {};
    }

    rsx! {
        div { class: "tab-panel",
            {props.children}
        }
    }
}
