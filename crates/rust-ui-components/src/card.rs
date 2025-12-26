//! Card component

use dioxus::prelude::*;

/// Card component props
#[derive(Props, Clone, PartialEq)]
pub struct CardProps {
    /// Card content
    children: Element,

    /// Additional CSS class
    #[props(default)]
    pub class: Option<String>,

    /// Click handler (makes card interactive)
    #[props(default)]
    pub onclick: Option<EventHandler<MouseEvent>>,
}

/// Card component
///
/// # Example
/// ```rust
/// rsx! {
///     Card {
///         CardHeader {
///             CardTitle { "Service Status" }
///             CardDescription { "Current status of all services" }
///         }
///         CardContent {
///             p { "Content here" }
///         }
///         CardFooter {
///             Button { "Action" }
///         }
///     }
/// }
/// ```
#[component]
pub fn Card(props: CardProps) -> Element {
    let class = format!(
        "rust-ui-card {} {}",
        if props.onclick.is_some() { "interactive" } else { "" },
        props.class.as_deref().unwrap_or(""),
    );

    rsx! {
        div {
            class: "{class}",
            onclick: move |evt| {
                if let Some(handler) = &props.onclick {
                    handler.call(evt);
                }
            },
            {props.children}
        }
    }
}

/// Card header section
#[derive(Props, Clone, PartialEq)]
pub struct CardHeaderProps {
    children: Element,
    #[props(default)]
    pub class: Option<String>,
}

#[component]
pub fn CardHeader(props: CardHeaderProps) -> Element {
    rsx! {
        div {
            class: "rust-ui-card-header {props.class.as_deref().unwrap_or(\"\")}",
            {props.children}
        }
    }
}

/// Card title
#[derive(Props, Clone, PartialEq)]
pub struct CardTitleProps {
    children: Element,
}

#[component]
pub fn CardTitle(props: CardTitleProps) -> Element {
    rsx! {
        h3 { class: "rust-ui-card-title", {props.children} }
    }
}

/// Card description
#[derive(Props, Clone, PartialEq)]
pub struct CardDescriptionProps {
    children: Element,
}

#[component]
pub fn CardDescription(props: CardDescriptionProps) -> Element {
    rsx! {
        p { class: "rust-ui-card-description", {props.children} }
    }
}

/// Card content section
#[derive(Props, Clone, PartialEq)]
pub struct CardContentProps {
    children: Element,
    #[props(default)]
    pub class: Option<String>,
}

#[component]
pub fn CardContent(props: CardContentProps) -> Element {
    rsx! {
        div {
            class: "rust-ui-card-content {props.class.as_deref().unwrap_or(\"\")}",
            {props.children}
        }
    }
}

/// Card footer section
#[derive(Props, Clone, PartialEq)]
pub struct CardFooterProps {
    children: Element,
    #[props(default)]
    pub class: Option<String>,
}

#[component]
pub fn CardFooter(props: CardFooterProps) -> Element {
    rsx! {
        div {
            class: "rust-ui-card-footer {props.class.as_deref().unwrap_or(\"\")}",
            {props.children}
        }
    }
}
