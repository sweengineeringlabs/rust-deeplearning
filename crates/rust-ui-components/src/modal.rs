//! Modal component

use dioxus::prelude::*;
use crate::Size;

/// Modal component props
#[derive(Props, Clone, PartialEq)]
pub struct ModalProps {
    /// Modal content
    children: Element,

    /// Open state
    pub open: Signal<bool>,

    /// Modal title
    #[props(default)]
    pub title: Option<String>,

    /// Modal size
    #[props(default)]
    pub size: Size,

    /// Close on backdrop click
    #[props(default = true)]
    pub close_on_backdrop: bool,

    /// Close on escape key
    #[props(default = true)]
    pub close_on_escape: bool,

    /// Show close button
    #[props(default = true)]
    pub show_close: bool,

    /// Close handler
    #[props(default)]
    pub onclose: Option<EventHandler<()>>,
}

/// Modal component
///
/// # Example
/// ```rust
/// let show_modal = use_signal(|| false);
///
/// rsx! {
///     Button { onclick: move |_| show_modal.set(true), "Open Modal" }
///
///     Modal {
///         open: show_modal,
///         title: "Confirm Action",
///
///         p { "Are you sure?" }
///
///         ModalFooter {
///             Button { onclick: move |_| show_modal.set(false), "Cancel" }
///             Button { variant: Variant::Primary, "Confirm" }
///         }
///     }
/// }
/// ```
#[component]
pub fn Modal(props: ModalProps) -> Element {
    let is_open = *props.open.read();

    if !is_open {
        return rsx! {};
    }

    let close = move || {
        props.open.set(false);
        if let Some(handler) = &props.onclose {
            handler.call(());
        }
    };

    let backdrop_click = move |_| {
        if props.close_on_backdrop {
            close();
        }
    };

    let modal_size_class = match props.size {
        Size::Xs => "modal-xs",
        Size::Sm => "modal-sm",
        Size::Md => "modal-md",
        Size::Lg => "modal-lg",
        Size::Xl => "modal-xl",
    };

    rsx! {
        div {
            class: "rust-ui-modal-backdrop",
            onclick: backdrop_click,

            div {
                class: "rust-ui-modal {modal_size_class}",
                onclick: move |evt| evt.stop_propagation(),

                // Header
                if props.title.is_some() || props.show_close {
                    div { class: "modal-header",
                        if let Some(title) = &props.title {
                            h2 { class: "modal-title", "{title}" }
                        }
                        if props.show_close {
                            button {
                                class: "modal-close",
                                onclick: move |_| close(),
                                "×"
                            }
                        }
                    }
                }

                // Body
                div { class: "modal-body",
                    {props.children}
                }
            }
        }
    }
}

/// Modal footer section
#[derive(Props, Clone, PartialEq)]
pub struct ModalFooterProps {
    children: Element,
}

#[component]
pub fn ModalFooter(props: ModalFooterProps) -> Element {
    rsx! {
        div { class: "rust-ui-modal-footer", {props.children} }
    }
}
