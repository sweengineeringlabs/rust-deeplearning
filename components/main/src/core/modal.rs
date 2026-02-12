//! Modal component

use dioxus::prelude::*;
use crate::api::types::Size;

/// Modal component props
#[derive(Props, Clone, PartialEq)]
pub struct ModalProps {
    /// Modal content
    children: Element,

    /// Open state
    pub open: bool,

    /// Modal title
    #[props(default)]
    pub title: Option<String>,

    /// Modal size
    #[props(default)]
    pub size: Size,

    /// Close handler
    #[props(default)]
    pub on_close: Option<EventHandler<()>>,
}

/// Modal component
#[component]
pub fn Modal(props: ModalProps) -> Element {
    if !props.open {
        return rsx! {};
    }

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
            onclick: move |_| {
                if let Some(handler) = &props.on_close {
                    handler.call(());
                }
            },

            div {
                class: "rust-ui-modal {modal_size_class}",
                onclick: move |evt| evt.stop_propagation(),

                if let Some(title) = &props.title {
                    div { class: "modal-header",
                        h2 { class: "modal-title", "{title}" }
                        button {
                            class: "modal-close",
                            onclick: move |_| {
                                if let Some(handler) = &props.on_close {
                                    handler.call(());
                                }
                            },
                            "\u{00d7}"
                        }
                    }
                }

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
