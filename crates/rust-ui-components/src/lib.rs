//! rust-ui-components
//!
//! A comprehensive UI component library built on Dioxus.

pub mod button;
pub mod input;
pub mod select;
pub mod card;
pub mod modal;
pub mod badge;
pub mod spinner;
pub mod alert;

pub mod prelude {
    pub use crate::button::*;
    pub use crate::input::*;
    pub use crate::select::*;
    pub use crate::card::*;
    pub use crate::modal::*;
    pub use crate::badge::*;
    pub use crate::spinner::*;
    pub use crate::alert::*;

    pub use crate::{Variant, Size};
}

/// Component variants for styling
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum Variant {
    #[default]
    Default,
    Primary,
    Secondary,
    Success,
    Warning,
    Danger,
    Ghost,
    Link,
}

impl Variant {
    pub fn class(&self) -> &'static str {
        match self {
            Variant::Default => "variant-default",
            Variant::Primary => "variant-primary",
            Variant::Secondary => "variant-secondary",
            Variant::Success => "variant-success",
            Variant::Warning => "variant-warning",
            Variant::Danger => "variant-danger",
            Variant::Ghost => "variant-ghost",
            Variant::Link => "variant-link",
        }
    }
}

/// Component sizes
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum Size {
    Xs,
    Sm,
    #[default]
    Md,
    Lg,
    Xl,
}

impl Size {
    pub fn class(&self) -> &'static str {
        match self {
            Size::Xs => "size-xs",
            Size::Sm => "size-sm",
            Size::Md => "size-md",
            Size::Lg => "size-lg",
            Size::Xl => "size-xl",
        }
    }
}
