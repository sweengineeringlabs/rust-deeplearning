//! AudioLearn - Audio-first Online Course Platform
//!
//! Architecture: SEA (Stratified Encapsulation Architecture)
//!
//! Layers:
//! - api: Public API contracts, domain models, types, errors
//! - spi: Service Provider Interfaces (traits for external services)
//! - core: Business logic implementations
//! - saf: Facade re-exports and platform-dispatch functions

pub mod api;
pub mod spi;
pub mod core;
mod saf;

pub use saf::*;
