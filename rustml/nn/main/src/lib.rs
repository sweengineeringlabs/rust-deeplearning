//! # RustML Neural Network
//!
//! Neural network layers and modules for RustML.
//!
//! This crate provides building blocks for neural networks including:
//! - Linear layers
//! - Embedding layers
//! - Layer normalization
//! - Attention mechanisms (including causal self-attention for GPT)
//!
//! ## Example
//!
//! ```rust,ignore
//! use rustml_nn::{Linear, LayerNorm, CausalSelfAttention};
//! use rustml_core::Tensor;
//!
//! let linear = Linear::new(768, 768);
//! let output = linear.forward(&input)?;
//! ```

pub mod api;
mod core;
mod saf;

pub use saf::*;
