//! # RustML Core
//!
//! Core tensor operations for the RustML machine learning library.
//!
//! This crate provides a `Tensor` type with operations needed for neural network
//! computations, including support for GPT-2 style transformer models.
//!
//! ## Features
//!
//! - Multi-dimensional tensor operations
//! - Automatic shape inference
//! - Broadcasting support
//! - GPU-ready design (future)
//!
//! ## Example
//!
//! ```rust
//! use rustml_core::Tensor;
//!
//! let a = Tensor::randn([2, 3]);
//! let b = Tensor::randn([3, 4]);
//! let c = a.matmul(&b).unwrap();
//! assert_eq!(c.shape(), &[2, 4]);
//! ```

pub mod api;
mod core;
mod saf;

pub use saf::*;
