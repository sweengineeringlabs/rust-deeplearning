//! # RustML Hub
//!
//! HuggingFace Hub integration for loading pre-trained models.
//!
//! This crate provides functionality to:
//! - Download models from HuggingFace Hub
//! - Load SafeTensors format weights
//! - Map weights to RustML model architectures
//!
//! ## Example
//!
//! ```rust,ignore
//! use rustml_hub::HubApi;
//!
//! let api = HubApi::new();
//! let bundle = api.download_model("openai-community/gpt2").await?;
//! let weights = bundle.load_tensors()?;
//! ```

pub mod api;
pub(crate) mod core;
mod saf;

pub use saf::*;
