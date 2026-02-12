//! # RustML NLP
//!
//! Natural Language Processing models for RustML, including GPT-2.
//!
//! This crate provides:
//! - GPT-2 model implementation with support for all variants (small, medium, large, xl)
//! - Text generation with temperature, top-k, and top-p sampling
//! - BPE tokenizer for GPT-2
//!
//! ## Example
//!
//! ```rust,ignore
//! use rustml_nlp::{GptModel, GptConfig, TextGenerator, GenerationConfig};
//! use rustml_nlp::BpeTokenizer;
//! use rustml_hub::HubApi;
//!
//! // Load model
//! let api = HubApi::new();
//! let bundle = api.download_model("openai-community/gpt2").await?;
//! let weights = bundle.load_tensors()?;
//! let model = GptModel::from_hub_weights(GptConfig::gpt2_small(), weights)?;
//!
//! // Load tokenizer
//! let tokenizer = BpeTokenizer::from_files("vocab.json", "merges.txt")?;
//!
//! // Generate text
//! let generator = TextGenerator::new(&model);
//! let output = generator.generate(
//!     &tokenizer.encode("Hello world"),
//!     &GenerationConfig::default(),
//! )?;
//!
//! println!("{}", tokenizer.decode(&output));
//! ```

pub mod api;
pub(crate) mod core;
mod saf;

pub use saf::*;
