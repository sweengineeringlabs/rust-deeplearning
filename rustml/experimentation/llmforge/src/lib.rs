//! LLMForge — a from-scratch LLM inference engine in pure Rust.
//!
//! Provides tensor operations, transformer blocks, model loading (SafeTensors, GGUF),
//! tokenization (HuggingFace `tokenizers`), and text generation with sampling strategies
//! (top-k, nucleus, beam search, streaming).
//!
//! # Supported models
//!
//! - **GPT-2** — via SafeTensors (`ModelLoader::load_safetensors` + `LlmModel::from_pretrained_gpt2`)
//! - **Llama-2 / TinyLlama** — via GGUF (`ModelLoader::load_gguf` + `LlmModel::from_pretrained`)
//!
//! # Quantization
//!
//! Native support for Q4_0 and Q8_0 block quantization with SIMD-accelerated dot products
//! (AVX2, SSE2, NEON). K-quant types (Q2_K–Q8_K) are dequantized to F32 on load.
//!
//! # Quick start
//!
//! See `examples/gpt2_inference.rs` (auto-downloads GPT-2 124M) and
//! `examples/gguf_inference.rs` (auto-downloads TinyLlama-1.1B Q4_0).

pub mod error;
pub mod core;
pub mod nn;
pub mod attention;
pub mod transformer;
pub mod tokenization;
pub mod training;
pub mod inference;
pub mod distributed;
pub mod quantization;
pub mod models;
pub mod loader;
pub mod config;

pub use error::{LLMForgeError, Result};
pub use core::tensor::Tensor;
pub use config::RuntimeConfig;
