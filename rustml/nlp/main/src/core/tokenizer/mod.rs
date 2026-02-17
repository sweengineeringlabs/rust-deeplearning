//! Tokenizer module
//!
//! Provides:
//! - `Tokenizer` trait for any tokenizer backend
//! - `BpeTokenizer` — GPT-2 BPE tokenizer
//! - `HFTokenizer` — wrapper around the HuggingFace `tokenizers` crate
//! - `ByteTokenizer` — naive byte-level tokenizer for testing

mod bpe;

pub use bpe::BpeTokenizer;

use crate::api::error::{NlpError, NlpResult};

/// Common tokenizer interface.
pub trait Tokenizer {
    /// Encode text to token IDs.
    fn encode(&self, text: &str) -> NlpResult<Vec<u32>>;
    /// Decode token IDs to text.
    fn decode(&self, tokens: &[u32]) -> NlpResult<String>;
    /// Vocabulary size.
    fn vocab_size(&self) -> usize;
    /// Look up a special token by name, returning its ID if present.
    fn token_to_id(&self, token: &str) -> Option<u32>;
}

/// Implement Tokenizer for BpeTokenizer (wraps its non-Result methods).
impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> NlpResult<Vec<u32>> {
        Ok(BpeTokenizer::encode(self, text))
    }
    fn decode(&self, tokens: &[u32]) -> NlpResult<String> {
        Ok(BpeTokenizer::decode(self, tokens))
    }
    fn vocab_size(&self) -> usize {
        BpeTokenizer::vocab_size(self)
    }
    fn token_to_id(&self, _token: &str) -> Option<u32> {
        None
    }
}

/// HuggingFace Tokenizer wrapper (uses the `tokenizers` crate).
///
/// Supports all tokenizer formats loadable by HuggingFace: BPE, SentencePiece,
/// WordPiece, etc. Load from a `tokenizer.json` file.
pub struct HFTokenizer {
    inner: tokenizers::Tokenizer,
}

impl HFTokenizer {
    /// Load from a `tokenizer.json` file.
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> NlpResult<Self> {
        let p = path.as_ref();
        let tokenizer = tokenizers::Tokenizer::from_file(p).map_err(|e| {
            NlpError::TokenizerError(format!(
                "Failed to load tokenizer file: {}: {}",
                p.display(),
                e
            ))
        })?;
        Ok(Self { inner: tokenizer })
    }
}

impl Tokenizer for HFTokenizer {
    fn encode(&self, text: &str) -> NlpResult<Vec<u32>> {
        let encoding = self.inner.encode(text, false).map_err(|e| {
            NlpError::TokenizerError(format!("Tokenizer encode failed: {}", e))
        })?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, tokens: &[u32]) -> NlpResult<String> {
        self.inner.decode(tokens, true).map_err(|e| {
            NlpError::TokenizerError(format!("Tokenizer decode failed: {}", e))
        })
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}

/// Naive byte-level tokenizer for testing. Maps each byte to its value as a token ID.
pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> NlpResult<Vec<u32>> {
        Ok(text.bytes().map(|b| b as u32).collect())
    }
    fn decode(&self, tokens: &[u32]) -> NlpResult<String> {
        let bytes: Vec<u8> = tokens
            .iter()
            .filter_map(|&t| if t < 256 { Some(t as u8) } else { None })
            .collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
    fn vocab_size(&self) -> usize {
        256
    }
    fn token_to_id(&self, _token: &str) -> Option<u32> {
        None
    }
}
