//! HuggingFace Tokenizer wrapper

use crate::api::error::{TokenizerError, TokenizerResult};
use crate::spi::contract::Tokenizer;

/// HuggingFace Tokenizer wrapper (uses the `tokenizers` crate).
///
/// Supports all tokenizer formats loadable by HuggingFace: BPE, SentencePiece,
/// WordPiece, etc. Load from a `tokenizer.json` file.
pub struct HFTokenizer {
    inner: tokenizers::Tokenizer,
}

impl HFTokenizer {
    /// Load from a `tokenizer.json` file.
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> TokenizerResult<Self> {
        let p = path.as_ref();
        let tokenizer = tokenizers::Tokenizer::from_file(p).map_err(|e| {
            TokenizerError::TokenizerError(format!(
                "Failed to load tokenizer file: {}: {}",
                p.display(),
                e
            ))
        })?;
        Ok(Self { inner: tokenizer })
    }
}

impl Tokenizer for HFTokenizer {
    fn encode(&self, text: &str) -> TokenizerResult<Vec<u32>> {
        let encoding = self.inner.encode(text, false).map_err(|e| {
            TokenizerError::TokenizerError(format!("Tokenizer encode failed: {}", e))
        })?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, tokens: &[u32]) -> TokenizerResult<String> {
        self.inner.decode(tokens, true).map_err(|e| {
            TokenizerError::TokenizerError(format!("Tokenizer decode failed: {}", e))
        })
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}
