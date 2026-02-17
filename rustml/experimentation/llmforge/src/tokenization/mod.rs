use crate::error::{LLMForgeError, Result};

// Simple Tokenizer Interface
pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> Result<String>;
    fn vocab_size(&self) -> usize;
    /// Look up a token string in the vocabulary, returning its ID if present.
    /// Used to resolve special tokens (e.g. `<|system|>`, `</s>`) to their IDs.
    fn token_to_id(&self, token: &str) -> Option<u32>;
}

// Naive Tokenizer that operates on Unicode code points for testing
pub struct NaiveTokenizer {
    vocab_size: usize,
}

impl NaiveTokenizer {
    pub fn new() -> Self {
        Self { vocab_size: 256 }
    }
}

impl Tokenizer for NaiveTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut tokens = Vec::with_capacity(text.len());
        for ch in text.chars() {
            tokens.push(ch as u32);
        }
        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut s = String::with_capacity(tokens.len());
        for &t in tokens {
            match char::from_u32(t) {
                Some(ch) => s.push(ch),
                None => {
                    return Err(LLMForgeError::TokenizerError(
                        format!("Invalid Unicode code point: {}", t)
                    ));
                }
            }
        }
        Ok(s)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn token_to_id(&self, _token: &str) -> Option<u32> {
        None
    }
}

// Real HF Tokenizer wrapper
pub struct HFTokenizer {
    inner: tokenizers::Tokenizer,
}

impl HFTokenizer {
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let p = path.as_ref();
        let tokenizer = tokenizers::Tokenizer::from_file(p)
            .map_err(|e| LLMForgeError::TokenizerError(
                format!("Failed to load tokenizer file: {}: {}", p.display(), e)
            ))?;
        Ok(Self { inner: tokenizer })
    }
}

impl Tokenizer for HFTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, false)
            .map_err(|e| LLMForgeError::TokenizerError(
                format!("Tokenizer encode failed: {}", e)
            ))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner.decode(tokens, true)
            .map_err(|e| LLMForgeError::TokenizerError(
                format!("Tokenizer decode failed: {}", e)
            ))
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}
