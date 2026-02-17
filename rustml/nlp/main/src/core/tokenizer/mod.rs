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

/// Simple lookup tokenizer built from GGUF vocabulary.
///
/// Provides basic encode (greedy longest-match) and decode (token lookup).
/// For production use, prefer HFTokenizer with a proper tokenizer.json.
pub struct GgufTokenizer {
    id_to_token: Vec<String>,
    token_to_id: std::collections::HashMap<String, u32>,
}

impl GgufTokenizer {
    /// Build from GGUF metadata tokens array.
    pub fn from_gguf(gguf: &rustml_gguf::GGUFFile) -> NlpResult<Self> {
        let tokens_val = gguf
            .metadata
            .get("tokenizer.ggml.tokens")
            .ok_or_else(|| {
                NlpError::TokenizerError("GGUF missing tokenizer.ggml.tokens".to_string())
            })?;

        let tokens_arr = match tokens_val {
            rustml_gguf::GGUFValue::Array(arr) => arr,
            _ => {
                return Err(NlpError::TokenizerError(
                    "tokenizer.ggml.tokens is not an array".to_string(),
                ))
            }
        };

        let mut id_to_token = Vec::with_capacity(tokens_arr.len());
        let mut token_to_id = std::collections::HashMap::with_capacity(tokens_arr.len());

        for (i, val) in tokens_arr.iter().enumerate() {
            let s = match val {
                rustml_gguf::GGUFValue::String(s) => s.clone(),
                _ => format!("<unk_{}>", i),
            };
            token_to_id.entry(s.clone()).or_insert(i as u32);
            id_to_token.push(s);
        }

        Ok(Self {
            id_to_token,
            token_to_id,
        })
    }
}

impl Tokenizer for GgufTokenizer {
    fn encode(&self, text: &str) -> NlpResult<Vec<u32>> {
        // Greedy longest-match tokenization (not optimal, but functional)
        let mut tokens = Vec::new();
        let bytes = text.as_bytes();
        let mut pos = 0;

        while pos < bytes.len() {
            let mut best_len = 0;
            let mut best_id = None;

            // Try matching from longest to shortest
            let max_len = std::cmp::min(64, bytes.len() - pos);
            for len in (1..=max_len).rev() {
                if let Ok(substr) = std::str::from_utf8(&bytes[pos..pos + len]) {
                    // SentencePiece uses ▁ (U+2581) for space
                    let sp_token = if pos == 0 || bytes[pos - 1] == b' ' {
                        format!("\u{2581}{}", substr.trim_start_matches(' '))
                    } else {
                        substr.to_string()
                    };

                    if let Some(&id) = self.token_to_id.get(&sp_token) {
                        best_len = len;
                        best_id = Some(id);
                        break;
                    }
                    if let Some(&id) = self.token_to_id.get(substr) {
                        best_len = len;
                        best_id = Some(id);
                        break;
                    }
                }
            }

            if let Some(id) = best_id {
                tokens.push(id);
                pos += best_len;
            } else {
                // Fallback: encode as byte token
                let byte_token = format!("<0x{:02X}>", bytes[pos]);
                if let Some(&id) = self.token_to_id.get(&byte_token) {
                    tokens.push(id);
                } else {
                    tokens.push(3); // <unk>
                }
                pos += 1;
            }
        }

        Ok(tokens)
    }

    fn decode(&self, tokens: &[u32]) -> NlpResult<String> {
        let mut result = String::new();
        for &id in tokens {
            if (id as usize) < self.id_to_token.len() {
                let token = &self.id_to_token[id as usize];
                // Replace SentencePiece space marker with actual space
                result.push_str(&token.replace('\u{2581}', " "));
            }
        }
        Ok(result)
    }

    fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
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
