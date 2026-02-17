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

/// GGUF token type flags (from `tokenizer.ggml.token_type`).
const TOKEN_TYPE_USER_DEFINED: i32 = 4;

/// SentencePiece BPE tokenizer built from GGUF vocabulary and merge scores.
///
/// Implements proper BPE encoding using the token scores stored in GGUF metadata.
/// Handles the SentencePiece "▁" space marker convention and USER_DEFINED tokens
/// (matched greedily on the original text before BPE).
pub struct GgufTokenizer {
    id_to_token: Vec<String>,
    token_to_id: std::collections::HashMap<String, u32>,
    scores: Vec<f32>,
    add_space_prefix: bool,
    /// USER_DEFINED tokens sorted by length descending for greedy matching.
    user_defined: Vec<(String, u32)>,
}

impl GgufTokenizer {
    /// Build from GGUF metadata: tokens, scores, and config.
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

        // Load BPE merge scores
        let mut scores = vec![f32::NEG_INFINITY; id_to_token.len()];
        if let Some(rustml_gguf::GGUFValue::Array(scores_arr)) =
            gguf.metadata.get("tokenizer.ggml.scores")
        {
            for (i, val) in scores_arr.iter().enumerate() {
                if i < scores.len() {
                    scores[i] = match val {
                        rustml_gguf::GGUFValue::F32(f) => *f,
                        _ => f32::NEG_INFINITY,
                    };
                }
            }
        }

        // Load token types and collect USER_DEFINED tokens
        let mut user_defined = Vec::new();
        if let Some(rustml_gguf::GGUFValue::Array(types_arr)) =
            gguf.metadata.get("tokenizer.ggml.token_type")
        {
            for (i, val) in types_arr.iter().enumerate() {
                let ttype = match val {
                    rustml_gguf::GGUFValue::I32(v) => *v,
                    rustml_gguf::GGUFValue::U32(v) => *v as i32,
                    _ => 0,
                };
                if ttype == TOKEN_TYPE_USER_DEFINED && i < id_to_token.len() {
                    let tok = &id_to_token[i];
                    // Only collect non-empty tokens with len >= 2 (single chars
                    // are already handled by initial_tokens)
                    if tok.len() >= 2 {
                        user_defined.push((tok.clone(), i as u32));
                    }
                }
            }
        }
        // Sort by length descending for greedy longest-match
        user_defined.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        // Check add_space_prefix setting
        let add_space_prefix = gguf
            .metadata
            .get("tokenizer.ggml.add_space_prefix")
            .and_then(|v| match v {
                rustml_gguf::GGUFValue::Bool(b) => Some(*b),
                _ => None,
            })
            .unwrap_or(true);

        Ok(Self {
            id_to_token,
            token_to_id,
            scores,
            add_space_prefix,
            user_defined,
        })
    }

    /// Match USER_DEFINED tokens on raw text before normalization.
    /// Returns a list of segments: either a matched token ID or a text slice
    /// that needs further BPE processing.
    fn split_user_defined<'a>(&self, text: &'a str) -> Vec<UserDefinedSegment<'a>> {
        if self.user_defined.is_empty() {
            return vec![UserDefinedSegment::Text(text)];
        }

        let mut segments = Vec::new();
        let mut pos = 0;
        let bytes = text.as_bytes();

        while pos < bytes.len() {
            let mut matched = false;
            for (tok_str, tok_id) in &self.user_defined {
                let tok_bytes = tok_str.as_bytes();
                if pos + tok_bytes.len() <= bytes.len()
                    && &bytes[pos..pos + tok_bytes.len()] == tok_bytes
                {
                    // Flush pending text
                    if let Some(UserDefinedSegment::Text(prev)) = segments.last() {
                        if prev.is_empty() {
                            segments.pop();
                        }
                    }
                    segments.push(UserDefinedSegment::Token(*tok_id));
                    pos += tok_bytes.len();
                    matched = true;
                    break;
                }
            }
            if !matched {
                // Extend the current text segment or start a new one
                match segments.last_mut() {
                    Some(UserDefinedSegment::Text(_)) => {
                        // We need to rebuild the text segment since we can't extend a &str
                        // Instead, track the start position
                    }
                    _ => {}
                }
                // Find the end of this text run (until next USER_DEFINED match or end)
                let start = pos;
                pos += 1;
                // Advance to next char boundary
                while pos < bytes.len() && !text.is_char_boundary(pos) {
                    pos += 1;
                }
                // Try to extend with more unmatched characters
                'outer: while pos < bytes.len() {
                    for (tok_str, _) in &self.user_defined {
                        let tok_bytes = tok_str.as_bytes();
                        if pos + tok_bytes.len() <= bytes.len()
                            && &bytes[pos..pos + tok_bytes.len()] == tok_bytes
                        {
                            break 'outer;
                        }
                    }
                    pos += 1;
                    while pos < bytes.len() && !text.is_char_boundary(pos) {
                        pos += 1;
                    }
                }
                segments.push(UserDefinedSegment::Text(&text[start..pos]));
            }
        }
        segments
    }

    /// Normalize a text segment: replace spaces with ▁ (U+2581).
    fn normalize_segment(&self, text: &str, is_first: bool) -> String {
        if is_first && self.add_space_prefix {
            format!("\u{2581}{}", text.replace(' ', "\u{2581}"))
        } else {
            text.replace(' ', "\u{2581}")
        }
    }

    /// Split text into initial character/byte tokens for BPE.
    fn initial_tokens(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        for ch in text.chars() {
            let ch_str = ch.to_string();
            if let Some(&id) = self.token_to_id.get(&ch_str) {
                tokens.push(id);
            } else {
                // Byte fallback: encode each UTF-8 byte as <0xHH>
                for byte in ch_str.as_bytes() {
                    let byte_token = format!("<0x{:02X}>", byte);
                    if let Some(&id) = self.token_to_id.get(&byte_token) {
                        tokens.push(id);
                    } else {
                        tokens.push(3); // <unk>
                    }
                }
            }
        }
        tokens
    }

    /// Apply BPE merges using scores: repeatedly merge the pair with highest score.
    fn bpe_merge(&self, tokens: &mut Vec<u32>) {
        loop {
            if tokens.len() < 2 {
                break;
            }

            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = usize::MAX;
            let mut best_id = 0u32;

            for i in 0..tokens.len() - 1 {
                let merged_str = format!(
                    "{}{}",
                    &self.id_to_token[tokens[i] as usize],
                    &self.id_to_token[tokens[i + 1] as usize]
                );
                if let Some(&merged_id) = self.token_to_id.get(&merged_str) {
                    let score = self.scores[merged_id as usize];
                    if score > best_score {
                        best_score = score;
                        best_idx = i;
                        best_id = merged_id;
                    }
                }
            }

            if best_idx == usize::MAX {
                break;
            }

            tokens[best_idx] = best_id;
            tokens.remove(best_idx + 1);
        }
    }
}

/// Segment produced by USER_DEFINED token matching.
enum UserDefinedSegment<'a> {
    /// A matched USER_DEFINED token ID.
    Token(u32),
    /// A text segment that needs normalization + BPE.
    Text(&'a str),
}

impl Tokenizer for GgufTokenizer {
    fn encode(&self, text: &str) -> NlpResult<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // Step 1: Match USER_DEFINED tokens on the original text (greedy longest-match).
        // These tokens (e.g. multi-space "   ", newline runs, HTML tags) are matched
        // before SentencePiece normalization because GGUF stores them with literal
        // spaces while normalization converts spaces to ▁.
        let segments = self.split_user_defined(text);

        // Step 2: For each segment, either emit the matched token ID or
        // normalize + BPE the text segment.
        let mut result = Vec::new();
        let mut is_first = true;
        for seg in &segments {
            match seg {
                UserDefinedSegment::Token(id) => {
                    result.push(*id);
                    is_first = false;
                }
                UserDefinedSegment::Text(s) => {
                    if s.is_empty() {
                        continue;
                    }
                    let normalized = self.normalize_segment(s, is_first);
                    let mut tokens = self.initial_tokens(&normalized);
                    self.bpe_merge(&mut tokens);
                    result.extend_from_slice(&tokens);
                    is_first = false;
                }
            }
        }

        Ok(result)
    }

    fn decode(&self, tokens: &[u32]) -> NlpResult<String> {
        let mut bytes = Vec::new();
        for &id in tokens {
            if (id as usize) >= self.id_to_token.len() {
                continue;
            }
            let token = &self.id_to_token[id as usize];

            // Check for byte tokens: <0xHH>
            if token.len() == 6
                && token.starts_with("<0x")
                && token.ends_with('>')
            {
                if let Ok(byte_val) = u8::from_str_radix(&token[3..5], 16) {
                    bytes.push(byte_val);
                    continue;
                }
            }

            // Replace SentencePiece space marker with actual space
            let text = token.replace('\u{2581}', " ");
            bytes.extend_from_slice(text.as_bytes());
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
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
