use std::collections::HashMap;
use std::path::{Path, PathBuf};

use llmforge::error::{LLMForgeError, Result};
use llmforge::loader::gguf::{GGUFFile, GGUFValue};
use llmforge::tokenization::Tokenizer;

/// Tokenizer built from vocabulary and BPE merges embedded in GGUF metadata.
///
/// Handles SentencePiece conventions: `▁` for word-initial space and
/// `<0xHH>` byte-fallback tokens.
pub struct GGUFTokenizer {
    /// token_id → token string
    vocab: Vec<String>,
    /// token string → token_id
    token_to_id: HashMap<String, u32>,
    /// BPE merge pair → rank (lower = higher priority)
    merge_ranks: HashMap<(String, String), usize>,
}

impl GGUFTokenizer {
    /// Build a tokenizer by parsing GGUF file header metadata.
    pub fn from_gguf_file<P: AsRef<Path>>(path: P) -> std::result::Result<Self, anyhow::Error> {
        let header = GGUFFile::parse_header(path)
            .map_err(|e| anyhow::anyhow!("Failed to parse GGUF header: {}", e))?;
        Self::from_metadata(&header.metadata)
    }

    /// Build a tokenizer from pre-parsed GGUF metadata.
    pub fn from_metadata(
        metadata: &HashMap<String, GGUFValue>,
    ) -> std::result::Result<Self, anyhow::Error> {
        let vocab = extract_string_array(metadata, "tokenizer.ggml.tokens")?;
        let merges =
            extract_string_array(metadata, "tokenizer.ggml.merges").unwrap_or_default();

        let token_to_id: HashMap<String, u32> = vocab
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();

        let merge_ranks: HashMap<(String, String), usize> = merges
            .iter()
            .enumerate()
            .filter_map(|(rank, m)| {
                let mut parts = m.splitn(2, ' ');
                let a = parts.next()?;
                let b = parts.next()?;
                Some(((a.to_string(), b.to_string()), rank))
            })
            .collect();

        eprintln!(
            "  GGUF tokenizer: {} tokens, {} merges",
            vocab.len(),
            merge_ranks.len()
        );

        Ok(Self {
            vocab,
            token_to_id,
            merge_ranks,
        })
    }
}

fn extract_string_array(
    metadata: &HashMap<String, GGUFValue>,
    key: &str,
) -> std::result::Result<Vec<String>, anyhow::Error> {
    match metadata.get(key) {
        Some(GGUFValue::Array(arr)) => arr
            .iter()
            .map(|v| {
                v.as_string()
                    .map(|s| s.to_string())
                    .ok_or_else(|| anyhow::anyhow!("Non-string value in {}", key))
            })
            .collect(),
        Some(_) => Err(anyhow::anyhow!("{} is not an array", key)),
        None => Err(anyhow::anyhow!("{} not found in metadata", key)),
    }
}

/// Check whether a GGUF file has embedded tokenizer metadata, and if so,
/// generate a `tokenizer.json` next to it (if one doesn't already exist).
///
/// Returns `Some(path)` when a tokenizer.json is available (pre-existing or
/// freshly generated), `None` when the GGUF lacks tokenizer metadata.
pub fn ensure_tokenizer_json(gguf_path: &Path) -> std::result::Result<Option<PathBuf>, anyhow::Error> {
    let parent = gguf_path.parent().unwrap_or(Path::new("."));
    let dest = parent.join("tokenizer.json");

    // Already exists — nothing to do
    if dest.is_file() {
        return Ok(Some(dest));
    }

    // Parse the GGUF header (reads only metadata, not tensor data)
    let header = GGUFFile::parse_header(gguf_path)
        .map_err(|e| anyhow::anyhow!("Failed to parse GGUF header: {}", e))?;
    let metadata = &header.metadata;

    let tokens = match extract_string_array(metadata, "tokenizer.ggml.tokens") {
        Ok(t) => t,
        Err(_) => return Ok(None), // no vocab in this GGUF
    };
    let merges = extract_string_array(metadata, "tokenizer.ggml.merges")
        .unwrap_or_default();

    // Special token IDs
    let unk_id = metadata
        .get("tokenizer.ggml.unknown_token_id")
        .and_then(|v| v.as_u32())
        .unwrap_or(0);
    let bos_id = metadata
        .get("tokenizer.ggml.bos_token_id")
        .and_then(|v| v.as_u32())
        .unwrap_or(1);
    let eos_id = metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| v.as_u32())
        .unwrap_or(2);

    // Build vocab map: { "token": id, ... }
    let vocab_map: serde_json::Map<String, serde_json::Value> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), serde_json::json!(i)))
        .collect();

    // Build added_tokens list for special tokens
    let special_ids = [unk_id, bos_id, eos_id];
    let added_tokens: Vec<serde_json::Value> = special_ids
        .iter()
        .copied()
        .filter(|&id| (id as usize) < tokens.len())
        .map(|id| {
            serde_json::json!({
                "id": id,
                "content": tokens[id as usize],
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            })
        })
        .collect();

    let unk_token = tokens
        .get(unk_id as usize)
        .cloned()
        .unwrap_or_else(|| "<unk>".to_string());

    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": {
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": "always",
            "split": true
        },
        "post_processor": null,
        "decoder": {
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": "always"
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": unk_token,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": true,
            "vocab": vocab_map,
            "merges": merges
        }
    });

    let json_bytes = serde_json::to_string_pretty(&tokenizer_json)?;
    std::fs::write(&dest, json_bytes)?;
    eprintln!("  Generated tokenizer.json ({} tokens, {} merges)", tokens.len(), merges.len());

    Ok(Some(dest))
}

impl Tokenizer for GGUFTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut all_tokens = Vec::new();

        // Pre-tokenize: split on spaces, prepend ▁ to each word (SentencePiece)
        for (i, word) in text.split(' ').enumerate() {
            if word.is_empty() && i > 0 {
                // Consecutive spaces: emit the ▁ token on its own
                if let Some(&id) = self.token_to_id.get("▁") {
                    all_tokens.push(id);
                }
                continue;
            }
            if word.is_empty() {
                continue;
            }

            let piece = format!("▁{}", word);

            // Split into individual characters
            let mut symbols: Vec<String> = piece.chars().map(|c| c.to_string()).collect();

            // BPE merge loop: repeatedly apply highest-priority (lowest rank) merge
            loop {
                if symbols.len() < 2 {
                    break;
                }

                let mut best_rank = usize::MAX;
                let mut best_pos = 0;

                for j in 0..symbols.len() - 1 {
                    let pair = (symbols[j].clone(), symbols[j + 1].clone());
                    if let Some(&rank) = self.merge_ranks.get(&pair) {
                        if rank < best_rank {
                            best_rank = rank;
                            best_pos = j;
                        }
                    }
                }

                if best_rank == usize::MAX {
                    break;
                }

                let merged = format!("{}{}", symbols[best_pos], symbols[best_pos + 1]);
                symbols.splice(best_pos..best_pos + 2, std::iter::once(merged));
            }

            // Map each BPE piece to its token ID
            for sym in &symbols {
                if let Some(&id) = self.token_to_id.get(sym.as_str()) {
                    all_tokens.push(id);
                } else {
                    // Byte fallback: encode unknown chars as <0xHH>
                    for byte in sym.as_bytes() {
                        let byte_token = format!("<0x{:02X}>", byte);
                        if let Some(&id) = self.token_to_id.get(&byte_token) {
                            all_tokens.push(id);
                        }
                    }
                }
            }
        }

        Ok(all_tokens)
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();

        for &token_id in tokens {
            let id = token_id as usize;
            if id >= self.vocab.len() {
                continue;
            }

            let token_str = &self.vocab[id];

            // Skip special tokens
            if token_str == "<s>" || token_str == "</s>" || token_str == "<unk>" {
                continue;
            }

            // Byte-fallback tokens: <0xHH>
            if token_str.starts_with("<0x") && token_str.ends_with('>') && token_str.len() == 6 {
                if let Ok(byte) = u8::from_str_radix(&token_str[3..5], 16) {
                    bytes.push(byte);
                    continue;
                }
            }

            // Regular token: ▁ represents a space
            let text = token_str.replace('▁', " ");
            bytes.extend_from_slice(text.as_bytes());
        }

        String::from_utf8(bytes).map_err(|e| {
            LLMForgeError::TokenizerError(format!("Invalid UTF-8 in decoded output: {}", e))
        })
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }
}
