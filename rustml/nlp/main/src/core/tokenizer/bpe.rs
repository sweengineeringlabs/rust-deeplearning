//! BPE (Byte Pair Encoding) Tokenizer for GPT-2

use crate::api::error::{NlpError, NlpResult};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// GPT-2 BPE Tokenizer
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    encoder: HashMap<String, u32>,
    decoder: HashMap<u32, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    byte_encoder: HashMap<u8, char>,
    byte_decoder: HashMap<char, u8>,
    pat: regex::Regex,
    eos_token_id: u32,
}

impl BpeTokenizer {
    /// GPT-2 EOS token ID
    pub const GPT2_EOS_TOKEN_ID: u32 = 50256;

    /// Create a tokenizer from vocab.json and merges.txt files
    pub fn from_files(vocab_path: impl AsRef<Path>, merges_path: impl AsRef<Path>) -> NlpResult<Self> {
        let vocab_content = fs::read_to_string(vocab_path)?;
        let merges_content = fs::read_to_string(merges_path)?;
        Self::from_strings(&vocab_content, &merges_content)
    }

    /// Create a tokenizer from vocab and merges content strings
    pub fn from_strings(vocab_json: &str, merges_txt: &str) -> NlpResult<Self> {
        let encoder: HashMap<String, u32> = serde_json::from_str(vocab_json)
            .map_err(|e| NlpError::TokenizerError(format!("Failed to parse vocab: {}", e)))?;

        let decoder: HashMap<u32, String> = encoder.iter().map(|(k, v)| (*v, k.clone())).collect();
        let bpe_ranks = Self::parse_merges(merges_txt)?;
        let (byte_encoder, byte_decoder) = Self::bytes_to_unicode();

        // GPT-2 tokenization pattern (simplified - no lookahead)
        // Original: r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        // We use a simplified version that captures the main tokenization patterns
        let pat = regex::Regex::new(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+"
        ).map_err(|e| NlpError::TokenizerError(format!("Regex error: {}", e)))?;

        // EOS token: the token at position 50256. We look it up or default.
        let eos_token_id = Self::GPT2_EOS_TOKEN_ID;

        Ok(Self {
            encoder,
            decoder,
            bpe_ranks,
            byte_encoder,
            byte_decoder,
            pat,
            eos_token_id,
        })
    }

    /// Get the EOS token ID
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();

        for mat in self.pat.find_iter(text) {
            let token_str = mat.as_str();
            // Convert bytes to unicode representation
            let unicode_str: String = token_str
                .bytes()
                .filter_map(|b| self.byte_encoder.get(&b).copied())
                .collect();

            // Apply BPE
            let bpe_tokens = self.bpe(&unicode_str);

            // Convert to token IDs
            for bpe_token in bpe_tokens {
                if let Some(&id) = self.encoder.get(&bpe_token) {
                    tokens.push(id);
                }
            }
        }

        tokens
    }

    /// Decode token IDs to text
    pub fn decode(&self, tokens: &[u32]) -> String {
        let text: String = tokens
            .iter()
            .filter_map(|&id| self.decoder.get(&id))
            .flat_map(|s| s.chars())
            .filter_map(|c| self.byte_decoder.get(&c).copied())
            .map(|b| b as char)
            .collect();
        text
    }

    /// Apply BPE to a word
    fn bpe(&self, token: &str) -> Vec<String> {
        if token.is_empty() {
            return vec![];
        }

        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();

        if word.len() == 1 {
            return word;
        }

        loop {
            // Find the pair with lowest rank
            let mut min_pair: Option<(String, String)> = None;
            let mut min_rank = usize::MAX;

            for i in 0..word.len() - 1 {
                let pair = (word[i].clone(), word[i + 1].clone());
                if let Some(&rank) = self.bpe_ranks.get(&pair) {
                    if rank < min_rank {
                        min_rank = rank;
                        min_pair = Some(pair);
                    }
                }
            }

            // If no pair found, we're done
            let Some((first, second)) = min_pair else {
                break;
            };

            // Merge the pair
            let merged = format!("{}{}", first, second);
            let mut new_word = Vec::new();
            let mut i = 0;

            while i < word.len() {
                if i < word.len() - 1 && word[i] == first && word[i + 1] == second {
                    new_word.push(merged.clone());
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }

            word = new_word;

            if word.len() == 1 {
                break;
            }
        }

        word
    }

    /// Parse merges.txt file
    fn parse_merges(content: &str) -> NlpResult<HashMap<(String, String), usize>> {
        let mut ranks = HashMap::new();

        for (rank, line) in content.lines().enumerate() {
            // Skip header line if present
            if line.starts_with("#version") {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                ranks.insert((parts[0].to_string(), parts[1].to_string()), rank);
            }
        }

        Ok(ranks)
    }

    /// Create byte to unicode mapping (GPT-2 style)
    fn bytes_to_unicode() -> (HashMap<u8, char>, HashMap<char, u8>) {
        let mut bs: Vec<u8> = Vec::new();

        // Printable ASCII characters
        for b in b'!'..=b'~' {
            bs.push(b);
        }
        // Latin-1 supplement (printable)
        for b in 0xA1u8..=0xACu8 {
            bs.push(b);
        }
        for b in 0xAEu8..=0xFFu8 {
            bs.push(b);
        }

        let mut cs: Vec<char> = bs.iter().map(|&b| b as char).collect();
        let mut n = 0u32;

        for b in 0u8..=255u8 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(char::from_u32(256 + n).unwrap_or('?'));
                n += 1;
            }
        }

        let byte_encoder: HashMap<u8, char> = bs.iter().copied().zip(cs.iter().copied()).collect();
        let byte_decoder: HashMap<char, u8> = cs.iter().copied().zip(bs.iter().copied()).collect();

        (byte_encoder, byte_decoder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_unicode() {
        let (encoder, decoder) = BpeTokenizer::bytes_to_unicode();
        assert_eq!(encoder.len(), 256);
        assert_eq!(decoder.len(), 256);

        // Check that printable ASCII maps to itself
        assert_eq!(encoder.get(&b'a'), Some(&'a'));
        assert_eq!(encoder.get(&b'Z'), Some(&'Z'));
    }

    #[test]
    fn test_bpe_simple() {
        // Create a minimal tokenizer for testing
        let vocab = r#"{"hello": 0, "world": 1, "Ġ": 2}"#;
        let merges = "";

        let tokenizer = BpeTokenizer::from_strings(vocab, merges).unwrap();
        assert_eq!(tokenizer.vocab_size(), 3);
    }
}
