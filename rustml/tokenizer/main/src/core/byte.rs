//! Naive byte-level tokenizer for testing

use crate::api::error::TokenizerResult;
use crate::spi::contract::Tokenizer;

/// Naive byte-level tokenizer for testing. Maps each byte to its value as a token ID.
pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> TokenizerResult<Vec<u32>> {
        Ok(text.bytes().map(|b| b as u32).collect())
    }
    fn decode(&self, tokens: &[u32]) -> TokenizerResult<String> {
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
