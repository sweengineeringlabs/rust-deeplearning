use crate::api::error::TokenizerResult;

/// Common tokenizer interface.
pub trait Tokenizer {
    /// Encode text to token IDs.
    fn encode(&self, text: &str) -> TokenizerResult<Vec<u32>>;
    /// Decode token IDs to text.
    fn decode(&self, tokens: &[u32]) -> TokenizerResult<String>;
    /// Vocabulary size.
    fn vocab_size(&self) -> usize;
    /// Look up a special token by name, returning its ID if present.
    fn token_to_id(&self, token: &str) -> Option<u32>;
}
