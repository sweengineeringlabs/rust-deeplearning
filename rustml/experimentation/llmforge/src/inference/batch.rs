use crate::error::Result;
use super::generator::Generator;

impl<'a> Generator<'a> {
    /// Generate completions for multiple prompts sequentially.
    /// Each prompt gets its own independent KVCache and generation run.
    pub fn generate_batch(&self, prompts: &[&str], max_tokens: usize) -> Result<Vec<String>> {
        prompts.iter()
            .map(|prompt| self.generate(prompt, max_tokens))
            .collect()
    }
}
