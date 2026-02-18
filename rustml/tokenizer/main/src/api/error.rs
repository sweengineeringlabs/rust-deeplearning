pub type TokenizerResult<T> = Result<T, TokenizerError>;

#[derive(thiserror::Error, Debug)]
pub enum TokenizerError {
    #[error("Tokenizer error: {0}")]
    TokenizerError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
