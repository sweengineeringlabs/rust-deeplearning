use std::io::{self, BufRead, Read};
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};

use rustml_tokenizer::{BpeTokenizer, ByteTokenizer, GgufTokenizer, HFTokenizer, Tokenizer};

/// Tokenizer backend selection (exactly one required).
#[derive(Args)]
#[group(required = true, multiple = false)]
pub struct Backend {
    /// Load tokenizer from a GGUF model file.
    #[arg(long, value_name = "PATH")]
    gguf: Option<PathBuf>,

    /// Load tokenizer from a HuggingFace tokenizer.json file.
    #[arg(long, value_name = "PATH")]
    hf: Option<PathBuf>,

    /// Load GPT-2 BPE tokenizer from vocab.json and merges.txt files.
    #[arg(long, value_names = ["VOCAB", "MERGES"], num_args = 2)]
    bpe: Option<Vec<PathBuf>>,

    /// Use the trivial byte-level tokenizer (maps each byte to its value).
    #[arg(long)]
    byte: bool,
}

#[derive(Args)]
pub struct TokenizerArgs {
    #[command(flatten)]
    backend: Backend,

    #[command(subcommand)]
    command: TokenizerCommand,
}

#[derive(Subcommand)]
enum TokenizerCommand {
    /// Encode text into token IDs.
    Encode {
        /// Text to encode (reads from stdin if omitted and --file not given).
        text: Option<String>,

        /// Read input text from a file.
        #[arg(long, value_name = "PATH")]
        file: Option<PathBuf>,

        /// Output token IDs as a JSON array instead of space-separated.
        #[arg(long)]
        json: bool,
    },

    /// Decode token IDs back into text.
    Decode {
        /// Token IDs to decode (reads from stdin if omitted and --file not given).
        ids: Vec<u32>,

        /// Read token IDs from a file (one per line or space-separated).
        #[arg(long, value_name = "PATH")]
        file: Option<PathBuf>,
    },

    /// Display tokenizer info (vocab size, token lookups).
    Info {
        /// Look up specific tokens and print their IDs.
        #[arg(long, value_name = "TOKEN")]
        lookup: Vec<String>,
    },
}

fn load_backend(backend: &Backend) -> Result<Box<dyn Tokenizer>> {
    if let Some(path) = &backend.gguf {
        eprintln!("Loading GGUF tokenizer from {}", path.display());
        let gguf_file = rustml_gguf::GGUFFile::parse_header(path)
            .with_context(|| format!("Failed to parse GGUF file: {}", path.display()))?;
        let tok = GgufTokenizer::from_gguf(&gguf_file)
            .with_context(|| "Failed to build GGUF tokenizer")?;
        Ok(Box::new(tok))
    } else if let Some(path) = &backend.hf {
        eprintln!("Loading HuggingFace tokenizer from {}", path.display());
        let tok = HFTokenizer::from_file(path)
            .with_context(|| format!("Failed to load HF tokenizer: {}", path.display()))?;
        Ok(Box::new(tok))
    } else if let Some(paths) = &backend.bpe {
        let vocab = &paths[0];
        let merges = &paths[1];
        eprintln!(
            "Loading BPE tokenizer from {} and {}",
            vocab.display(),
            merges.display()
        );
        let tok = BpeTokenizer::from_files(vocab, merges)
            .with_context(|| "Failed to load BPE tokenizer")?;
        Ok(Box::new(tok))
    } else if backend.byte {
        eprintln!("Using byte-level tokenizer");
        Ok(Box::new(ByteTokenizer))
    } else {
        bail!("No tokenizer backend specified");
    }
}

fn read_text(text: Option<&str>, file: Option<&PathBuf>) -> Result<String> {
    if let Some(t) = text {
        Ok(t.to_string())
    } else if let Some(path) = file {
        std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read file: {}", path.display()))
    } else {
        eprintln!("Reading from stdin...");
        let mut buf = String::new();
        io::stdin()
            .read_to_string(&mut buf)
            .context("Failed to read stdin")?;
        Ok(buf)
    }
}

fn read_ids(ids: &[u32], file: Option<&PathBuf>) -> Result<Vec<u32>> {
    if !ids.is_empty() {
        return Ok(ids.to_vec());
    }

    let raw = if let Some(path) = file {
        std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read file: {}", path.display()))?
    } else {
        eprintln!("Reading IDs from stdin...");
        let mut buf = String::new();
        for line in io::stdin().lock().lines() {
            let line = line.context("Failed to read stdin")?;
            buf.push_str(&line);
            buf.push(' ');
        }
        buf
    };

    raw.split_whitespace()
        .map(|s| {
            s.parse::<u32>()
                .with_context(|| format!("Invalid token ID: {s}"))
        })
        .collect()
}

pub fn run(args: TokenizerArgs) -> Result<()> {
    let tokenizer = load_backend(&args.backend)?;

    match &args.command {
        TokenizerCommand::Encode { text, file, json } => {
            let input = read_text(text.as_deref(), file.as_ref())?;
            let ids = tokenizer
                .encode(&input)
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            if *json {
                println!(
                    "[{}]",
                    ids.iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            } else {
                println!(
                    "{}",
                    ids.iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                );
            }
        }

        TokenizerCommand::Decode { ids, file } => {
            let token_ids = read_ids(ids, file.as_ref())?;
            let text = tokenizer
                .decode(&token_ids)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            print!("{text}");
        }

        TokenizerCommand::Info { lookup } => {
            println!("Vocab size: {}", tokenizer.vocab_size());

            for token in lookup {
                match tokenizer.token_to_id(token) {
                    Some(id) => println!("  {token} -> {id}"),
                    None => println!("  {token} -> (not found)"),
                }
            }
        }
    }

    Ok(())
}
