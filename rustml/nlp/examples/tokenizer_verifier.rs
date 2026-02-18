//! TokenizerVerifier: Compare GgufTokenizer against HFTokenizer (reference).
//!
//! Loads both tokenizers and runs a suite of test strings through each,
//! reporting PASS/FAIL with token-by-token diff on mismatch.
//!
//! Usage:
//!   cargo run -p rustml-nlp --release --example tokenizer_verifier -- \
//!     /path/to/model.gguf /path/to/tokenizer.json
//!
//! Optional: pass additional test strings as extra arguments:
//!   cargo run -p rustml-nlp --release --example tokenizer_verifier -- \
//!     /path/to/model.gguf /path/to/tokenizer.json "custom test string"

use rustml_gguf::GGUFFile;
use rustml_tokenizer::{GgufTokenizer, HFTokenizer, Tokenizer};

/// Default test strings covering common tokenization edge cases.
const DEFAULT_TEST_STRINGS: &[&str] = &[
    // Basic prompts
    "The capital of France is",
    "Hello, world!",
    // Multi-word with punctuation
    "What is the meaning of life? Answer in one sentence.",
    // Numbers and special characters
    "The year 2025 had 365 days, as usual.",
    "price: $19.99 (20% off!)",
    // Unicode and accented characters
    "café résumé naïve",
    "Tokyo (東京) is the capital of Japan.",
    // Code-like content
    "fn main() { println!(\"hello\"); }",
    // Repeated characters and whitespace
    "aaaaabbbbbccccc",
    "word   with   extra   spaces",
    // Single characters
    "a",
    " ",
    // Empty-adjacent
    ".",
    // Mixed case
    "HTTP GET /api/v2/users?id=42&name=Alice",
    // Newlines and tabs
    "line1\nline2\ttab",
    // Long word
    "supercalifragilisticexpialidocious",
];

struct VerifyResult {
    text: String,
    passed: bool,
    gguf_ids: Vec<u32>,
    hf_ids: Vec<u32>,
}

fn verify(
    text: &str,
    gguf_tok: &GgufTokenizer,
    hf_tok: &HFTokenizer,
) -> VerifyResult {
    let gguf_ids = gguf_tok.encode(text).unwrap_or_default();
    let hf_ids = hf_tok.encode(text).unwrap_or_default();
    let passed = gguf_ids == hf_ids;
    VerifyResult {
        text: text.to_string(),
        passed,
        gguf_ids,
        hf_ids,
    }
}

fn print_token_diff(gguf_ids: &[u32], hf_ids: &[u32], gguf_tok: &GgufTokenizer, hf_tok: &HFTokenizer) {
    let max_len = gguf_ids.len().max(hf_ids.len());
    println!("    {:>5}  {:>8}  {:>8}  {:<20}  {:<20}", "idx", "GGUF", "HF", "GGUF token", "HF token");
    println!("    {:->5}  {:->8}  {:->8}  {:->20}  {:->20}", "", "", "", "", "");

    for i in 0..max_len {
        let gguf_id = gguf_ids.get(i);
        let hf_id = hf_ids.get(i);

        let gguf_str = gguf_id
            .map(|&id| gguf_tok.decode(&[id]).unwrap_or_default())
            .unwrap_or_default();
        let hf_str = hf_id
            .map(|&id| hf_tok.decode(&[id]).unwrap_or_default())
            .unwrap_or_default();

        let match_marker = match (gguf_id, hf_id) {
            (Some(a), Some(b)) if a == b => " ",
            _ => "!",
        };

        println!(
            "  {} {:>5}  {:>8}  {:>8}  {:<20}  {:<20}",
            match_marker,
            i,
            gguf_id.map(|id| format!("{}", id)).unwrap_or_else(|| "-".to_string()),
            hf_id.map(|id| format!("{}", id)).unwrap_or_else(|| "-".to_string()),
            format!("{:?}", gguf_str),
            format!("{:?}", hf_str),
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <model.gguf> <tokenizer.json> [test_string ...]", args[0]);
        std::process::exit(1);
    }

    let gguf_path = &args[1];
    let hf_path = &args[2];

    println!("=== TokenizerVerifier ===\n");

    // Load GGUF tokenizer
    println!("Loading GGUF tokenizer from: {}", gguf_path);
    let gguf = GGUFFile::parse_header(gguf_path)?;
    let gguf_tok = GgufTokenizer::from_gguf(&gguf)?;
    println!("  Vocab size: {}", gguf_tok.vocab_size());

    // Load HF reference tokenizer
    println!("Loading HF tokenizer from: {}", hf_path);
    let hf_tok = HFTokenizer::from_file(hf_path)?;
    println!("  Vocab size: {}", hf_tok.vocab_size());

    if gguf_tok.vocab_size() != hf_tok.vocab_size() {
        println!(
            "\n  WARNING: Vocab size mismatch! GGUF={} vs HF={}",
            gguf_tok.vocab_size(),
            hf_tok.vocab_size()
        );
    }

    // Build test strings: defaults + any extra CLI args
    let mut test_strings: Vec<&str> = DEFAULT_TEST_STRINGS.to_vec();
    let extra_args: Vec<String> = args[3..].to_vec();
    for s in &extra_args {
        test_strings.push(s.as_str());
    }

    println!("\nRunning {} test cases...\n", test_strings.len());

    let mut results = Vec::new();
    for text in &test_strings {
        results.push(verify(text, &gguf_tok, &hf_tok));
    }

    // Print results
    let mut pass_count = 0;
    let mut fail_count = 0;

    for r in &results {
        if r.passed {
            pass_count += 1;
            println!("  PASS  [{:>3} tokens]  {:?}", r.gguf_ids.len(), truncate(&r.text, 60));
        } else {
            fail_count += 1;
            println!(
                "  FAIL  [gguf={}, hf={}]  {:?}",
                r.gguf_ids.len(),
                r.hf_ids.len(),
                truncate(&r.text, 60),
            );
            print_token_diff(&r.gguf_ids, &r.hf_ids, &gguf_tok, &hf_tok);
            println!();
        }
    }

    // Summary
    println!("\n--- Summary ---");
    println!(
        "  {} passed, {} failed, {} total",
        pass_count,
        fail_count,
        pass_count + fail_count
    );

    if fail_count > 0 {
        println!("\n  RESULT: FAIL");
        std::process::exit(1);
    } else {
        println!("\n  RESULT: ALL PASS");
    }

    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}
