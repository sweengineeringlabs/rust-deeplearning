//! TokenizerVerifierTest: Integration test comparing GgufTokenizer vs HFTokenizer.
//!
//! Run with:
//!   cargo test -p rustml-nlp --release --test tokenizer_verifier_test -- --nocapture
//!
//! Requires two environment variables:
//!   GGUF_MODEL_PATH  - path to a .gguf file (e.g. gemma-3-1b-it-Q4_0.gguf)
//!   HF_TOKENIZER_PATH - path to the matching tokenizer.json
//!
//! If either variable is unset, the test is skipped (not failed).

use rustml_gguf::GGUFFile;
use rustml_nlp::{GgufTokenizer, HFTokenizer, Tokenizer};

/// Get paths from env, returning None if either is missing.
fn get_paths() -> Option<(String, String)> {
    let gguf = std::env::var("GGUF_MODEL_PATH").ok()?;
    let hf = std::env::var("HF_TOKENIZER_PATH").ok()?;
    Some((gguf, hf))
}

/// Test strings covering common tokenization patterns.
const TEST_STRINGS: &[&str] = &[
    "The capital of France is",
    "Hello, world!",
    "What is the meaning of life? Answer in one sentence.",
    "The year 2025 had 365 days, as usual.",
    "price: $19.99 (20% off!)",
    "café résumé naïve",
    "fn main() { println!(\"hello\"); }",
    "aaaaabbbbbccccc",
    "word   with   extra   spaces",
    "a",
    " ",
    ".",
    "HTTP GET /api/v2/users?id=42&name=Alice",
    "line1\nline2\ttab",
    "supercalifragilisticexpialidocious",
];

/// Strings where GGUF and HF tokenizers are known to diverge due to
/// pre-tokenizer differences. These produce correct round-trip results
/// but different token ID sequences.
const KNOWN_EDGE_CASES: &[&str] = &[];

#[test]
fn verify_gguf_tokenizer_matches_hf() {
    let (gguf_path, hf_path) = match get_paths() {
        Some(paths) => paths,
        None => {
            eprintln!(
                "SKIP: Set GGUF_MODEL_PATH and HF_TOKENIZER_PATH to run tokenizer verification.\n\
                 Example:\n  \
                 GGUF_MODEL_PATH=/tmp/gemma3-gguf/gemma-3-1b-it-Q4_0.gguf \\\n  \
                 HF_TOKENIZER_PATH=/tmp/gemma3-gguf/tokenizer.json \\\n  \
                 cargo test -p rustml-nlp --release --test tokenizer_verifier_test -- --nocapture"
            );
            return;
        }
    };

    println!("\n=== TokenizerVerifierTest ===");
    println!("  GGUF: {}", gguf_path);
    println!("  HF:   {}\n", hf_path);

    // Load tokenizers
    let gguf = GGUFFile::parse_header(&gguf_path).expect("Failed to parse GGUF file");
    let gguf_tok = GgufTokenizer::from_gguf(&gguf).expect("Failed to build GgufTokenizer");
    let hf_tok = HFTokenizer::from_file(&hf_path).expect("Failed to load HFTokenizer");

    println!(
        "  Vocab: GGUF={}, HF={}\n",
        gguf_tok.vocab_size(),
        hf_tok.vocab_size()
    );

    let mut failures = Vec::new();
    let mut known_diffs = Vec::new();

    for &text in TEST_STRINGS {
        let gguf_ids = gguf_tok.encode(text).expect("GGUF encode failed");
        let hf_ids = hf_tok.encode(text).expect("HF encode failed");
        let is_known_edge_case = KNOWN_EDGE_CASES.contains(&text);

        if gguf_ids == hf_ids {
            println!("  PASS  [{:>3} tokens]  {:?}", gguf_ids.len(), truncate(text, 50));
        } else if is_known_edge_case {
            println!(
                "  KNOWN [gguf={}, hf={}]  {:?}  (pre-tokenizer difference)",
                gguf_ids.len(),
                hf_ids.len(),
                truncate(text, 50),
            );
            known_diffs.push(text.to_string());
        } else {
            println!(
                "  FAIL  [gguf={}, hf={}]  {:?}",
                gguf_ids.len(),
                hf_ids.len(),
                truncate(text, 50),
            );

            // Print token-by-token diff
            let max_len = gguf_ids.len().max(hf_ids.len());
            for i in 0..max_len {
                let g = gguf_ids.get(i);
                let h = hf_ids.get(i);
                let g_str = g
                    .map(|&id| gguf_tok.decode(&[id]).unwrap_or_default())
                    .unwrap_or_default();
                let h_str = h
                    .map(|&id| hf_tok.decode(&[id]).unwrap_or_default())
                    .unwrap_or_default();
                let marker = if g == h { " " } else { "!" };
                println!(
                    "    {} [{}] gguf={} {:?}  hf={} {:?}",
                    marker,
                    i,
                    g.map(|id| id.to_string()).unwrap_or("-".into()),
                    g_str,
                    h.map(|id| id.to_string()).unwrap_or("-".into()),
                    h_str,
                );
            }

            failures.push(text.to_string());
        }
    }

    println!("\n--- Summary ---");
    println!(
        "  {} passed, {} known edge cases, {} failed, {} total",
        TEST_STRINGS.len() - failures.len() - known_diffs.len(),
        known_diffs.len(),
        failures.len(),
        TEST_STRINGS.len()
    );

    assert!(
        failures.is_empty(),
        "Tokenizer mismatch on {} test case(s): {:?}",
        failures.len(),
        failures
    );
}

/// Verify encode→decode round-trip preserves text (modulo SentencePiece normalization).
#[test]
fn verify_gguf_encode_decode_roundtrip() {
    let (gguf_path, _) = match get_paths() {
        Some(paths) => paths,
        None => return, // Skip silently
    };

    println!("\n=== Encode→Decode Round-trip Test ===\n");

    let gguf = GGUFFile::parse_header(&gguf_path).expect("Failed to parse GGUF");
    let gguf_tok = GgufTokenizer::from_gguf(&gguf).expect("Failed to build GgufTokenizer");

    for &text in TEST_STRINGS {
        let ids = gguf_tok.encode(text).expect("encode failed");
        let decoded = gguf_tok.decode(&ids).expect("decode failed");
        // SentencePiece uses ▁ for spaces; decoded text should have spaces restored
        let normalized_input = text.to_string();
        let normalized_decoded = decoded.to_string();

        if normalized_input == normalized_decoded {
            println!("  PASS  {:?}", truncate(text, 50));
        } else {
            println!("  FAIL  input={:?}  decoded={:?}", text, decoded);
            panic!(
                "Round-trip failed for {:?}: got {:?}",
                text, decoded
            );
        }
    }

    println!("\n  All round-trip tests passed.");
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}
