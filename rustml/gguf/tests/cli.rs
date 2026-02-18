use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_rustml-gguf-inspect"))
}

// ── GGUF byte builder ──────────────────────────────────────────────

const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

/// Builds a synthetic GGUF v3 file in memory.
struct GgufBuilder {
    metadata: Vec<(String, u32, Vec<u8>)>, // (key, type_id, encoded_value)
    tensors: Vec<(String, Vec<usize>, u32, u64)>, // (name, dims, ggml_type, offset)
    tensor_data: Vec<Vec<u8>>,             // raw bytes per tensor (parallel to tensors)
}

impl GgufBuilder {
    fn new() -> Self {
        Self {
            metadata: Vec::new(),
            tensors: Vec::new(),
            tensor_data: Vec::new(),
        }
    }

    fn add_string(&mut self, key: &str, value: &str) -> &mut Self {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
        buf.extend_from_slice(value.as_bytes());
        self.metadata.push((key.to_string(), 8, buf)); // type 8 = String
        self
    }

    fn add_u32(&mut self, key: &str, value: u32) -> &mut Self {
        self.metadata
            .push((key.to_string(), 4, value.to_le_bytes().to_vec())); // type 4 = U32
        self
    }

    fn add_f32(&mut self, key: &str, value: f32) -> &mut Self {
        self.metadata
            .push((key.to_string(), 6, value.to_le_bytes().to_vec())); // type 6 = F32
        self
    }

    fn add_bool(&mut self, key: &str, value: bool) -> &mut Self {
        self.metadata
            .push((key.to_string(), 7, vec![value as u8])); // type 7 = Bool
        self
    }

    fn add_u32_array(&mut self, key: &str, values: &[u32]) -> &mut Self {
        let mut buf = Vec::new();
        buf.extend_from_slice(&4u32.to_le_bytes()); // elem_type = U32
        buf.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for v in values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        self.metadata.push((key.to_string(), 9, buf)); // type 9 = Array
        self
    }

    fn add_tensor(&mut self, name: &str, dims: &[usize], ggml_type: u32, offset: u64) -> &mut Self {
        self.tensors
            .push((name.to_string(), dims.to_vec(), ggml_type, offset));
        self
    }

    /// Add a tensor with associated raw data. Offset is auto-computed in `build_with_data()`.
    fn add_tensor_with_data(&mut self, name: &str, dims: &[usize], ggml_type: u32, raw_data: Vec<u8>) -> &mut Self {
        // offset=0 placeholder — build_with_data() will fix it
        self.tensors.push((name.to_string(), dims.to_vec(), ggml_type, 0));
        self.tensor_data.push(raw_data);
        self
    }

    fn build(&self) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC);
        data.extend_from_slice(&3u32.to_le_bytes()); // version
        data.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        data.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());

        // Metadata entries
        for (key, type_id, value_bytes) in &self.metadata {
            // Write key as GGUF string: u64 len + bytes
            data.extend_from_slice(&(key.len() as u64).to_le_bytes());
            data.extend_from_slice(key.as_bytes());
            // Write value type
            data.extend_from_slice(&type_id.to_le_bytes());
            // Write value
            data.extend_from_slice(value_bytes);
        }

        // Tensor info entries
        for (name, dims, ggml_type, offset) in &self.tensors {
            // name
            data.extend_from_slice(&(name.len() as u64).to_le_bytes());
            data.extend_from_slice(name.as_bytes());
            // n_dims
            data.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            // dimensions (as u64)
            for d in dims {
                data.extend_from_slice(&(*d as u64).to_le_bytes());
            }
            // ggml_type
            data.extend_from_slice(&ggml_type.to_le_bytes());
            // offset
            data.extend_from_slice(&offset.to_le_bytes());
        }

        data
    }

    /// Build GGUF with actual tensor data appended.
    /// Computes offsets and aligns data to 32-byte boundaries.
    fn build_with_data(&mut self) -> Vec<u8> {
        // First, compute tensor offsets relative to data section start
        let mut offset = 0u64;
        for (i, (_name, _dims, _ggml_type, tensor_offset)) in self.tensors.iter_mut().enumerate() {
            *tensor_offset = offset;
            let data_len = self.tensor_data[i].len() as u64;
            // Align next tensor to 32 bytes
            offset = (offset + data_len + 31) & !31;
        }

        // Build the header
        let mut data = self.build();

        // Pad header to 32-byte alignment (data_offset)
        let header_end = data.len();
        let data_offset = (header_end + 31) & !31;
        data.resize(data_offset, 0);

        // Append tensor data with 32-byte alignment
        for td in &self.tensor_data {
            data.extend_from_slice(td);
            // Pad to 32-byte alignment
            let padded = (data.len() + 31) & !31;
            data.resize(padded, 0);
        }

        data
    }
}

/// Encode f32 slice to little-endian bytes.
fn encode_f32(values: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(values.len() * 4);
    for v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Build a GGUF file with realistic model metadata (llama architecture).
fn build_llama_gguf() -> Vec<u8> {
    let mut b = GgufBuilder::new();
    b.add_string("general.architecture", "llama")
        .add_u32("llama.embedding_length", 4096)
        .add_u32("llama.feed_forward_length", 11008)
        .add_u32("llama.block_count", 32)
        .add_u32("llama.attention.head_count", 32)
        .add_u32("llama.attention.head_count_kv", 8)
        .add_u32("llama.context_length", 2048)
        .add_f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
        .add_f32("llama.rope.freq_base", 10000.0)
        .add_u32("tokenizer.ggml.bos_token_id", 1)
        .add_u32("tokenizer.ggml.eos_token_id", 2)
        .add_string("general.name", "LLaMA-7B")
        .add_bool("general.quantized", true)
        // Add tensor infos: F32 = 0, Q4_0 = 2
        .add_tensor("token_embd.weight", &[4096, 32000], 0, 0)
        .add_tensor("blk.0.attn_q.weight", &[4096, 4096], 2, 524288000)
        .add_tensor("blk.0.attn_k.weight", &[1024, 4096], 2, 524296192)
        .add_tensor("output.weight", &[4096, 32000], 0, 999999999);
    b.build()
}

use std::sync::atomic::{AtomicU64, Ordering};
static COUNTER: AtomicU64 = AtomicU64::new(0);

fn tempdir() -> std::path::PathBuf {
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rustml-gguf-test-{}-{}",
        std::process::id(),
        id
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_gguf(data: &[u8]) -> std::path::PathBuf {
    let dir = tempdir();
    let path = dir.join("test.gguf");
    std::fs::write(&path, data).unwrap();
    path
}

// ── help ────────────────────────────────────────────────────────────

#[test]
fn help_flag() {
    let out = bin().arg("--help").output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("rustml-gguf-inspect"));
    assert!(stdout.contains("info"));
    assert!(stdout.contains("meta"));
    assert!(stdout.contains("tensors"));
}

#[test]
fn info_subcommand_help() {
    let out = bin().args(["info", "--help"]).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("PATH"));
}

#[test]
fn meta_subcommand_help() {
    let out = bin().args(["meta", "--help"]).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("--key"));
}

#[test]
fn tensors_subcommand_help() {
    let out = bin().args(["tensors", "--help"]).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("--filter"));
}

// ── info subcommand ─────────────────────────────────────────────────

#[test]
fn info_shows_version_and_tensor_count() {
    let path = write_gguf(&build_llama_gguf());
    let out = bin().args(["info"]).arg(&path).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("GGUF version:  3"));
    assert!(stdout.contains("Tensor count:  4"));
}

#[test]
fn info_shows_architecture() {
    let path = write_gguf(&build_llama_gguf());
    let out = bin().args(["info"]).arg(&path).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Architecture:  llama"));
}

#[test]
fn info_shows_model_dimensions() {
    let path = write_gguf(&build_llama_gguf());
    let out = bin().args(["info"]).arg(&path).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Dimensions:    4096"));
    assert!(stdout.contains("Hidden dim:    11008"));
    assert!(stdout.contains("Layers:        32"));
    assert!(stdout.contains("Heads:         32"));
    assert!(stdout.contains("KV heads:      8"));
    assert!(stdout.contains("Vocab size:    32000"));
    assert!(stdout.contains("Max seq len:   2048"));
}

#[test]
fn info_minimal_gguf_no_model_config() {
    // A minimal GGUF with no metadata can't produce a model config
    let b = GgufBuilder::new();
    // No metadata at all
    let path = write_gguf(&b.build());
    let out = bin().args(["info"]).arg(&path).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("GGUF version:  3"));
    assert!(stdout.contains("Tensor count:  0"));
    // Should print error about missing config on stderr
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("Could not extract model config"));
}

// ── meta subcommand ─────────────────────────────────────────────────

#[test]
fn meta_lists_all_keys_sorted() {
    let path = write_gguf(&build_llama_gguf());
    let out = bin().args(["meta"]).arg(&path).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Check that keys are present
    assert!(stdout.contains("general.architecture = \"llama\""));
    assert!(stdout.contains("general.name = \"LLaMA-7B\""));
    assert!(stdout.contains("general.quantized = true"));
    assert!(stdout.contains("llama.embedding_length = 4096"));
    assert!(stdout.contains("llama.feed_forward_length = 11008"));
    assert!(stdout.contains("llama.block_count = 32"));
    assert!(stdout.contains("llama.rope.freq_base = 10000"));

    // Verify sorted order: "general.*" before "llama.*" before "tokenizer.*"
    let lines: Vec<&str> = stdout.lines().collect();
    let keys: Vec<&str> = lines.iter().map(|l| l.split(" = ").next().unwrap()).collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted, "metadata keys should be sorted");
}

#[test]
fn meta_filter_by_key() {
    let path = write_gguf(&build_llama_gguf());
    let out = bin()
        .args(["meta", "--key", "general.architecture"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        stdout.trim(),
        "general.architecture = \"llama\""
    );
}

#[test]
fn meta_filter_by_key_u32() {
    let path = write_gguf(&build_llama_gguf());
    let out = bin()
        .args(["meta", "--key", "llama.block_count"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "llama.block_count = 32"
    );
}

#[test]
fn meta_filter_missing_key() {
    let path = write_gguf(&build_llama_gguf());
    let out = bin()
        .args(["meta", "--key", "nonexistent.key"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success());
    // stdout should be empty, stderr should show error
    assert!(String::from_utf8_lossy(&out.stdout).trim().is_empty());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("Key not found: nonexistent.key"));
}

#[test]
fn meta_formats_bool_value() {
    let path = write_gguf(&build_llama_gguf());
    let out = bin()
        .args(["meta", "--key", "general.quantized"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "general.quantized = true"
    );
}

#[test]
fn meta_formats_f32_value() {
    let path = write_gguf(&build_llama_gguf());
    let out = bin()
        .args(["meta", "--key", "llama.attention.layer_norm_rms_epsilon"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("0.00001"));
}

#[test]
fn meta_formats_string_truncation() {
    let long_str = "x".repeat(300);
    let mut b = GgufBuilder::new();
    b.add_string("long.key", &long_str);
    let path = write_gguf(&b.build());

    let out = bin().args(["meta"]).arg(&path).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("...\" (len=300)"));
    // Ensure only 200 chars of the string appear before truncation
    assert!(stdout.contains(&"x".repeat(200)));
}

#[test]
fn meta_formats_array_value() {
    let mut b = GgufBuilder::new();
    b.add_u32_array("test.array", &[1, 2, 3, 4, 5]);
    let path = write_gguf(&b.build());

    let out = bin().args(["meta"]).arg(&path).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("test.array = [array, len=5]"));
}

// ── tensors subcommand ──────────────────────────────────────────────

#[test]
fn tensors_lists_all() {
    let path = write_gguf(&build_llama_gguf());
    let out = bin().args(["tensors"]).arg(&path).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("token_embd.weight"));
    assert!(stdout.contains("blk.0.attn_q.weight"));
    assert!(stdout.contains("blk.0.attn_k.weight"));
    assert!(stdout.contains("output.weight"));
    // Stderr should show count
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("4 tensor(s) listed"));
}

#[test]
fn tensors_shows_dtype_and_shape() {
    let path = write_gguf(&build_llama_gguf());
    let out = bin().args(["tensors"]).arg(&path).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    // token_embd.weight is F32 with dims [4096, 32000]
    let embd_line = stdout
        .lines()
        .find(|l| l.contains("token_embd.weight"))
        .unwrap();
    assert!(embd_line.contains("F32"));
    assert!(embd_line.contains("4096"));
    assert!(embd_line.contains("32000"));
    // attn_q is Q4_0
    let attn_line = stdout
        .lines()
        .find(|l| l.contains("blk.0.attn_q.weight"))
        .unwrap();
    assert!(attn_line.contains("Q4_0"));
}

#[test]
fn tensors_shows_offset() {
    let path = write_gguf(&build_llama_gguf());
    let out = bin().args(["tensors"]).arg(&path).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    // token_embd.weight has offset=0
    let embd_line = stdout
        .lines()
        .find(|l| l.contains("token_embd.weight"))
        .unwrap();
    assert!(embd_line.contains("offset=0"));
}

#[test]
fn tensors_filter_by_name() {
    let path = write_gguf(&build_llama_gguf());
    let out = bin()
        .args(["tensors", "--filter", "attn"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("blk.0.attn_q.weight"));
    assert!(stdout.contains("blk.0.attn_k.weight"));
    assert!(!stdout.contains("token_embd.weight"));
    assert!(!stdout.contains("output.weight"));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("2 tensor(s) listed"));
}

#[test]
fn tensors_filter_no_match() {
    let path = write_gguf(&build_llama_gguf());
    let out = bin()
        .args(["tensors", "--filter", "nonexistent_pattern"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.trim().is_empty());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("0 tensor(s) listed"));
}

#[test]
fn tensors_empty_file() {
    let b = GgufBuilder::new();
    let path = write_gguf(&b.build());
    let out = bin().args(["tensors"]).arg(&path).output().unwrap();
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("0 tensor(s) listed"));
}

// ── error cases ─────────────────────────────────────────────────────

#[test]
fn nonexistent_file_fails() {
    let out = bin()
        .args(["info", "/tmp/nonexistent_model_xyz.gguf"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

#[test]
fn invalid_gguf_file_fails() {
    let dir = tempdir();
    let path = dir.join("bad.gguf");
    std::fs::write(&path, b"not a gguf file").unwrap();
    let out = bin().args(["info"]).arg(&path).output().unwrap();
    assert!(!out.status.success());
}

#[test]
fn no_subcommand_shows_help() {
    let out = bin().output().unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("Usage"));
}

#[test]
fn meta_nonexistent_file_fails() {
    let out = bin()
        .args(["meta", "/tmp/nonexistent_model_xyz.gguf"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

#[test]
fn tensors_nonexistent_file_fails() {
    let out = bin()
        .args(["tensors", "/tmp/nonexistent_model_xyz.gguf"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

// ── verify subcommand ───────────────────────────────────────────────

/// Build a GGUF with full layer tensors that passes all verify checks.
fn build_verify_passing_gguf() -> GgufBuilder {
    let mut b = GgufBuilder::new();
    let dim = 64usize;
    let vocab = 128usize;
    let hidden = 256usize;
    let n_layers = 2usize;
    let n_kv_heads = 2usize;
    let kv_dim = (dim / 4) * n_kv_heads; // 32

    b.add_string("general.architecture", "llama")
        .add_u32("llama.embedding_length", dim as u32)
        .add_u32("llama.feed_forward_length", hidden as u32)
        .add_u32("llama.block_count", n_layers as u32)
        .add_u32("llama.attention.head_count", 4)
        .add_u32("llama.attention.head_count_kv", n_kv_heads as u32)
        .add_u32("llama.vocab_size", vocab as u32)
        .add_u32("llama.context_length", 512)
        .add_f32("llama.attention.layer_norm_rms_epsilon", 1e-5)
        .add_f32("llama.rope.freq_base", 10000.0)
        // Global tensors
        .add_tensor("token_embd.weight", &[dim, vocab], 0, 0)
        .add_tensor("output.weight", &[dim, vocab], 0, 0);

    // Per-layer tensors
    for layer in 0..n_layers {
        let l = layer;
        b.add_tensor(&format!("blk.{l}.attn_norm.weight"), &[dim], 0, 0)
            .add_tensor(&format!("blk.{l}.ffn_norm.weight"), &[dim], 0, 0)
            .add_tensor(&format!("blk.{l}.attn_q.weight"), &[dim, dim], 0, 0)
            .add_tensor(&format!("blk.{l}.attn_k.weight"), &[dim, kv_dim], 0, 0)
            .add_tensor(&format!("blk.{l}.attn_v.weight"), &[dim, kv_dim], 0, 0)
            .add_tensor(&format!("blk.{l}.attn_output.weight"), &[dim, dim], 0, 0)
            .add_tensor(&format!("blk.{l}.ffn_gate.weight"), &[dim, hidden], 0, 0)
            .add_tensor(&format!("blk.{l}.ffn_up.weight"), &[dim, hidden], 0, 0)
            .add_tensor(&format!("blk.{l}.ffn_down.weight"), &[hidden, dim], 0, 0);
    }
    b
}

#[test]
fn verify_pass_valid_model() {
    let b = build_verify_passing_gguf();
    let path = write_gguf(&b.build());
    let out = bin().args(["verify"]).arg(&path).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("PASS  token_embd.weight exists"));
    assert!(stdout.contains("PASS  output.weight exists"));
    assert!(stdout.contains("PASS  all layer tensors present"));
    // Check no FAIL check lines (the summary line "0 FAIL" is fine)
    assert!(!stdout.lines().any(|l| l.starts_with("FAIL")), "unexpected FAIL in output:\n{stdout}");
}

#[test]
fn verify_fail_missing_token_embd() {
    let mut b = GgufBuilder::new();
    b.add_string("general.architecture", "llama")
        .add_u32("llama.embedding_length", 64)
        .add_u32("llama.block_count", 0)
        .add_u32("llama.attention.head_count", 4)
        .add_u32("llama.context_length", 512);
    // No token_embd.weight
    let path = write_gguf(&b.build());
    let out = bin().args(["verify"]).arg(&path).output().unwrap();
    assert!(!out.status.success()); // exit code 1
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("FAIL  token_embd.weight missing"));
}

#[test]
fn verify_fail_wrong_vocab_size() {
    let mut b = GgufBuilder::new();
    b.add_string("general.architecture", "llama")
        .add_u32("llama.embedding_length", 64)
        .add_u32("llama.block_count", 0)
        .add_u32("llama.attention.head_count", 4)
        .add_u32("llama.vocab_size", 999) // mismatch!
        .add_u32("llama.context_length", 512)
        .add_tensor("token_embd.weight", &[64, 128], 0, 0);
    let path = write_gguf(&b.build());
    let out = bin().args(["verify"]).arg(&path).output().unwrap();
    assert!(!out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("FAIL  vocab_size=999 != token_embd.weight[1]=128"));
}

#[test]
fn verify_fail_missing_layer_tensors() {
    let mut b = GgufBuilder::new();
    b.add_string("general.architecture", "llama")
        .add_u32("llama.embedding_length", 64)
        .add_u32("llama.block_count", 2) // 2 layers
        .add_u32("llama.attention.head_count", 4)
        .add_u32("llama.vocab_size", 128)
        .add_u32("llama.context_length", 512)
        .add_tensor("token_embd.weight", &[64, 128], 0, 0)
        .add_tensor("output.weight", &[64, 128], 0, 0);
    // Only blk.0 tensors, no blk.1
    for suffix in &[
        "attn_norm.weight", "ffn_norm.weight", "attn_q.weight",
        "attn_k.weight", "attn_v.weight", "attn_output.weight",
        "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    ] {
        b.add_tensor(&format!("blk.0.{suffix}"), &[64], 0, 0);
    }
    let path = write_gguf(&b.build());
    let out = bin().args(["verify"]).arg(&path).output().unwrap();
    assert!(!out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("FAIL  missing blk.1."));
}

#[test]
fn verify_warns_no_output_weight() {
    let mut b = GgufBuilder::new();
    b.add_string("general.architecture", "llama")
        .add_u32("llama.embedding_length", 64)
        .add_u32("llama.block_count", 0)
        .add_u32("llama.attention.head_count", 4)
        .add_u32("llama.vocab_size", 128)
        .add_u32("llama.context_length", 512)
        .add_tensor("token_embd.weight", &[64, 128], 0, 0);
    // No output.weight
    let path = write_gguf(&b.build());
    let out = bin().args(["verify"]).arg(&path).output().unwrap();
    assert!(out.status.success()); // exit 0 — warns only
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("WARN  output.weight missing"));
}

#[test]
fn verify_help() {
    let out = bin().args(["verify", "--help"]).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("PATH"));
}

// ── tensors --stats ─────────────────────────────────────────────────

#[test]
fn tensors_stats_f32_known_values() {
    let values: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let raw = encode_f32(&values);
    let mut b = GgufBuilder::new();
    b.add_string("general.architecture", "llama")
        .add_u32("llama.embedding_length", 32)
        .add_u32("llama.block_count", 0)
        .add_u32("llama.attention.head_count", 1)
        .add_u32("llama.context_length", 32)
        .add_tensor_with_data("test.weight", &[32], 0, raw);
    let path = write_gguf(&b.build_with_data());
    let out = bin()
        .args(["tensors", "--stats"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("stats:"), "expected stats output, got:\n{stdout}");
    assert!(stdout.contains("min=0.0"));
    assert!(stdout.contains("max=31.0"));
    assert!(stdout.contains("mean=15.5"));
    assert!(stdout.contains("n=32"));
}

#[test]
fn tensors_stats_with_filter() {
    let v1: Vec<f32> = vec![1.0; 32];
    let v2: Vec<f32> = vec![2.0; 32];
    let mut b = GgufBuilder::new();
    b.add_string("general.architecture", "llama")
        .add_u32("llama.embedding_length", 32)
        .add_u32("llama.block_count", 0)
        .add_u32("llama.attention.head_count", 1)
        .add_u32("llama.context_length", 32)
        .add_tensor_with_data("alpha.weight", &[32], 0, encode_f32(&v1))
        .add_tensor_with_data("beta.weight", &[32], 0, encode_f32(&v2));
    let path = write_gguf(&b.build_with_data());
    let out = bin()
        .args(["tensors", "--stats", "--filter", "beta"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("beta.weight"));
    assert!(!stdout.contains("alpha.weight"));
    assert!(stdout.contains("min=2.0"));
}

#[test]
fn tensors_stats_no_flag_no_stats() {
    let values: Vec<f32> = vec![1.0; 32];
    let mut b = GgufBuilder::new();
    b.add_string("general.architecture", "llama")
        .add_u32("llama.embedding_length", 32)
        .add_u32("llama.block_count", 0)
        .add_u32("llama.attention.head_count", 1)
        .add_u32("llama.context_length", 32)
        .add_tensor_with_data("test.weight", &[32], 0, encode_f32(&values));
    let path = write_gguf(&b.build_with_data());
    // No --stats flag
    let out = bin()
        .args(["tensors"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(!stdout.contains("stats:"));
}

#[test]
fn tensors_stats_combined_with_head() {
    let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.5).collect();
    let mut b = GgufBuilder::new();
    b.add_string("general.architecture", "llama")
        .add_u32("llama.embedding_length", 32)
        .add_u32("llama.block_count", 0)
        .add_u32("llama.attention.head_count", 1)
        .add_u32("llama.context_length", 32)
        .add_tensor_with_data("test.weight", &[32], 0, encode_f32(&values));
    let path = write_gguf(&b.build_with_data());
    let out = bin()
        .args(["tensors", "--stats", "--head", "3"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("stats:"));
    assert!(stdout.contains("head(3):"));
}

// ── tensors --head ──────────────────────────────────────────────────

#[test]
fn tensors_head_shows_first_n_values() {
    let values: Vec<f32> = (0..64).map(|i| i as f32 + 0.5).collect();
    let mut b = GgufBuilder::new();
    b.add_string("general.architecture", "llama")
        .add_u32("llama.embedding_length", 64)
        .add_u32("llama.block_count", 0)
        .add_u32("llama.attention.head_count", 1)
        .add_u32("llama.context_length", 64)
        .add_tensor_with_data("test.weight", &[64], 0, encode_f32(&values));
    let path = write_gguf(&b.build_with_data());
    let out = bin()
        .args(["tensors", "--head", "5"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("head(5):"));
    assert!(stdout.contains("0.5"));
    assert!(stdout.contains("1.5"));
    assert!(stdout.contains("4.5"));
}

#[test]
fn tensors_head_larger_than_tensor() {
    let values: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let mut b = GgufBuilder::new();
    b.add_string("general.architecture", "llama")
        .add_u32("llama.embedding_length", 16)
        .add_u32("llama.block_count", 0)
        .add_u32("llama.attention.head_count", 1)
        .add_u32("llama.context_length", 16)
        .add_tensor_with_data("test.weight", &[16], 0, encode_f32(&values));
    let path = write_gguf(&b.build_with_data());
    let out = bin()
        .args(["tensors", "--head", "100"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    // Should print all 16 elements, clamped to tensor size
    assert!(stdout.contains("head(16):"));
    assert!(stdout.contains("15.0"));
}

#[test]
fn tensors_head_zero() {
    let values: Vec<f32> = vec![1.0; 32];
    let mut b = GgufBuilder::new();
    b.add_string("general.architecture", "llama")
        .add_u32("llama.embedding_length", 32)
        .add_u32("llama.block_count", 0)
        .add_u32("llama.attention.head_count", 1)
        .add_u32("llama.context_length", 32)
        .add_tensor_with_data("test.weight", &[32], 0, encode_f32(&values));
    let path = write_gguf(&b.build_with_data());
    let out = bin()
        .args(["tensors", "--head", "0"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("head(0): []"));
}

#[test]
fn tensors_help_shows_new_flags() {
    let out = bin().args(["tensors", "--help"]).output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("--stats"));
    assert!(stdout.contains("--head"));
}
