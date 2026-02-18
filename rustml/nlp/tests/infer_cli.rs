use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_rustml-infer"))
}

fn tempdir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("rustml-infer-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ── help ────────────────────────────────────────────────────────────

#[test]
fn help_flag() {
    let out = bin().arg("--help").output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("rustml-infer"));
    assert!(stdout.contains("--prompt"));
    assert!(stdout.contains("--max-tokens"));
    assert!(stdout.contains("--temperature"));
    assert!(stdout.contains("--top-k"));
    assert!(stdout.contains("--top-p"));
    assert!(stdout.contains("--repetition-penalty"));
    assert!(stdout.contains("--stream"));
    assert!(stdout.contains("--chat"));
    assert!(stdout.contains("GGUF_PATH"));
}

#[test]
fn version_flag() {
    let out = bin().arg("--version").output().unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("rustml-infer"));
}

// ── error cases ─────────────────────────────────────────────────────

#[test]
fn no_arguments_shows_usage() {
    let out = bin().output().unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("Usage") || stderr.contains("GGUF_PATH"));
}

#[test]
fn nonexistent_gguf_file_fails() {
    let out = bin()
        .args([
            "/tmp/nonexistent_model_xyz.gguf",
            "--prompt",
            "Hello",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("Failed to parse GGUF") || stderr.contains("No such file"));
}

#[test]
fn invalid_gguf_file_fails() {
    let dir = tempdir();
    let path = dir.join("bad.gguf");
    std::fs::write(&path, b"this is not a valid gguf file").unwrap();

    let out = bin()
        .arg(&path)
        .args(["--prompt", "Hello"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("Failed to parse GGUF") || stderr.contains("Invalid GGUF"));
}

#[test]
fn empty_stdin_with_no_prompt_fails() {
    let dir = tempdir();
    let path = dir.join("dummy.gguf");
    // Write minimal valid GGUF header but without model metadata
    // (it will fail at config extraction, but we want to test stdin handling)
    let mut data = Vec::new();
    data.extend_from_slice(&[0x47, 0x47, 0x55, 0x46]); // GGUF magic
    data.extend_from_slice(&3u32.to_le_bytes());         // version
    data.extend_from_slice(&0u64.to_le_bytes());         // tensor_count
    data.extend_from_slice(&0u64.to_le_bytes());         // metadata_count
    std::fs::write(&path, &data).unwrap();

    // This will fail because the model config can't be extracted (no metadata),
    // but it tests that the binary starts up and processes arguments correctly
    let mut child = bin()
        .arg(&path)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    // Close stdin immediately (empty input)
    drop(child.stdin.take());
    let out = child.wait_with_output().unwrap();
    assert!(!out.status.success());
}

// ── argument parsing ────────────────────────────────────────────────

#[test]
fn invalid_temperature_type_fails() {
    let out = bin()
        .args(["dummy.gguf", "--temperature", "not_a_number"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("invalid value") || stderr.contains("not_a_number"));
}

#[test]
fn invalid_max_tokens_type_fails() {
    let out = bin()
        .args(["dummy.gguf", "--max-tokens", "abc"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

#[test]
fn invalid_top_k_type_fails() {
    let out = bin()
        .args(["dummy.gguf", "--top-k", "3.14"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

#[test]
fn invalid_top_p_type_fails() {
    let out = bin()
        .args(["dummy.gguf", "--top-p", "abc"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

#[test]
fn unknown_flag_fails() {
    let out = bin()
        .args(["dummy.gguf", "--unknown-flag"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

// ── flag acceptance (argument parsing succeeds even if file doesn't exist) ──

#[test]
fn all_sampling_flags_accepted() {
    // Verify that all sampling flags parse correctly (will fail on file open, not arg parse)
    let out = bin()
        .args([
            "/tmp/nonexistent_xyz.gguf",
            "--prompt", "Hello",
            "--max-tokens", "128",
            "--temperature", "0.7",
            "--top-k", "50",
            "--top-p", "0.9",
            "--repetition-penalty", "1.1",
            "--stream",
            "--chat",
        ])
        .output()
        .unwrap();
    // Should fail on file open, NOT on argument parsing
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    // If it fails on arg parsing, it would say "error:" and mention the flag
    // Instead it should mention the GGUF file
    assert!(
        stderr.contains("Failed to parse GGUF") || stderr.contains("No such file"),
        "Expected file-not-found error, got: {}",
        stderr
    );
}

#[test]
fn default_values_used_when_flags_omitted() {
    // With only required GGUF_PATH, the defaults should be applied
    // (temperature=0.8, max_tokens=256)
    // We can't check defaults directly, but we can verify it doesn't fail on arg parsing
    let out = bin()
        .args(["/tmp/nonexistent_xyz.gguf", "--prompt", "Hello"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Failed to parse GGUF") || stderr.contains("No such file"),
        "Expected file-not-found error, got: {}",
        stderr
    );
}
