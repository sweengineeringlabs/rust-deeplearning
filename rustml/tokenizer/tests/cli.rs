use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_rustml-tokenizer"))
}

// ── encode ──────────────────────────────────────────────────────────

#[test]
fn encode_hello() {
    let out = bin()
        .args(["--byte", "encode", "Hello"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "72 101 108 108 111");
}

#[test]
fn encode_json_output() {
    let out = bin()
        .args(["--byte", "encode", "--json", "Hello"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "[72, 101, 108, 108, 111]"
    );
}

#[test]
fn encode_empty_string() {
    let out = bin()
        .args(["--byte", "encode", ""])
        .output()
        .unwrap();
    assert!(out.status.success());
    // Empty input → empty output (just a newline from println)
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "");
}

#[test]
fn encode_spaces_and_punctuation() {
    let out = bin()
        .args(["--byte", "encode", "a b!"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "97 32 98 33"
    );
}

#[test]
fn encode_multibyte_utf8() {
    // '€' is U+20AC, encoded as 3 bytes: 0xE2 0x82 0xAC
    let out = bin()
        .args(["--byte", "encode", "€"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "226 130 172"
    );
}

#[test]
fn encode_from_file() {
    let dir = tempdir();
    let path = dir.join("input.txt");
    std::fs::write(&path, "Hi").unwrap();

    let out = bin()
        .args(["--byte", "encode", "--file"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "72 105");
}

#[test]
fn encode_from_stdin() {
    use std::io::Write;
    let mut child = bin()
        .args(["--byte", "encode"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    child.stdin.take().unwrap().write_all(b"OK").unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "79 75");
}

// ── decode ──────────────────────────────────────────────────────────

#[test]
fn decode_hello() {
    let out = bin()
        .args(["--byte", "decode", "72", "101", "108", "108", "111"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(String::from_utf8_lossy(&out.stdout), "Hello");
}

#[test]
fn decode_empty() {
    // decode with no IDs and a file containing nothing
    let dir = tempdir();
    let path = dir.join("empty.txt");
    std::fs::write(&path, "").unwrap();

    let out = bin()
        .args(["--byte", "decode", "--file"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(String::from_utf8_lossy(&out.stdout), "");
}

#[test]
fn decode_from_file() {
    let dir = tempdir();
    let path = dir.join("ids.txt");
    std::fs::write(&path, "72 105\n33").unwrap();

    let out = bin()
        .args(["--byte", "decode", "--file"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(String::from_utf8_lossy(&out.stdout), "Hi!");
}

#[test]
fn decode_from_stdin() {
    use std::io::Write;
    let mut child = bin()
        .args(["--byte", "decode"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();

    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"79 75")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    assert_eq!(String::from_utf8_lossy(&out.stdout), "OK");
}

// ── round-trip ──────────────────────────────────────────────────────

#[test]
fn roundtrip_ascii() {
    let text = "The quick brown fox jumps over 42 lazy dogs!";
    let enc = bin()
        .args(["--byte", "encode", text])
        .output()
        .unwrap();
    assert!(enc.status.success());

    let ids: Vec<&str> = String::from_utf8_lossy(&enc.stdout)
        .trim()
        .split(' ')
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .into_iter()
        .map(|s| Box::leak(s.into_boxed_str()) as &str)
        .collect();

    let mut args = vec!["--byte", "decode"];
    args.extend(ids.iter());
    let dec = bin().args(&args).output().unwrap();
    assert!(dec.status.success());
    assert_eq!(String::from_utf8_lossy(&dec.stdout), text);
}

// ── info ────────────────────────────────────────────────────────────

#[test]
fn info_vocab_size() {
    let out = bin()
        .args(["--byte", "info"])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8_lossy(&out.stdout).trim(),
        "Vocab size: 256"
    );
}

#[test]
fn info_lookup_not_found() {
    let out = bin()
        .args(["--byte", "info", "--lookup", "<bos>"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("Vocab size: 256"));
    assert!(stdout.contains("<bos> -> (not found)"));
}

// ── error cases ─────────────────────────────────────────────────────

#[test]
fn missing_backend_fails() {
    let out = bin()
        .args(["encode", "hello"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

#[test]
fn nonexistent_gguf_file_fails() {
    let out = bin()
        .args(["--gguf", "/tmp/nonexistent_model.gguf", "encode", "hello"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

#[test]
fn nonexistent_hf_file_fails() {
    let out = bin()
        .args(["--hf", "/tmp/nonexistent_tokenizer.json", "encode", "hello"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}

#[test]
fn decode_invalid_ids_from_file_fails() {
    let dir = tempdir();
    let path = dir.join("bad_ids.txt");
    std::fs::write(&path, "not_a_number").unwrap();

    let out = bin()
        .args(["--byte", "decode", "--file"])
        .arg(&path)
        .output()
        .unwrap();
    assert!(!out.status.success());
}

// ── stderr diagnostics ──────────────────────────────────────────────

#[test]
fn stderr_shows_backend_message() {
    let out = bin()
        .args(["--byte", "encode", "x"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("byte-level tokenizer"));
}

// ── helpers ─────────────────────────────────────────────────────────

fn tempdir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("rustml-cli-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
