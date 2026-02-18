# Manual Tokenizer Tests

> **TLDR:** Manual test checklist for the `rustml-tokenizer` CLI: encode, decode, vocab info, and all four backends.

**Audience**: Developers, QA

**WHAT**: Manual test procedures for the tokenizer CLI tool
**WHY**: Validates tokenizer encode/decode correctness across BPE, GGUF, HuggingFace, and byte backends
**HOW**: Step-by-step test tables with expected outcomes

---

## Table of Contents

- [Help & Version](#1-help--version)
- [Byte Backend — Encode](#2-byte-backend--encode)
- [Byte Backend — Decode](#3-byte-backend--decode)
- [Byte Backend — Info](#4-byte-backend--info)
- [Round-trip](#5-round-trip)
- [GGUF Backend](#6-gguf-backend)
- [HuggingFace Backend](#7-huggingface-backend)
- [BPE Backend](#8-bpe-backend)
- [File & Stdin Input](#9-file--stdin-input)
- [Error Cases](#10-error-cases)

---

## 1. Help & Version

| Test | Command | Expected |
|------|---------|----------|
| Help flag | `rustml-tokenizer --help` | Prints usage with `encode`, `decode`, `info` subcommands |
| Version flag | `rustml-tokenizer --version` | Prints version string |
| No args | `rustml-tokenizer` | Shows error and usage (backend required) |

## 2. Byte Backend — Encode

| Test | Command | Expected |
|------|---------|----------|
| Encode ASCII | `rustml-tokenizer --byte encode "Hello"` | `72 101 108 108 111` |
| Encode JSON | `rustml-tokenizer --byte encode --json "Hello"` | `[72, 101, 108, 108, 111]` |
| Encode empty | `rustml-tokenizer --byte encode ""` | Empty output |
| Encode spaces | `rustml-tokenizer --byte encode "a b!"` | `97 32 98 33` |
| Encode UTF-8 | `rustml-tokenizer --byte encode "€"` | `226 130 172` (3 UTF-8 bytes) |

## 3. Byte Backend — Decode

| Test | Command | Expected |
|------|---------|----------|
| Decode ASCII | `rustml-tokenizer --byte decode 72 101 108 108 111` | `Hello` |
| Decode empty | `rustml-tokenizer --byte decode` (no IDs, empty file) | Empty output |

## 4. Byte Backend — Info

| Test | Command | Expected |
|------|---------|----------|
| Vocab size | `rustml-tokenizer --byte info` | `Vocab size: 256` |
| Lookup missing | `rustml-tokenizer --byte info --lookup "<bos>"` | `<bos> -> (not found)` |

## 5. Round-trip

| Test | Steps | Expected |
|------|-------|----------|
| Encode then decode | Encode `"The quick brown fox"`, feed IDs to decode | Original text recovered |

## 6. GGUF Backend

> **Prerequisite**: A GGUF model file (e.g., `gemma-3-1b-it-Q4_0.gguf`).

| Test | Command | Expected |
|------|---------|----------|
| Encode text | `rustml-tokenizer --gguf /path/to/model.gguf encode "Hello world"` | Prints token IDs |
| Decode IDs | `rustml-tokenizer --gguf /path/to/model.gguf decode <IDs>` | Recovers text |
| Vocab info | `rustml-tokenizer --gguf /path/to/model.gguf info` | Shows vocab size (e.g., `262144` for Gemma 3) |
| Lookup special | `rustml-tokenizer --gguf /path/to/model.gguf info --lookup "<bos>"` | Shows BOS token ID (e.g., `2`) |
| Round-trip | Encode then decode | Text recovered (may differ in whitespace due to SentencePiece normalization) |

## 7. HuggingFace Backend

> **Prerequisite**: A `tokenizer.json` file from HuggingFace.

| Test | Command | Expected |
|------|---------|----------|
| Encode text | `rustml-tokenizer --hf /path/to/tokenizer.json encode "Hello world"` | Prints token IDs |
| Decode IDs | `rustml-tokenizer --hf /path/to/tokenizer.json decode <IDs>` | Recovers text |
| Vocab info | `rustml-tokenizer --hf /path/to/tokenizer.json info` | Shows vocab size |

## 8. BPE Backend

> **Prerequisite**: GPT-2 `vocab.json` and `merges.txt` files.

| Test | Command | Expected |
|------|---------|----------|
| Encode text | `rustml-tokenizer --bpe vocab.json merges.txt encode "Hello world"` | Prints GPT-2 token IDs |
| Decode IDs | `rustml-tokenizer --bpe vocab.json merges.txt decode <IDs>` | Recovers text |

## 9. File & Stdin Input

| Test | Steps | Expected |
|------|-------|----------|
| Encode from file | `echo "Hi" > /tmp/input.txt && rustml-tokenizer --byte encode --file /tmp/input.txt` | Encodes file contents |
| Encode from stdin | `echo "OK" \| rustml-tokenizer --byte encode` | `79 75` |
| Decode from file | Write IDs to file, `rustml-tokenizer --byte decode --file /tmp/ids.txt` | Decoded text |
| Decode from stdin | `echo "72 105 33" \| rustml-tokenizer --byte decode` | `Hi!` |

## 10. Error Cases

| Test | Command | Expected |
|------|---------|----------|
| No backend | `rustml-tokenizer encode "hello"` | Error: no backend specified |
| Bad GGUF path | `rustml-tokenizer --gguf /nonexistent.gguf encode "hello"` | Error: failed to parse GGUF |
| Bad HF path | `rustml-tokenizer --hf /nonexistent.json encode "hello"` | Error: failed to load tokenizer |
| Invalid IDs | Write `"not_a_number"` to file, decode from it | Error: invalid token ID |

---

## See Also

- [Manual Testing Hub](manual_testing.md) — prerequisites and setup
- [GGUF Inspector Tests](manual_gguf_inspect_tests.md) — inspecting GGUF model files
