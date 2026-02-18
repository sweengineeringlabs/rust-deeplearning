# Manual GGUF Inspector Tests

> **TLDR:** Manual test checklist for the `rustml-gguf-inspect` CLI: model info, metadata dump, tensor listing, tensor stats/head inspection, and verify checks.

**Audience**: Developers, QA

**WHAT**: Manual test procedures for the GGUF inspector CLI tool
**WHY**: Validates correct parsing and display of GGUF model files — the primary format for quantized models
**HOW**: Step-by-step test tables with expected outcomes

---

## Table of Contents

- [Help & Version](#1-help--version)
- [Info Subcommand](#2-info-subcommand)
- [Meta Subcommand](#3-meta-subcommand)
- [Meta Filtering](#4-meta-filtering)
- [Tensors Subcommand](#5-tensors-subcommand)
- [Tensor Filtering](#6-tensor-filtering)
- [Tensor Stats](#7-tensor-stats)
- [Tensor Head](#8-tensor-head)
- [Verify Subcommand](#9-verify-subcommand)
- [Value Formatting](#10-value-formatting)
- [Error Cases](#11-error-cases)

---

> **Prerequisite**: A GGUF model file. Examples below use a Gemma 3 1B IT Q4_0 model. Substitute your own model path.
>
> **Unified CLI**: All `rustml-gguf-inspect` commands below can also be run as `sweai gguf` (e.g., `sweai gguf info model.gguf`). Build with `cargo build -p rustml-cli`.

## 1. Help & Version

| Test | Command | Expected |
|------|---------|----------|
| Help flag | `rustml-gguf-inspect --help` | Lists `info`, `meta`, `tensors`, `verify` subcommands |
| Version flag | `rustml-gguf-inspect --version` | Prints version string |
| No args | `rustml-gguf-inspect` | Shows error and usage |
| Info help | `rustml-gguf-inspect info --help` | Shows PATH positional arg |
| Meta help | `rustml-gguf-inspect meta --help` | Shows `--key` option |
| Tensors help | `rustml-gguf-inspect tensors --help` | Shows `--filter`, `--stats`, `--head` options |
| Verify help | `rustml-gguf-inspect verify --help` | Shows PATH positional arg |
| Unified help | `sweai gguf --help` | Same subcommands as `rustml-gguf-inspect --help` |

## 2. Info Subcommand

| Test | Command | Expected |
|------|---------|----------|
| GGUF version | `rustml-gguf-inspect info model.gguf` | Shows `GGUF version:  3` |
| Tensor count | (same output) | Shows `Tensor count:` with correct number (e.g., 339 for Gemma 3 1B) |
| Architecture | (same output) | Shows `Architecture:  gemma3` (or `llama`, etc.) |
| Dimensions | (same output) | Shows `Dimensions:`, `Hidden dim:`, `Layers:`, `Heads:` |
| Vocab size | (same output) | Shows `Vocab size:` (e.g., 262144 for Gemma 3) |
| KV heads | (same output) | Shows `KV heads:` if GQA model |
| Max seq len | (same output) | Shows `Max seq len:` (e.g., 32768) |
| Head dim | (same output) | Shows `Head dim:` for Gemma 3 (256) |

## 3. Meta Subcommand

| Test | Command | Expected |
|------|---------|----------|
| All keys sorted | `rustml-gguf-inspect meta model.gguf` | Prints all metadata key-value pairs in alphabetical order |
| Architecture key | (in output) | `general.architecture = "gemma3"` |
| Name key | (in output) | `general.name = "..."` |
| Tokenizer keys | (in output) | `tokenizer.ggml.bos_token_id`, `tokenizer.ggml.eos_token_id` present |
| Model config keys | (in output) | `{arch}.embedding_length`, `{arch}.block_count`, etc. present |

## 4. Meta Filtering

| Test | Command | Expected |
|------|---------|----------|
| Filter by key | `rustml-gguf-inspect meta model.gguf --key general.architecture` | Single line: `general.architecture = "gemma3"` |
| Filter numeric key | `rustml-gguf-inspect meta model.gguf --key llama.block_count` | Single line with numeric value |
| Missing key | `rustml-gguf-inspect meta model.gguf --key nonexistent.key` | Stderr: `Key not found: nonexistent.key` |

## 5. Tensors Subcommand

| Test | Command | Expected |
|------|---------|----------|
| List all tensors | `rustml-gguf-inspect tensors model.gguf` | Lists all tensors with name, dtype, shape, offset |
| Tensor count | (stderr) | Shows `N tensor(s) listed` |
| Embedding tensor | (in output) | `token_embd.weight` with correct dtype and shape |
| Attention tensors | (in output) | `blk.0.attn_q.weight` etc. with Q4_0 or similar dtype |
| Output tensor | (in output) | `output.weight` present |

## 6. Tensor Filtering

| Test | Command | Expected |
|------|---------|----------|
| Filter by name | `rustml-gguf-inspect tensors model.gguf --filter attn_q` | Only attention Q tensors listed |
| Filter by layer | `rustml-gguf-inspect tensors model.gguf --filter blk.0.` | Only layer 0 tensors |
| Filter no match | `rustml-gguf-inspect tensors model.gguf --filter nonexistent` | No tensors; `0 tensor(s) listed` |

## 7. Tensor Stats

| Test | Command | Expected |
|------|---------|----------|
| Stats for single tensor | `rustml-gguf-inspect tensors model.gguf --filter token_embd --stats` | Shows `stats: min=... max=... mean=... std=... n=...` below tensor line |
| Stats for all tensors | `rustml-gguf-inspect tensors model.gguf --stats` | Each tensor followed by a stats line |
| Stats with filter | `rustml-gguf-inspect tensors model.gguf --filter attn_q --stats` | Only filtered tensors show stats |
| No stats without flag | `rustml-gguf-inspect tensors model.gguf --filter token_embd` | No `stats:` line in output |
| Stats + head combined | `rustml-gguf-inspect tensors model.gguf --filter token_embd --stats --head 5` | Both stats and head lines printed |

## 8. Tensor Head

| Test | Command | Expected |
|------|---------|----------|
| Head first N values | `rustml-gguf-inspect tensors model.gguf --filter token_embd --head 10` | Shows `head(10): [v0, v1, ..., v9]` with dequantized f32 values |
| Head larger than tensor | `rustml-gguf-inspect tensors model.gguf --filter some_small_tensor --head 999999` | Clamps to tensor size, prints all values |
| Head zero | `rustml-gguf-inspect tensors model.gguf --filter token_embd --head 0` | Shows `head(0): []` |
| Head without stats | `rustml-gguf-inspect tensors model.gguf --filter token_embd --head 5` | Shows head line but no stats line |

## 9. Verify Subcommand

| Test | Command | Expected |
|------|---------|----------|
| Valid model passes | `rustml-gguf-inspect verify model.gguf` | PASS for all checks, exit code 0 |
| token_embd.weight check | (in output) | `PASS  token_embd.weight exists` |
| Embedding shape check | (in output) | `PASS  token_embd.weight shape [DxV] matches dim=D` |
| Vocab size consistency | (in output) | `PASS  vocab_size=V matches token_embd.weight[1]` |
| output.weight check | (in output) | `PASS  output.weight exists` (or `WARN  output.weight missing` for tied-embedding models) |
| Layer tensor completeness | (in output) | `PASS  all layer tensors present for N layers (9 suffixes each)` |
| Embedding dim consistency | (in output) | `PASS  embedding dim consistent across attention projections` |
| Summary line | (in output) | `--- Summary: N PASS, 0 FAIL, M WARN ---` |
| Missing token_embd | Use a broken GGUF | `FAIL  token_embd.weight missing`, exit code 1 |
| Tied embeddings | Model without output.weight | `WARN  output.weight missing`, exit code 0 |

## 10. Value Formatting

| Test | What to check | Expected |
|------|---------------|----------|
| String values | Meta output for string keys | Quoted: `"value"` |
| Long strings | Meta output for chat_template | Truncated: `"first 200 chars..." (len=N)` |
| Numeric values | Meta output for U32/F32 keys | Displayed inline (e.g., `4096`, `10000`) |
| Boolean values | Meta output for bool keys | `true` or `false` |
| Array values | Meta output for token arrays | `[array, len=N]` |

## 11. Error Cases

| Test | Command | Expected |
|------|---------|----------|
| Nonexistent file | `rustml-gguf-inspect info /nonexistent.gguf` | Error: failed to parse GGUF |
| Invalid file | `rustml-gguf-inspect info /tmp/not_a_gguf.bin` | Error: invalid GGUF magic |
| Meta on bad file | `rustml-gguf-inspect meta /nonexistent.gguf` | Error |
| Tensors on bad file | `rustml-gguf-inspect tensors /nonexistent.gguf` | Error |
| Verify on bad file | `rustml-gguf-inspect verify /nonexistent.gguf` | Error: failed to parse GGUF |
| Stats on bad file | `rustml-gguf-inspect tensors /nonexistent.gguf --stats` | Error |

---

## See Also

- [Manual Testing Hub](manual_testing.md) — prerequisites and setup
- [Manual Tokenizer Tests](manual_tokenizer_tests.md) — tokenizer CLI tests
- [Manual Inference Tests](manual_infer_tests.md) — inference CLI tests
- [GGUF User-Defined Tokens](../0-ideation/research/gguf-user-defined-tokens.md) — GGUF format research
