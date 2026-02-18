# Manual SweAI Unified CLI Tests

> **TLDR:** Manual test checklist for the `sweai` unified CLI: top-level help, subcommand dispatch, parity with standalone binaries, and error handling.

**Audience**: Developers, QA

**WHAT**: Manual test procedures for the unified `sweai` CLI binary
**WHY**: Validates that the facade binary correctly dispatches to all four underlying tools and produces identical results to standalone binaries
**HOW**: Step-by-step test tables with expected outcomes

---

## Table of Contents

- [Help & Version](#1-help--version)
- [Infer Subcommand](#2-infer-subcommand)
- [GGUF Subcommand](#3-gguf-subcommand)
- [Hub Subcommand](#4-hub-subcommand)
- [Tokenizer Subcommand](#5-tokenizer-subcommand)
- [Parity with Standalone Binaries](#6-parity-with-standalone-binaries)
- [Error Cases](#7-error-cases)

---

> **Prerequisite**: Build the unified binary: `cargo build -p rustml-cli` (or `cargo build --release -p rustml-cli` for inference).
>
> **Gated models**: Gated models (e.g., `google/gemma-3-1b-it`) require a HuggingFace token. Set `export HF_TOKEN=hf_xxx` before running, or use `~/.cache/huggingface/token`. See [Manual Testing Hub](manual_testing.md#huggingface-token-setup) for full token precedence.
>
> **Standalone equivalents**: Each `sweai` subcommand maps 1-to-1 with a standalone binary — see [Manual Testing Hub](manual_testing.md) for the full equivalence table.

## 1. Help & Version

| Test | Command | Expected |
|------|---------|----------|
| Top-level help | `sweai --help` | Lists `infer`, `gguf`, `hub`, `tokenizer` subcommands with descriptions |
| Version flag | `sweai --version` | Prints version string matching workspace version |
| No args | `sweai` | Shows error and usage listing all subcommands |
| Infer help | `sweai infer --help` | Lists `GGUF_PATH` positional, `--safetensors`, `--prompt`, `--batch-file`, `--max-tokens`, `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--stream`, `--chat`, `--timeout` |
| GGUF help | `sweai gguf --help` | Lists `info`, `meta`, `tensors`, `verify` subcommands |
| Hub help | `sweai hub --help` | Lists `download`, `list`, `info` subcommands and `--cache-dir`, `--token` flags |
| Tokenizer help | `sweai tokenizer --help` | Lists `--gguf`, `--hf`, `--bpe`, `--byte` backends and `encode`, `decode`, `info` subcommands |

## 2. Infer Subcommand

> **Prerequisite**: A GGUF model file (e.g., Gemma 3 1B IT Q4_0). Release build recommended: `cargo build --release -p rustml-cli`.

| Test | Command | Expected |
|------|---------|----------|
| Basic generation | `sweai infer model.gguf --prompt "The capital of France is"` | Prints generated continuation; stderr shows loading diagnostics |
| Max tokens | `sweai infer model.gguf --prompt "Hello" --max-tokens 10` | Output is at most ~10 tokens |
| Greedy decoding | `sweai infer model.gguf --prompt "1+1=" --temperature 0.0` | Deterministic output |
| Sampling flags | `sweai infer model.gguf --prompt "Hello" --temperature 0.7 --top-k 50 --top-p 0.9 --repetition-penalty 1.1 --max-tokens 32` | All flags accepted; generates text |
| Streaming | `sweai infer model.gguf --prompt "Once upon" --stream --max-tokens 50` | Tokens printed incrementally |
| Chat mode | `sweai infer model.gguf --prompt "What is 2+2?" --chat --max-tokens 64` | Prompt wrapped in chat template |
| Batch file | `sweai infer model.gguf --batch-file /tmp/prompts.txt --max-tokens 64` | Outputs `[0] ...`, `[1] ...` per prompt |
| Timeout | `sweai infer model.gguf --prompt "Tell me a long story" --max-tokens 4096 --timeout 5` | Generation stops after ~5s |
| Stdin prompt | `echo "Hello world" \| sweai infer model.gguf` | Reads prompt from stdin; stderr shows `Reading prompt from stdin...` |
| SafeTensors basic | `sweai infer --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 10` | Loads GPT-2; stderr: `Config: arch=gpt2, ...`, `Building model...`, `Tokenizer: <N> tokens (BPE)` |
| SafeTensors streaming | `sweai infer --safetensors openai-community/gpt2 --prompt "The quick brown fox" --stream --max-tokens 20` | Tokens printed incrementally |
| SafeTensors greedy | `sweai infer --safetensors openai-community/gpt2 --prompt "The capital of France is" --temperature 0 --max-tokens 20` | Deterministic output |
| SafeTensors sampling | `sweai infer --safetensors openai-community/gpt2 --prompt "Once upon" --temperature 0.7 --top-k 50 --top-p 0.9 --max-tokens 32` | All flags accepted; generates text |
| SafeTensors Gemma 3 | `sweai infer --safetensors google/gemma-3-1b-it --prompt "The capital of France is" --max-tokens 20 --stream` | Auto-detects `arch=gemma3`; stderr: `Tokenizer: <N> tokens (tokenizer.json)`; generates text |
| SafeTensors Gemma 3 chat | `sweai infer --safetensors google/gemma-3-1b-it --prompt "What is 2+2?" --chat --max-tokens 32` | Uses chat template from config; generates conversational response |
| SafeTensors bad model | `sweai infer --safetensors nonexistent/model-xyz --prompt "Hi"` | Error: `Failed to download model` |

## 3. GGUF Subcommand

> **Prerequisite**: A GGUF model file.

| Test | Command | Expected |
|------|---------|----------|
| Info | `sweai gguf info model.gguf` | Shows GGUF version, tensor count, architecture, dimensions, vocab size |
| Meta all keys | `sweai gguf meta model.gguf` | Prints all metadata key-value pairs in alphabetical order |
| Meta filter key | `sweai gguf meta model.gguf --key general.architecture` | Single line: `general.architecture = "<arch>"` |
| Tensors list | `sweai gguf tensors model.gguf` | Lists all tensors with name, dtype, shape, offset |
| Tensors filter | `sweai gguf tensors model.gguf --filter attn_q` | Only attention Q tensors listed |
| Tensor stats | `sweai gguf tensors model.gguf --filter token_embd --stats` | Shows `stats: min=... max=... mean=... std=... n=...` |
| Tensor head | `sweai gguf tensors model.gguf --filter token_embd --head 5` | Shows `head(5): [v0, v1, ..., v4]` |
| Verify | `sweai gguf verify model.gguf` | PASS for all checks, exit code 0 |

## 4. Hub Subcommand

> **Prerequisite**: Internet access for download tests. A HuggingFace token for gated/private model tests (see [token setup](manual_testing.md#huggingface-token-setup)).

| Test | Command | Expected |
|------|---------|----------|
| Download SafeTensors | `sweai hub download openai-community/gpt2` | Downloads model files; prints model directory path |
| Download GGUF | `sweai hub download <repo-id> --gguf <filename>.gguf` | Downloads single GGUF file; prints file path |
| List shows GGUF | `sweai hub list` (after GGUF download) | Lists the GGUF repo ID alongside SafeTensors models |
| List cached | `sweai hub list` | Shows model IDs from cache directories |
| Show config | `sweai hub info openai-community/gpt2` | Pretty-printed JSON with model config |
| Custom cache dir | `sweai hub --cache-dir /tmp/sweai-cache list` | Uses specified cache directory |
| Token flag | `sweai hub --token hf_xxx list` | Accepts token (overrides `HF_TOKEN` env var and `~/.cache/huggingface/token`) |

## 5. Tokenizer Subcommand

| Test | Command | Expected |
|------|---------|----------|
| Byte encode | `sweai tokenizer --byte encode "Hello"` | `72 101 108 108 111` |
| Byte encode JSON | `sweai tokenizer --byte encode --json "Hello"` | `[72, 101, 108, 108, 111]` |
| Byte decode | `sweai tokenizer --byte decode 72 101 108 108 111` | `Hello` |
| Byte info | `sweai tokenizer --byte info` | `Vocab size: 256` |
| GGUF encode | `sweai tokenizer --gguf model.gguf encode "Hello world"` | Prints token IDs |
| HF encode | `sweai tokenizer --hf /path/to/tokenizer.json encode "Hello"` | Prints token IDs |
| BPE encode | `sweai tokenizer --bpe vocab.json merges.txt encode "Hello"` | Prints GPT-2 token IDs |
| Stdin input | `echo "OK" \| sweai tokenizer --byte encode` | `79 75` |

## 6. Parity with Standalone Binaries

> These tests verify that `sweai` produces identical output to standalone binaries. Run each pair and diff stdout/stderr.

| Test | Standalone command | Unified command | Check |
|------|-------------------|-----------------|-------|
| Infer parity | `rustml-infer model.gguf --prompt "Hi" --temperature 0 --max-tokens 16` | `sweai infer model.gguf --prompt "Hi" --temperature 0 --max-tokens 16` | stdout identical (deterministic with temp 0) |
| GGUF info parity | `rustml-gguf-inspect info model.gguf` | `sweai gguf info model.gguf` | stdout identical |
| GGUF meta parity | `rustml-gguf-inspect meta model.gguf` | `sweai gguf meta model.gguf` | stdout identical |
| GGUF tensors parity | `rustml-gguf-inspect tensors model.gguf` | `sweai gguf tensors model.gguf` | stdout identical |
| GGUF verify parity | `rustml-gguf-inspect verify model.gguf` | `sweai gguf verify model.gguf` | stdout identical, same exit code |
| Hub list parity | `rustml-hub-cli list` | `sweai hub list` | stdout identical |
| Tokenizer parity | `rustml-tokenizer --byte encode "Hello"` | `sweai tokenizer --byte encode "Hello"` | stdout identical: `72 101 108 108 111` |
| SafeTensors infer parity | `rustml-infer --safetensors openai-community/gpt2 --prompt "Hi" --temperature 0 --max-tokens 10` | `sweai infer --safetensors openai-community/gpt2 --prompt "Hi" --temperature 0 --max-tokens 10` | stdout identical (deterministic with temp 0) |
| Version match | `rustml-infer --version` vs `sweai --version` | Both print same workspace version | version strings match |

## 7. Error Cases

| Test | Command | Expected |
|------|---------|----------|
| No subcommand | `sweai` | Error: shows usage with all subcommands |
| Unknown subcommand | `sweai unknown` | Error: unrecognized subcommand `unknown` |
| Unknown flag | `sweai --unknown-flag` | Error: unexpected argument |
| Infer bad temperature | `sweai infer model.gguf --prompt "Hi" --temperature -1` | Error: `--temperature must be >= 0.0` |
| Infer top-k zero | `sweai infer model.gguf --prompt "Hi" --top-k 0` | Error: `--top-k must be > 0` |
| Infer top-p range | `sweai infer model.gguf --prompt "Hi" --top-p 1.5` | Error: `--top-p must be in (0.0, 1.0]` |
| Infer no model | `sweai infer --prompt "Hi"` | Error: `Provide a GGUF model path or --safetensors <MODEL_ID>` |
| Infer bad file | `sweai infer /nonexistent.gguf --prompt "Hi"` | Error: `Failed to parse GGUF` |
| Infer stream + batch | `sweai infer model.gguf --batch-file f.txt --stream` | Error: `--stream is not supported with --batch-file` |
| Infer timeout zero | `sweai infer model.gguf --prompt "Hi" --timeout 0` | Error: `--timeout must be > 0.0` |
| GGUF bad file | `sweai gguf info /nonexistent.gguf` | Error: `Failed to parse GGUF` |
| Hub uncached info | `sweai hub info nonexistent/model` | Error: model not cached |
| Tokenizer no backend | `sweai tokenizer encode "hello"` | Error: no backend specified |
| Tokenizer bad GGUF | `sweai tokenizer --gguf /nonexistent.gguf encode "hello"` | Error: failed to parse GGUF |
| Meta missing key | `sweai gguf meta model.gguf --key nonexistent.key` | Stderr: `Key not found: nonexistent.key` |

---

## See Also

- [Manual Testing Hub](manual_testing.md) — prerequisites, build instructions, and equivalence table
- [Manual Inference Tests](manual_infer_tests.md) — full `rustml-infer` test coverage
- [Manual GGUF Inspector Tests](manual_gguf_inspect_tests.md) — full `rustml-gguf-inspect` test coverage
- [Manual Hub CLI Tests](manual_hub_cli_tests.md) — full `rustml-hub-cli` test coverage
- [Manual Tokenizer Tests](manual_tokenizer_tests.md) — full `rustml-tokenizer` test coverage
