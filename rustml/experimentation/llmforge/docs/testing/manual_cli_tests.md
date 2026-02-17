# Manual CLI Tests

> **TLDR:** Manual test checklist for the llmforge CLI: help output, argument parsing, run/info/download commands.

**Audience**: Developers, QA

**WHAT**: Manual test procedures for the llmforge CLI binary and all subcommands
**WHY**: Validates argument parsing, error messages, and command dispatch
**HOW**: Step-by-step test tables with expected outcomes

---

## Table of Contents

- [Top-level help](#1-top-level-help)
- [run help and flags](#2-run-help-and-flags)
- [info help and flags](#3-info-help-and-flags)
- [download help and flags](#4-download-help-and-flags)
- [run — SafeTensors / GPT-2](#5-run--safetensors--gpt-2)
- [run — GGUF / Llama](#6-run--gguf--llama)
- [run — local files](#7-run--local-files)
- [info command](#8-info-command)
- [download command](#9-download-command)
- [error handling](#10-error-handling)

---

## 1. Top-level help

| Test | Command | Expected |
|------|---------|----------|
| Help flag | `llmforge --help` | Lists commands: run, info, download, help |
| Help short | `llmforge -h` | Same output as `--help` |
| No args | `llmforge` | Prints error asking for subcommand, shows usage |
| Unknown subcommand | `llmforge foo` | Prints error, suggests valid subcommands |

## 2. run help and flags

| Test | Command | Expected |
|------|---------|----------|
| Run help | `llmforge run --help` | Shows all flags: --model, --file, --tokenizer, --config, --arch, --prompt, --interactive, --temperature, --top-k, --top-p, --repetition-penalty, --max-tokens, --threads |
| Model required | `llmforge run` | Error: `--model <MODEL>` is required |
| Default temperature | `llmforge run --help` | Shows `[default: 0.8]` for --temperature |
| Default top-k | `llmforge run --help` | Shows `[default: 40]` for --top-k |
| Default top-p | `llmforge run --help` | Shows `[default: 0.95]` for --top-p |
| Default rep penalty | `llmforge run --help` | Shows `[default: 1.0]` for --repetition-penalty |
| Default max-tokens | `llmforge run --help` | Shows `[default: 128]` for --max-tokens |
| Default threads | `llmforge run --help` | Shows `[default: 0]` for --threads |
| Arch values | `llmforge run --help` | Shows `[possible values: auto, gpt2, llama]` for --arch |

## 3. info help and flags

| Test | Command | Expected |
|------|---------|----------|
| Info help | `llmforge info --help` | Shows --model (required) and --file (optional) |
| Model required | `llmforge info` | Error: `--model <MODEL>` is required |

## 4. download help and flags

| Test | Command | Expected |
|------|---------|----------|
| Download help | `llmforge download --help` | Shows `<REPO>` positional arg and `--files` option |
| Repo required | `llmforge download` | Error: `<REPO>` is required |

## 5. run — SafeTensors / GPT-2

> Requires internet for first run (downloads ~500 MB). Subsequent runs use HuggingFace cache.

| Test | Command | Expected |
|------|---------|----------|
| GPT-2 single-shot | `llmforge run --model openai-community/gpt2 --prompt "Hello" --max-tokens 16` | Downloads model files, prints model stats to stderr, prints generated text to stdout |
| GPT-2 auto-arch | `llmforge run --model openai-community/gpt2 --prompt "Once upon" --max-tokens 8` | Stderr shows `Architecture: gpt2` (auto-detected from config.json) |
| GPT-2 explicit arch | `llmforge run --model openai-community/gpt2 --arch gpt2 --prompt "Test" --max-tokens 8` | Works identically to auto-detection |
| GPT-2 model stats | `llmforge run --model openai-community/gpt2 --prompt "x" --max-tokens 1 2>&1 1>/dev/null` | Shows Parameters ~124.4M, Dimension 768, Layers 12, Heads 12 |
| GPT-2 token count | `llmforge run --model openai-community/gpt2 --prompt "x" --max-tokens 16 2>&1 \| grep tok/s` | Shows `[16 tokens in X.XXs — Y.Y tok/s]` |
| GPT-2 temp=0 | `llmforge run --model openai-community/gpt2 --prompt "The" --max-tokens 16 --temperature 0` | Deterministic output (greedy decoding) |

## 6. run — GGUF / Llama

> Requires internet for first run (downloads ~670 MB GGUF + tokenizer). Subsequent runs use HuggingFace cache.

| Test | Command | Expected |
|------|---------|----------|
| TinyLlama single-shot | `llmforge run --model TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --file tinyllama-1.1b-chat-v1.0.Q4_0.gguf --prompt "Hello" --max-tokens 16` | Downloads GGUF + tokenizer, prints quantization breakdown, generates text |
| GGUF quant stats | (same as above, check stderr) | Shows `Quantization Breakdown` with Q4_0 tensor count |
| GGUF model stats | (same as above, check stderr) | Shows Parameters ~1.1B range, Layers 22, Heads 32 |
| GGUF missing --file | `llmforge run --model TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --prompt "Hi"` | Attempts to download `model.safetensors` (the default), which fails with a clear error |

## 7. run — local files

| Test | Steps | Expected |
|------|-------|----------|
| Local GGUF | Download a .gguf file, then `llmforge run --model ./path/to/model.gguf --prompt "Hi" --max-tokens 8` | Loads directly without HuggingFace API, generates text |
| Local SafeTensors | Have model.safetensors + config.json + tokenizer.json in a directory, then `llmforge run --model ./dir/ --prompt "Hi" --max-tokens 8` | Detects format from directory contents, loads model |
| Local with overrides | `llmforge run --model ./model.safetensors --config ./config.json --tokenizer ./tokenizer.json --arch gpt2 --prompt "Hi" --max-tokens 8` | Uses explicit paths instead of auto-detection |
| Sibling auto-detect | Place config.json and tokenizer.json next to model.safetensors, then `llmforge run --model ./model.safetensors --prompt "Hi" --max-tokens 8` | Auto-detects sibling config and tokenizer |

## 8. info command

| Test | Command | Expected |
|------|---------|----------|
| Info GGUF (HF) | `llmforge info --model TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --file tinyllama-1.1b-chat-v1.0.Q4_0.gguf` | Downloads file, shows GGUF version, metadata, model config, tensor list, quant breakdown |
| Info SafeTensors (HF) | `llmforge info --model openai-community/gpt2` | Downloads file, shows tensor names, shapes, dtypes, total parameter count |
| Info local GGUF | `llmforge info --model ./path/to/model.gguf` | Shows metadata, config, tensors without any download |
| Info metadata keys | (GGUF info output) | Metadata section lists keys like `llama.embedding_length`, `llama.block_count`, etc. sorted alphabetically |
| Info tensor list | (GGUF info output) | Each tensor shows name, GGML type, dimensions |
| Info quant breakdown | (GGUF info output) | Shows count of tensors per quantization type (Q4_0, F32, etc.) |

## 9. download command

| Test | Command | Expected |
|------|---------|----------|
| Download GPT-2 | `llmforge download openai-community/gpt2` | Downloads config.json, tokenizer.json, model.safetensors; prints paths |
| Download specific files | `llmforge download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --files tinyllama-1.1b-chat-v1.0.Q4_0.gguf` | Downloads only the specified GGUF file |
| Download already cached | Run `llmforge download openai-community/gpt2` twice | Second run completes instantly (files cached) |
| Download nonexistent repo | `llmforge download fake-org/no-such-model` | Prints download failure for each file, does not crash |
| Download nonexistent file | `llmforge download openai-community/gpt2 --files no_such_file.bin` | Prints `FAILED` for missing file, exits cleanly |

## 10. Error handling

| Test | Command | Expected |
|------|---------|----------|
| Bad model path | `llmforge run --model /no/such/file.gguf` | Error: not a file, directory, or HuggingFace repo ID |
| Bad extension | `llmforge run --model ./readme.txt` | Error: cannot detect format from extension |
| Empty directory | `llmforge run --model /tmp/empty_dir/` (empty dir) | Error: no .gguf or model.safetensors found |
| Missing config for ST | `llmforge run --model ./model.safetensors` (no sibling config.json) | Error: SafeTensors model requires config.json |
| Invalid arch override | `llmforge run --model openai-community/gpt2 --arch llama --prompt "x" --max-tokens 1` | Error during config parsing or model building (mismatched architecture) |

---

## See Also

- [Manual Testing Hub](manual_testing.md) — prerequisites and setup
- [Manual Launcher Tests](manual_launcher_tests.md) — llmf script tests
- [Manual Model Tests](manual_model_tests.md) — model source resolution details
- [Manual Inference Tests](manual_inference_tests.md) — generation and sampling tests
