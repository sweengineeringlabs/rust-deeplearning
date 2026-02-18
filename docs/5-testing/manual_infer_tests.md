# Manual Inference Tests

> **TLDR:** Manual test checklist for the `rustml-infer` CLI: load GGUF models, generate text, sampling options, and streaming.

**Audience**: Developers, QA

**WHAT**: Manual test procedures for the inference CLI tool
**WHY**: Validates end-to-end text generation from GGUF models — the primary user-facing workflow
**HOW**: Step-by-step test tables with expected outcomes

---

## Table of Contents

- [Help & Version](#1-help--version)
- [Basic Generation](#2-basic-generation)
- [Sampling Options](#3-sampling-options)
- [Streaming](#4-streaming)
- [Chat Mode](#5-chat-mode)
- [Stdin Input](#6-stdin-input)
- [Diagnostic Output](#7-diagnostic-output)
- [Error Cases](#8-error-cases)

---

> **Prerequisite**: A GGUF model file (e.g., Gemma 3 1B IT Q4_0). Release build recommended for inference speed: `cargo build --release -p rustml-nlp`.

## 1. Help & Version

| Test | Command | Expected |
|------|---------|----------|
| Help flag | `rustml-infer --help` | Lists `GGUF_PATH` positional, `--prompt`, `--max-tokens`, `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--stream`, `--chat` |
| Version flag | `rustml-infer --version` | Prints version string |
| No args | `rustml-infer` | Shows error and usage with `GGUF_PATH` |

## 2. Basic Generation

| Test | Command | Expected |
|------|---------|----------|
| Generate with prompt | `rustml-infer model.gguf --prompt "The capital of France is"` | Prints generated continuation (e.g., "Paris...") |
| Short output | `rustml-infer model.gguf --prompt "Hello" --max-tokens 10` | Output is at most ~10 tokens |
| Long output | `rustml-infer model.gguf --prompt "Once upon a time" --max-tokens 512` | Generates up to 512 tokens |
| Greedy decoding | `rustml-infer model.gguf --prompt "1+1=" --temperature 0.0` | Deterministic output (same result on repeated runs) |

## 3. Sampling Options

| Test | Command | Expected |
|------|---------|----------|
| Temperature | `rustml-infer model.gguf --prompt "Hello" --temperature 1.5` | Higher randomness in output |
| Top-k | `rustml-infer model.gguf --prompt "Hello" --top-k 10` | Generates text (constrained to top 10 tokens per step) |
| Top-p | `rustml-infer model.gguf --prompt "Hello" --top-p 0.9` | Generates text with nucleus sampling |
| Repetition penalty | `rustml-infer model.gguf --prompt "The cat sat on the" --repetition-penalty 1.2` | Less repetitive output compared to default |
| All sampling flags | `rustml-infer model.gguf --prompt "Hello" --temperature 0.7 --top-k 50 --top-p 0.9 --repetition-penalty 1.1 --max-tokens 64` | All flags accepted; generates text |

## 4. Streaming

| Test | Command | Expected |
|------|---------|----------|
| Stream mode | `rustml-infer model.gguf --prompt "Once upon" --stream --max-tokens 50` | Tokens printed incrementally as generated |
| Stream vs batch | Compare `--stream` vs without | Same final text content; stream prints progressively |

## 5. Chat Mode

| Test | Command | Expected |
|------|---------|----------|
| Chat template | `rustml-infer model.gguf --prompt "What is 2+2?" --chat --max-tokens 64` | Prompt wrapped in model's chat template; response is conversational |
| Chat + stream | `rustml-infer model.gguf --prompt "Hi there" --chat --stream --max-tokens 32` | Streaming works with chat mode |

## 6. Stdin Input

| Test | Command | Expected |
|------|---------|----------|
| Prompt from stdin | `echo "Hello world" \| rustml-infer model.gguf` | Reads prompt from stdin; prints `Reading prompt from stdin...` on stderr |
| Multiline stdin | `printf "Line one\nLine two" \| rustml-infer model.gguf --max-tokens 32` | Accepts multiline input |

## 7. Diagnostic Output

| Test | What to check | Expected |
|------|---------------|----------|
| Loading message | stderr during generation | `Loading GGUF: <path>` |
| GGUF version | stderr | `GGUF v3, N tensors` |
| Config summary | stderr | `arch=<arch>, dim=<N>, layers=<N>, heads=<N>, vocab=<N>` |
| Tokenizer info | stderr | `Tokenizer: <N> tokens` |
| Tensor loading | stderr | `Loading tensors...` then `<N> tensors loaded` |
| Model ready | stderr | `Model ready: <N>M params` |
| Separator | stderr | `---` before generation output |

## 8. Error Cases

| Test | Command | Expected |
|------|---------|----------|
| Nonexistent file | `rustml-infer /nonexistent.gguf --prompt "Hello"` | Error: `Failed to parse GGUF` |
| Invalid GGUF | `rustml-infer /tmp/not_a_gguf.bin --prompt "Hello"` | Error: invalid GGUF |
| Bad temperature type | `rustml-infer model.gguf --temperature abc` | Error: invalid value for `--temperature` |
| Bad max-tokens type | `rustml-infer model.gguf --max-tokens xyz` | Error: invalid value for `--max-tokens` |
| Unknown flag | `rustml-infer model.gguf --unknown-flag` | Error: unexpected argument |

---

## See Also

- [Manual Testing Hub](manual_testing.md) — prerequisites and setup
- [Manual Tokenizer Tests](manual_tokenizer_tests.md) — tokenizer CLI tests
- [Manual GGUF Inspector Tests](manual_gguf_inspect_tests.md) — inspecting GGUF model files
- [Manual Hub CLI Tests](manual_hub_cli_tests.md) — model download and cache management
