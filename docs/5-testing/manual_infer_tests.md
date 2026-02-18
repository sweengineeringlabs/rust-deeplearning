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
- [Batch File](#7-batch-file)
- [Timeout](#8-timeout)
- [Diagnostic Output](#9-diagnostic-output)
- [Error Cases](#10-error-cases)

---

> **Prerequisite**: A GGUF model file (e.g., Gemma 3 1B IT Q4_0). Release build recommended for inference speed: `cargo build --release -p rustml-nlp`.
>
> **Unified CLI**: All `rustml-infer` commands below can also be run as `sweai infer` (e.g., `sweai infer model.gguf --prompt "Hello"`). Build with `cargo build --release -p rustml-cli`.

## 1. Help & Version

| Test | Command | Expected |
|------|---------|----------|
| Help flag | `rustml-infer --help` | Lists `GGUF_PATH` positional, `--prompt`, `--batch-file`, `--max-tokens`, `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--stream`, `--chat`, `--timeout` |
| Version flag | `rustml-infer --version` | Prints version string |
| No args | `rustml-infer` | Shows error and usage with `GGUF_PATH` |
| Unified help | `sweai infer --help` | Same flags as `rustml-infer --help` |

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

## 7. Batch File

> **Setup:** Create a prompts file with one prompt per line:
> ```
> echo "What is Rust?" > /tmp/prompts.txt
> echo "Explain transformers" >> /tmp/prompts.txt
> echo "What is GGUF?" >> /tmp/prompts.txt
> ```

| Test | Command | Expected |
|------|---------|----------|
| Batch generation | `rustml-infer model.gguf --batch-file /tmp/prompts.txt --max-tokens 64` | Outputs `[0] ...`, `[1] ...`, `[2] ...` with one response per prompt |
| Batch prompt count | (same as above) | stderr shows `Batch: 3 prompts` and `3 prompts in Xs (Y prompts/sec)` |
| Batch with sampling | `rustml-infer model.gguf --batch-file /tmp/prompts.txt --temperature 0.5 --top-k 20 --max-tokens 32` | Sampling flags applied to all prompts |
| Batch conflicts with prompt | `rustml-infer model.gguf --batch-file /tmp/prompts.txt --prompt "Hello"` | Error: `--prompt` cannot be used with `--batch-file` |
| Batch conflicts with stream | `rustml-infer model.gguf --batch-file /tmp/prompts.txt --stream` | Error: `--stream is not supported with --batch-file` |
| Empty batch file | `touch /tmp/empty.txt && rustml-infer model.gguf --batch-file /tmp/empty.txt` | Error: `Batch file is empty` |
| Nonexistent batch file | `rustml-infer model.gguf --batch-file /nonexistent.txt` | Error: `Failed to read batch file` |
| Blank lines skipped | File with blank lines between prompts | Only non-empty lines treated as prompts |

## 8. Timeout

| Test | Command | Expected |
|------|---------|----------|
| Timeout flag | `rustml-infer model.gguf --prompt "Tell me a long story" --max-tokens 4096 --timeout 5` | Generation stops after ~5s; stderr shows `Timeout: 5.0s` |
| Timeout with stream | `rustml-infer model.gguf --prompt "Tell me a story" --stream --timeout 3` | Streaming stops after ~3s |
| Zero timeout rejected | `rustml-infer model.gguf --prompt "Hello" --timeout 0` | Error: `--timeout must be > 0.0` |
| Negative timeout rejected | `rustml-infer model.gguf --prompt "Hello" --timeout -1` | Error: `--timeout must be > 0.0` |
| No timeout (default) | `rustml-infer model.gguf --prompt "Hello" --max-tokens 32` | No `Timeout:` line in stderr; runs to completion |

## 9. Diagnostic Output

| Test | What to check | Expected |
|------|---------------|----------|
| Loading message | stderr during generation | `Loading GGUF: <path>` |
| GGUF version | stderr | `GGUF v3, N tensors` |
| Config summary | stderr | `arch=<arch>, dim=<N>, layers=<N>, heads=<N>, vocab=<N>` |
| Tokenizer info | stderr | `Tokenizer: <N> tokens` |
| Tensor loading | stderr | `Loading tensors...` then `<N> tensors loaded` |
| Model ready | stderr | `Model ready: <N>M params` |
| KV cache memory | stderr | `KV cache: <N> MB (<layers>layers x <heads>heads x <seq>seq x <dim>dim x f32 x 2)` |
| Timeout (if set) | stderr | `Timeout: <N>s` |
| Separator | stderr | `---` before and after generation output |
| Metrics (stream) | stderr after streaming | `<N> tokens in <T>s (<R> tokens/sec)` |
| Metrics (non-stream) | stderr after generation | `Generated in <T>s` |
| Metrics (batch) | stderr after batch | `<N> prompts in <T>s (<R> prompts/sec)` |

## 10. Error Cases

| Test | Command | Expected |
|------|---------|----------|
| Nonexistent file | `rustml-infer /nonexistent.gguf --prompt "Hello"` | Error: `Failed to parse GGUF` |
| Invalid GGUF | `rustml-infer /tmp/not_a_gguf.bin --prompt "Hello"` | Error: invalid GGUF |
| Bad temperature type | `rustml-infer model.gguf --temperature abc` | Error: invalid value for `--temperature` |
| Bad max-tokens type | `rustml-infer model.gguf --max-tokens xyz` | Error: invalid value for `--max-tokens` |
| Unknown flag | `rustml-infer model.gguf --unknown-flag` | Error: unexpected argument |
| Negative temperature | `rustml-infer model.gguf --prompt "Hi" --temperature -1` | Error: `--temperature must be >= 0.0` |
| Top-k zero | `rustml-infer model.gguf --prompt "Hi" --top-k 0` | Error: `--top-k must be > 0` |
| Top-p out of range | `rustml-infer model.gguf --prompt "Hi" --top-p 1.5` | Error: `--top-p must be in (0.0, 1.0]` |
| Repetition penalty zero | `rustml-infer model.gguf --prompt "Hi" --repetition-penalty 0` | Error: `--repetition-penalty must be > 0.0` |

---

## See Also

- [Manual Testing Hub](manual_testing.md) — prerequisites and setup
- [Manual Tokenizer Tests](manual_tokenizer_tests.md) — tokenizer CLI tests
- [Manual GGUF Inspector Tests](manual_gguf_inspect_tests.md) — inspecting GGUF model files
- [Manual Hub CLI Tests](manual_hub_cli_tests.md) — model download and cache management
