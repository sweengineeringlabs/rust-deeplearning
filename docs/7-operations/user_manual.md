# SweAI User Manual

> **TLDR:** End-user guide for the `sweai` unified CLI — download models, run inference, inspect GGUF files, and tokenize text, all from a single binary.

**Audience**: End users, developers

**WHAT**: Complete usage guide for the SweAI CLI toolkit
**WHY**: Single reference for all CLI workflows — from first install to advanced generation options
**HOW**: Step-by-step instructions with copy-paste commands

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [Inference](#inference)
  - [SafeTensors Models](#safetensors-models)
  - [GGUF Models](#gguf-models)
  - [Streaming](#streaming)
  - [Chat Mode](#chat-mode)
  - [Interactive Mode](#interactive-mode)
  - [Sampling Options](#sampling-options)
  - [Batch Generation](#batch-generation)
  - [Timeout](#timeout)
- [Model Hub](#model-hub)
  - [Download Models](#download-models)
  - [List Cached Models](#list-cached-models)
  - [Inspect Model Config](#inspect-model-config)
- [GGUF Inspector](#gguf-inspector)
- [Tokenizer](#tokenizer)
- [Standalone Binaries](#standalone-binaries)
- [Supported Architectures](#supported-architectures)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

1. **Build the CLI** (release mode recommended for inference speed):

   ```bash
   cd /path/to/rust-deeplearning
   cargo build --release -p rustml-cli
   ```

   This produces the `sweai` binary at `./target/release/sweai`.

2. **Verify the install**:

   ```bash
   ./target/release/sweai --version
   ./target/release/sweai --help
   ```

> **Tip:** All commands below assume you are in the workspace root (`rust-deeplearning/`). If you add `./target/release` to your `PATH`, you can run `sweai` directly without the path prefix.

---

## Quick Start

Download a model and generate text in two commands:

```bash
# Download GPT-2 (no token needed)
./target/release/sweai hub download openai-community/gpt2

# Generate text
./target/release/sweai infer --safetensors openai-community/gpt2 \
  --prompt "The capital of France is" --max-tokens 20
```

The model is cached after the first download — subsequent runs load immediately without re-downloading.

---

## Authentication

Gated models (e.g., `google/gemma-3-1b-it`) require a HuggingFace token with accepted license terms.

**Token precedence** (highest to lowest):

| Priority | Source | How to set |
|----------|--------|------------|
| 1 | `--token` CLI flag | `sweai hub --token hf_xxx download ...` |
| 2 | `HF_TOKEN` environment variable | `export HF_TOKEN=hf_xxx` |
| 3 | Token file | `~/.cache/huggingface/token` (written by `huggingface-cli login`) |

**Set your token for gated model access:**

```bash
export HF_TOKEN=hf_YOUR_TOKEN_HERE
```

---

## Inference

### SafeTensors Models

Use `--safetensors <MODEL_ID>` with a HuggingFace model ID. The architecture is auto-detected from the model's `config.json`.

**Basic generation:**

```bash
./target/release/sweai infer --safetensors google/gemma-3-1b-it \
  --prompt "The capital of France is" --max-tokens 20
```

**Greedy (deterministic) decoding:**

```bash
./target/release/sweai infer --safetensors google/gemma-3-1b-it \
  --prompt "Hello" --temperature 0 --max-tokens 20
```

### GGUF Models

Pass a GGUF file path as the first positional argument. You provide **either** a `[GGUF_PATH]` **or** `--safetensors <MODEL_ID>`, not both:

```bash
./target/release/sweai infer /path/to/model.gguf \
  --prompt "The capital of France is" --max-tokens 20
```

**Where to find GGUF files:** Models downloaded via `sweai hub download` are cached under `~/.cache/huggingface/hub/`:

```bash
# Download a GGUF model
./target/release/sweai hub download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  --gguf tinyllama-1.1b-chat-v1.0.Q4_0.gguf

# The file is cached at:
# ~/.cache/huggingface/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/snapshots/<rev>/tinyllama-1.1b-chat-v1.0.Q4_0.gguf

# Run inference with the cached file
./target/release/sweai infer ~/.cache/huggingface/hub/models--TheBloke--TinyLlama-1.1B-Chat-v1.0-GGUF/snapshots/*/tinyllama-1.1b-chat-v1.0.Q4_0.gguf \
  --prompt "Hello" --max-tokens 20
```

You can also use any local GGUF file directly (e.g., `/mnt/c/models/model.gguf`). Override the cache directory with `HF_HOME` or `HF_HUB_CACHE` environment variables.

### Streaming

Add `--stream` to print tokens as they are generated instead of waiting for the full output:

```bash
./target/release/sweai infer --safetensors google/gemma-3-1b-it \
  --prompt "The capital of France is" --max-tokens 20 --stream
```

### Chat Mode

Add `--chat` to wrap the prompt in the model's chat template for conversational responses:

```bash
./target/release/sweai infer --safetensors google/gemma-3-1b-it \
  --prompt "What is 2+2?" --chat --max-tokens 32
```

Chat mode and streaming can be combined:

```bash
./target/release/sweai infer --safetensors google/gemma-3-1b-it \
  --prompt "Explain transformers briefly" --chat --stream --max-tokens 64
```

### Interactive Mode

Add `--interactive` to start a multi-turn REPL-style chat session. This implies `--chat` and `--stream` behavior:

```bash
./target/release/sweai infer --safetensors google/gemma-3-1b-it \
  --interactive --max-tokens 256
```

Or with a GGUF model:

```bash
./target/release/sweai infer /path/to/model.gguf \
  --interactive --max-tokens 256
```

**How it works:**

1. The CLI shows a `> ` prompt and waits for your input
2. Type a message and press Enter — the model streams its response token-by-token
3. The full conversation history is maintained across turns, so the model can reference prior messages
4. The entire conversation is re-encoded and re-prefilled each turn

**REPL commands:**

| Input | Effect |
|-------|--------|
| Any text + Enter | Send message, get response |
| Empty Enter | Skipped (no generation) |
| `quit` or `exit` | Exit the session |
| Ctrl-D | Exit the session (EOF) |
| `/clear` | Reset conversation history |

**Example session:**

```
> What is the capital of France?
Paris is the capital of France.
  [12 tokens in 2.45s (4.9 tok/s)]
> What about Germany?
The capital of Germany is Berlin.
  [10 tokens in 1.98s (5.1 tok/s)]
> /clear
[History cleared]
> quit
Goodbye!
```

**Sampling options work with interactive mode:**

```bash
./target/release/sweai infer --safetensors google/gemma-3-1b-it \
  --interactive --temperature 0.5 --max-tokens 128
```

> **Note:** `--interactive` cannot be used with `--prompt` or `--batch-file`.

### Sampling Options

Control text generation with these flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--max-tokens N` | 256 | Maximum number of tokens to generate |
| `--temperature F` | 0.8 | Randomness (0 = deterministic, higher = more random) |
| `--top-k N` | (off) | Sample from top N most likely tokens |
| `--top-p F` | (off) | Nucleus sampling threshold (0.0–1.0] |
| `--repetition-penalty F` | (off) | Penalize repeated tokens (> 1.0 reduces repetition) |

**Example with all sampling options:**

```bash
./target/release/sweai infer --safetensors google/gemma-3-1b-it \
  --prompt "Once upon a time" \
  --temperature 0.7 --top-k 50 --top-p 0.9 \
  --repetition-penalty 1.1 --max-tokens 64
```

### Batch Generation

Process multiple prompts from a file (one prompt per line):

```bash
# Create a prompts file
echo "What is Rust?" > /tmp/prompts.txt
echo "Explain transformers" >> /tmp/prompts.txt
echo "What is GGUF?" >> /tmp/prompts.txt

# Run batch inference
./target/release/sweai infer --safetensors openai-community/gpt2 \
  --batch-file /tmp/prompts.txt --max-tokens 64
```

Output is indexed: `[0] ...`, `[1] ...`, `[2] ...`.

> **Note:** `--stream` cannot be used with `--batch-file`.

### Timeout

Limit generation time with `--timeout` (in seconds):

```bash
./target/release/sweai infer --safetensors google/gemma-3-1b-it \
  --prompt "Tell me a long story" --max-tokens 4096 --timeout 5
```

### Stdin Input

Pipe a prompt via stdin instead of using `--prompt`:

```bash
echo "Hello world" | ./target/release/sweai infer --safetensors openai-community/gpt2 --max-tokens 10
```

---

## Model Hub

Manage HuggingFace model downloads and cache.

### Download Models

**SafeTensors models:**

```bash
./target/release/sweai hub download openai-community/gpt2
./target/release/sweai hub download google/gemma-3-1b-it   # requires HF_TOKEN
```

**GGUF models:**

```bash
./target/release/sweai hub download <repo-id> --gguf <filename>.gguf
```

**Custom cache directory:**

```bash
./target/release/sweai hub --cache-dir /tmp/sweai-cache download openai-community/gpt2
```

### List Cached Models

```bash
./target/release/sweai hub list
```

### Inspect Model Config

View the JSON config of a cached model:

```bash
./target/release/sweai hub info openai-community/gpt2
```

---

## GGUF Inspector

Inspect GGUF model files without running inference.

```bash
# Overview: version, tensor count, architecture, dimensions, vocab size
./target/release/sweai gguf info model.gguf

# All metadata key-value pairs
./target/release/sweai gguf meta model.gguf

# Single metadata key
./target/release/sweai gguf meta model.gguf --key general.architecture

# List all tensors (name, dtype, shape, offset)
./target/release/sweai gguf tensors model.gguf

# Filter tensors by name
./target/release/sweai gguf tensors model.gguf --filter attn_q

# Tensor statistics
./target/release/sweai gguf tensors model.gguf --filter token_embd --stats

# First N values of a tensor
./target/release/sweai gguf tensors model.gguf --filter token_embd --head 5

# Verify file integrity
./target/release/sweai gguf verify model.gguf
```

---

## Tokenizer

Encode and decode text with multiple tokenizer backends.

**Byte tokenizer** (raw UTF-8 bytes):

```bash
./target/release/sweai tokenizer --byte encode "Hello"
# Output: 72 101 108 108 111

./target/release/sweai tokenizer --byte decode 72 101 108 108 111
# Output: Hello

./target/release/sweai tokenizer --byte info
# Output: Vocab size: 256
```

**JSON output:**

```bash
./target/release/sweai tokenizer --byte encode --json "Hello"
# Output: [72, 101, 108, 108, 111]
```

**GGUF tokenizer:**

```bash
./target/release/sweai tokenizer --gguf model.gguf encode "Hello world"
```

**HuggingFace tokenizer** (tokenizer.json):

```bash
./target/release/sweai tokenizer --hf /path/to/tokenizer.json encode "Hello"
```

**BPE tokenizer** (GPT-2 style):

```bash
./target/release/sweai tokenizer --bpe vocab.json merges.txt encode "Hello"
```

**Stdin input:**

```bash
echo "OK" | ./target/release/sweai tokenizer --byte encode
# Output: 79 75
```

---

## Standalone Binaries

Each `sweai` subcommand has an equivalent standalone binary:

| Unified command | Standalone binary | Build package |
|-----------------|-------------------|---------------|
| `sweai infer` | `rustml-infer` | `rustml-nlp` |
| `sweai gguf` | `rustml-gguf-inspect` | `rustml-gguf` |
| `sweai hub` | `rustml-hub-cli` | `rustml-hub` |
| `sweai tokenizer` | `rustml-tokenizer` | `rustml-tokenizer` |

**Example with the standalone binary:**

```bash
./target/release/rustml-infer --safetensors google/gemma-3-1b-it \
  --prompt "Hello" --max-tokens 20
```

---

## Supported Architectures

SafeTensors inference auto-detects architecture from `config.json` `model_type`:

| Architecture | Model examples |
|--------------|---------------|
| `gpt2` | `openai-community/gpt2` |
| `gemma3` / `gemma3_text` | `google/gemma-3-1b-it` |
| `llama` | LLaMA family |
| `mistral` | Mistral family |
| `qwen2` | Qwen 2 family |
| `phi3` | Phi-3 family |
| `falcon` | Falcon family |
| `mixtral` | Mixtral (MoE) |

GGUF models support any architecture embedded in the GGUF file.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Failed to download model` | Check internet connection; for gated models, set `HF_TOKEN` and accept the model license on HuggingFace |
| `Provide a GGUF model path or --safetensors <MODEL_ID>` | Supply either a GGUF file path or `--safetensors <model-id>` |
| `Failed to parse GGUF` | The file is not a valid GGUF model or does not exist |
| `Unsupported SafeTensors model_type` | The model architecture is not yet supported — see [Supported Architectures](#supported-architectures) |
| Slow generation | Use a release build: `cargo build --release -p rustml-cli` |
| `--stream is not supported with --batch-file` | Remove `--stream` when using batch mode |
| `--interactive cannot be used with --prompt` | `--interactive` is a standalone mode — remove `--prompt` |
| `--interactive cannot be used with --batch-file` | `--interactive` is a standalone mode — remove `--batch-file` |
| `[warn] no chat template found in model` | The model doesn't have a recognized chat template; interactive mode will still work but multi-turn formatting may be suboptimal |
| `Key not found` (GGUF meta) | The metadata key does not exist in this GGUF file — list all keys with `sweai gguf meta model.gguf` |

---

## See Also

- [Architecture](../3-design/architecture.md) — system design and SPI architecture
- [Project Structure](../3-design/project_structure.md) — crate layout and conventions
- [Manual Testing Hub](../5-testing/manual_testing.md) — full test procedures
- [ADR-001: Unified LlmModel](../3-design/adr/adr-001-unified-llmmodel-for-gpt2.md) — why GGUF and SafeTensors share one model path
