# Manual Inference Tests

> **TLDR:** Manual test checklist for the `rustml-infer` CLI: load GGUF or SafeTensors models, generate text, sampling options, and streaming.

**Audience**: Developers, QA

**WHAT**: Manual test procedures for the inference CLI tool
**WHY**: Validates end-to-end text generation from GGUF and SafeTensors models — the primary user-facing workflow
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
- [SafeTensors Generation (GPT-2)](#10-safetensors-generation-gpt-2)
- [SafeTensors Multi-Architecture](#11-safetensors-multi-architecture)
- [SafeTensors Diagnostics](#12-safetensors-diagnostics)
- [Optimization Profiles](#14-optimization-profiles)
- [Performance Profiling (RUST_LOG)](#15-performance-profiling-rust_log)
- [Error Cases](#16-error-cases)

---

> **Prerequisite**: A GGUF model file (e.g., Gemma 3 1B IT Q4_0) for GGUF tests, or a cached SafeTensors model (e.g., `openai-community/gpt2`) for SafeTensors tests. Release build recommended for inference speed: `cargo build --release -p rustml-nlp`.
>
> **Gated models**: Gated models (e.g., `google/gemma-3-1b-it`) require a HuggingFace token. Set `export HF_TOKEN=hf_xxx` before running, or use `~/.cache/huggingface/token`. See [Manual Testing Hub](manual_testing.md#huggingface-token-setup) for full token precedence.
>
> **Unified CLI**: All `rustml-infer` commands below can also be run as `sweai infer` (e.g., `sweai infer model.gguf --prompt "Hello"`). Build with `cargo build --release -p rustml-cli`.

## 1. Help & Version

| Test | Command | Expected |
|------|---------|----------|
| Help flag | `rustml-infer --help` | Lists `GGUF_PATH` positional, `--safetensors`, `--prompt`, `--batch-file`, `--max-tokens`, `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty`, `--stream`, `--chat`, `--timeout`, `--opt-profile` |
| Version flag | `rustml-infer --version` | Prints version string |
| No args | `rustml-infer` | Error: `Provide a GGUF model path or --safetensors <MODEL_ID>` |
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

## 10. SafeTensors Generation (GPT-2)

> **Prerequisite**: Download `openai-community/gpt2` via `rustml-hub-cli download openai-community/gpt2` (or `sweai hub download openai-community/gpt2`).

| Test | Command | Expected |
|------|---------|----------|
| Basic generation | `rustml-infer --safetensors openai-community/gpt2 --prompt "The capital of France is" --max-tokens 20` | Prints generated continuation |
| Greedy decoding | `rustml-infer --safetensors openai-community/gpt2 --prompt "1+1=" --temperature 0 --max-tokens 10` | Deterministic output (same result on repeated runs) |
| Short output | `rustml-infer --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 5` | Output is at most ~5 tokens |
| Sampling flags | `rustml-infer --safetensors openai-community/gpt2 --prompt "Once upon" --temperature 0.7 --top-k 50 --top-p 0.9 --repetition-penalty 1.1 --max-tokens 32` | All flags accepted; generates text |
| Streaming | `rustml-infer --safetensors openai-community/gpt2 --prompt "The quick brown fox" --stream --max-tokens 20` | Tokens printed incrementally |
| Stdin prompt | `echo "Hello world" \| rustml-infer --safetensors openai-community/gpt2 --max-tokens 10` | Reads prompt from stdin |
| Batch file | `rustml-infer --safetensors openai-community/gpt2 --batch-file /tmp/prompts.txt --max-tokens 32` | Outputs `[0] ...`, `[1] ...` per prompt |
| Timeout | `rustml-infer --safetensors openai-community/gpt2 --prompt "Tell me a story" --max-tokens 4096 --timeout 5` | Generation stops after ~5s |
| Conflicts with GGUF | `rustml-infer model.gguf --safetensors openai-community/gpt2 --prompt "Hi"` | Error: cannot use both GGUF path and `--safetensors` |

## 11. SafeTensors Multi-Architecture

> **Prerequisite**: Download target models via `sweai hub download <model-id>`. Gated models (e.g., Gemma 3) require `export HF_TOKEN=hf_xxx` or `~/.cache/huggingface/token`. These tests verify that `--safetensors` auto-detects architecture from `config.json` `model_type`, selects the correct weight mapping and model constructor, and uses `HFTokenizer` (from `tokenizer.json`) when available.

| Test | Command | Expected |
|------|---------|----------|
| Gemma 3 1B IT | `rustml-infer --safetensors google/gemma-3-1b-it --prompt "The capital of France is" --max-tokens 20 --stream` | Auto-detects `arch=gemma3`; uses HFTokenizer; generates text |
| Gemma 3 config | (same as above) | stderr: `Config: arch=gemma3, dim=..., layers=..., heads=..., vocab=...` |
| Gemma 3 tokenizer | (same as above) | stderr: `Tokenizer: <N> tokens (tokenizer.json)` |
| GPT-2 BPE fallback | `rustml-infer --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 10` | stderr: `Tokenizer: <N> tokens (BPE)` (falls back when no tokenizer.json) |
| GPT-2 arch detection | (same as above) | stderr: `Config: arch=gpt2, ...` |
| Chat with template | `rustml-infer --safetensors google/gemma-3-1b-it --prompt "What is 2+2?" --chat --max-tokens 32` | Uses chat template from config if available |
| EOS from config | `rustml-infer --safetensors google/gemma-3-1b-it --prompt "Hi" --max-tokens 64` | Stops at model's EOS token (from config.json `eos_token_id`), not GPT-2 hardcoded EOS |
| Unsupported model_type | Download a model with unknown `model_type`, attempt inference | Error: `Unsupported SafeTensors model_type: '<type>'` |

> **Supported architectures**: `gpt2`, `llama`, `mistral`, `qwen2`, `phi3`, `gemma3`, `gemma3_text`, `falcon`, `mixtral`. The dispatch is in `build_safetensors_model()` in `rustml/nlp/main/src/core/model.rs`.

## 12. SafeTensors Diagnostics

> **Note:** SafeTensors inference uses `LlmModel` with a real KV cache — the same unified model path as GGUF models. Architecture is auto-detected from `config.json` `model_type` and weights are remapped via `build_safetensors_model()`. See [ADR-001](../3-design/adr/adr-001-unified-llmmodel-for-gpt2.md) for details.

| Test | What to check | Expected |
|------|---------------|----------|
| Cache hit | stderr when model is cached | `Using cached model: <model-id>` |
| Cache miss | stderr when model is not cached | `Downloading model: <model-id>` |
| Config with arch | stderr | `Config: arch=<type>, dim=<N>, layers=<N>, heads=<N>, vocab=<N>` |
| Weight loading | stderr | `Loading SafeTensors weights...` then `<N> tensors loaded` |
| Model build | stderr | `Building model...` |
| Model ready | stderr | `Model ready: <N>M params` |
| KV cache memory | stderr | `KV cache: <N> MB (<layers>layers x <heads>heads x <seq>seq x <dim>dim x f32 x 2)` |
| Tokenizer (HF) | stderr when tokenizer.json exists | `Tokenizer: <N> tokens (tokenizer.json)` |
| Tokenizer (BPE) | stderr when no tokenizer.json | `Tokenizer: <N> tokens (BPE)` |
| Metrics (stream) | stderr after streaming | `<N> tokens in <T>s (<R> tokens/sec)` |
| Metrics (non-stream) | stderr after generation | `Generated in <T>s` |
| Quantization | stderr during load | `Quantized <N> linear layers F32 -> Q8_0` (N depends on model dimensions; layers with max dim < 1024 are skipped) |
| Gate+up fusion | stderr during load | `Fused <N> gate+up projection pairs` (N = number of SwiGLU/GeGLU layers; absent for models without gate_proj like GPT-2) |

## 14. Optimization Profiles

> **What:** The `--opt-profile` flag selects a runtime optimization profile that toggles rayon parallelism thresholds, in-place tensor ops, and buffered sampling. Profiles: `optimized` (default), `baseline` (all optimizations off), `aggressive` (lower parallelism thresholds).
>
> **Why:** Enables A/B benchmarking, debugging correctness issues, and tuning thresholds for different hardware.

| Test | Command | Expected |
|------|---------|----------|
| Default profile | `rustml-infer --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 20` | stderr shows `Optimization profile: optimized`; generates text |
| Optimized profile | `rustml-infer --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 20 --opt-profile optimized` | stderr shows `Optimization profile: optimized`; generates text |
| Baseline profile | `rustml-infer --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 20 --opt-profile baseline` | stderr shows `Optimization profile: baseline`; generates text |
| Aggressive profile | `rustml-infer --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 20 --opt-profile aggressive` | stderr shows `Optimization profile: aggressive`; generates text |
| Greedy parity (optimized vs baseline) | Compare greedy output (`--temperature 0`) between `--opt-profile optimized` and `--opt-profile baseline` | Identical output text (both profiles produce the same numerical results) |
| Greedy parity (optimized vs aggressive) | Compare greedy output (`--temperature 0`) between `--opt-profile optimized` and `--opt-profile aggressive` | Identical output text |
| GGUF with profile | `rustml-infer model.gguf --prompt "Hello" --max-tokens 20 --opt-profile baseline` | Accepts `--opt-profile` with GGUF models; generates text |
| Stream with profile | `rustml-infer --safetensors openai-community/gpt2 --prompt "Hello" --stream --max-tokens 20 --opt-profile baseline` | Streaming works with baseline profile |
| Unknown profile | `rustml-infer --safetensors openai-community/gpt2 --prompt "Hello" --opt-profile unknown` | Falls back to `optimized` (default); generates text |

## 15. Performance Profiling (RUST_LOG)

> **What:** Setting `RUST_LOG=rustml=debug` or `RUST_LOG=rustml=trace` enables hierarchical performance tracing, printing per-step and per-layer timing breakdowns to stderr. Useful for identifying bottlenecks after optimization changes.
>
> **Build:** Always use `cargo build --release -p rustml-nlp` (or `-p rustml-cli`) — debug builds are too slow for meaningful profiling.

### Debug level (per-step breakdown)

| Test | Command | Expected |
|------|---------|----------|
| GPT-2 debug | `RUST_LOG=rustml=debug cargo run --release -p rustml-nlp --bin rustml-infer -- --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 10 --temperature 0` | Per-step timings on stderr showing Embedding, Layers, Norm, Projection columns |
| GPT-2 quantization | (same as above) | stderr: `Quantized 25 linear layers F32 -> Q8_0` (768-dim attention skipped, FFN + lm_head quantized) |
| GPT-2 no fusion | (same as above) | No `Fused` message (GPT-2 has no gate_proj) |
| GPT-2 output | (same as above) | `Hello, I'm sorry. I'm sorry. I` (deterministic at temp=0) |
| Gemma 3 debug | `RUST_LOG=rustml=debug cargo run --release -p rustml-nlp --bin rustml-infer -- --safetensors google/gemma-3-1b-it --prompt "Hello" --max-tokens 5 --temperature 0` | Per-step timings; all 183 layers quantized (all dims ≥ 1152) |
| Gemma 3 fusion | (same as above) | stderr: `Fused 26 gate+up projection pairs` (all 26 GeGLU layers fused) |
| Gemma 3 output | (same as above) | Generates 5 tokens; `Generated in <T>s` |

### Expected debug output format

```
[DEBUG rustml...] [perf] Step N forward: <T>ms (embed=<T>ms, layers=<T>ms, norm=<T>ms, proj=<T>ms)
```

### Typical per-step timings (release build, x86_64 AVX2)

| Model | Step | Embedding | Layers | Norm | Projection | Total |
|-------|------|-----------|--------|------|------------|-------|
| GPT-2 (25 Q8_0) | prefill | ~0.1ms | ~65ms | ~0.7ms | ~7ms | ~73ms |
| GPT-2 (25 Q8_0) | decode (avg) | ~0.1ms | ~60ms | ~0.6ms | ~5ms | ~66ms |
| Gemma 3 1B (183 Q8_0, 26 fused) | prefill | ~4ms | ~430ms | <0.1ms | ~48ms | ~483ms |
| Gemma 3 1B (183 Q8_0, 26 fused) | decode (avg) | ~0.2ms | ~215ms | <0.1ms | ~28ms | ~243ms |

> **Note:** Timings vary with system load and CPU cache state. First decode step is typically 1.2-1.5x slower than steady state. The numbers above are approximate baselines for comparison, not strict pass/fail thresholds.
>
> **Round 3 change:** Gemma 3 decode improved from ~280ms to ~240ms/step via fused gate+up matmul (26 GeGLU layers). FFN reduced from ~6ms to ~4.7ms/layer by eliminating one rayon dispatch per layer.

### Trace level (per-layer breakdown)

| Test | Command | Expected |
|------|---------|----------|
| Gemma 3 trace | `RUST_LOG=rustml=trace cargo run --release -p rustml-nlp --bin rustml-infer -- --safetensors google/gemma-3-1b-it --prompt "Hello" --max-tokens 3 --temperature 0` | Per-layer timing showing attention and FFN time per layer |

### Expected trace output format

```
[TRACE rustml...] [perf] Layer N: <T>ms (attn=<T>ms, ffn=<T>ms)
```

### Dimension threshold behavior

The runtime Q8_0 quantization applies a dimension threshold (`min_dim = 1024`): layers where `max(in_features, out_features) < 1024` remain F32. After quantization, gate+up projection pairs in SwiGLU/GeGLU layers are fused into single matmuls.

| Model | d_model | Layers quantized | Layers skipped | Gate+up fused | Reason |
|-------|---------|------------------|----------------|---------------|--------|
| GPT-2 | 768 | 25 (FFN + lm_head) | 48 (attention Q/K/V/O) | 0 (no gate_proj) | 768 < 1024 |
| Gemma 3 1B | 1152 | 183 (all) | 0 | 26 (all GeGLU) | 1152 ≥ 1024 |

## 16. Error Cases

| Test | Command | Expected |
|------|---------|----------|
| No model provided | `rustml-infer --prompt "Hello"` | Error: `Provide a GGUF model path or --safetensors <MODEL_ID>` |
| Nonexistent file | `rustml-infer /nonexistent.gguf --prompt "Hello"` | Error: `Failed to parse GGUF` |
| Invalid GGUF | `rustml-infer /tmp/not_a_gguf.bin --prompt "Hello"` | Error: invalid GGUF |
| Bad temperature type | `rustml-infer model.gguf --temperature abc` | Error: invalid value for `--temperature` |
| Bad max-tokens type | `rustml-infer model.gguf --max-tokens xyz` | Error: invalid value for `--max-tokens` |
| Unknown flag | `rustml-infer model.gguf --unknown-flag` | Error: unexpected argument |
| Negative temperature | `rustml-infer model.gguf --prompt "Hi" --temperature -1` | Error: `--temperature must be >= 0.0` |
| Top-k zero | `rustml-infer model.gguf --prompt "Hi" --top-k 0` | Error: `--top-k must be > 0` |
| Top-p out of range | `rustml-infer model.gguf --prompt "Hi" --top-p 1.5` | Error: `--top-p must be in (0.0, 1.0]` |
| Repetition penalty zero | `rustml-infer model.gguf --prompt "Hi" --repetition-penalty 0` | Error: `--repetition-penalty must be > 0.0` |
| SafeTensors bad model | `rustml-infer --safetensors nonexistent/model-xyz --prompt "Hi"` | Error: `Failed to download model` |

---

## See Also

- [Manual Testing Hub](manual_testing.md) — prerequisites and setup
- [Manual Tokenizer Tests](manual_tokenizer_tests.md) — tokenizer CLI tests
- [Manual GGUF Inspector Tests](manual_gguf_inspect_tests.md) — inspecting GGUF model files
- [Manual Hub CLI Tests](manual_hub_cli_tests.md) — model download and cache management
