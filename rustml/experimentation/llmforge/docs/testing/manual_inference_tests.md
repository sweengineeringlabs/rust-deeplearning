# Manual Inference Tests

> **TLDR:** Manual test checklist for text generation: sampling parameters, streaming, REPL mode, and performance.

**Audience**: Developers, QA

**WHAT**: Manual test procedures for the inference pipeline and generation modes
**WHY**: Validates that sampling, generation, and interactive features work correctly end-to-end
**HOW**: Step-by-step test tables with expected outcomes

---

## Table of Contents

- [Single-shot generation](#1-single-shot-generation)
- [Sampling parameters](#2-sampling-parameters)
- [Interactive REPL](#3-interactive-repl)
- [Streaming output](#4-streaming-output)
- [Threading](#5-threading)
- [Model stats output](#6-model-stats-output)

---

## 1. Single-shot generation

> All tests below use GPT-2 (`openai-community/gpt2`) unless noted. Prepend `llmforge run --model openai-community/gpt2` to each command.

| Test | Flags | Expected |
|------|-------|----------|
| Basic generation | `--prompt "Hello world" --max-tokens 16` | Generates 16 tokens of text following the prompt |
| Empty prompt | `--prompt "" --max-tokens 8` | Generates text (model generates from empty context) |
| One token | `--prompt "The" --max-tokens 1` | Generates exactly 1 token |
| Max tokens respected | `--prompt "Once upon a time" --max-tokens 32` | Generation stops at 32 tokens (stderr shows `[32 tokens in ...]`) |
| Prompt echoed | `--prompt "Hello" --max-tokens 8` | Stdout starts with generated text (prompt itself not in stdout, only continuation) |
| Performance stats | `--prompt "x" --max-tokens 16 2>&1 \| grep tok/s` | Stderr line shows `[N tokens in X.XXs — Y.Y tok/s]` |

## 2. Sampling parameters

| Test | Flags | Expected |
|------|-------|----------|
| Temperature 0 (greedy) | `--prompt "The capital of France is" --temperature 0 --max-tokens 8` | Deterministic output; run twice, get identical results |
| Temperature 1.0 | `--prompt "The" --temperature 1.0 --max-tokens 32` | More varied/creative output; run twice, likely different results |
| Temperature 2.0 | `--prompt "The" --temperature 2.0 --max-tokens 32` | Very random output, possibly incoherent |
| Top-k 1 | `--prompt "The" --top-k 1 --max-tokens 16` | Equivalent to greedy; deterministic |
| Top-k 5 | `--prompt "The" --top-k 5 --max-tokens 32` | Limited diversity, coherent output |
| Top-p 0.1 | `--prompt "The" --top-p 0.1 --max-tokens 32` | Very focused sampling, less diverse |
| Top-p 1.0 | `--prompt "The" --top-p 1.0 --max-tokens 32` | No nucleus filtering (all tokens eligible) |
| Repetition penalty 1.0 | `--prompt "The" --repetition-penalty 1.0 --max-tokens 64` | Default, no penalty applied |
| Repetition penalty 1.3 | `--prompt "The" --repetition-penalty 1.3 --max-tokens 64` | Less repetitive output compared to 1.0 |
| Combined params | `--prompt "Once" --temperature 0.6 --top-k 20 --top-p 0.9 --repetition-penalty 1.1 --max-tokens 32` | Coherent output with reduced repetition |

## 3. Interactive REPL

> These tests require an interactive terminal (not piped input).

| Test | Steps | Expected |
|------|-------|----------|
| Default REPL | `llmforge run --model openai-community/gpt2` (no --prompt) | Shows model stats, then `Prompt>` prompt on stderr |
| REPL prompt | Type `Hello` and press Enter | Generates text, shows token count and speed |
| REPL continue | After first generation, type another prompt | Generates new text (independent of previous) |
| REPL quit | Type `quit` | Prints `Goodbye!` and exits |
| REPL quit case | Type `QUIT` or `Quit` | Exits (case-insensitive) |
| REPL empty input | Press Enter without typing | No generation, shows `Prompt>` again |
| REPL params | `llmforge run --model openai-community/gpt2 --temperature 0.5 --max-tokens 32` | REPL banner shows `Temperature: 0.5 ... Max tokens: 32` |
| Force REPL | `llmforge run --model openai-community/gpt2 --prompt "Hello" --interactive` | Runs single-shot first, then enters REPL |
| Prompt-only mode | `llmforge run --model openai-community/gpt2 --prompt "Hello" --max-tokens 16` | Single-shot generation only, exits immediately |

## 4. Streaming output

| Test | Steps | Expected |
|------|-------|----------|
| Tokens appear incrementally | Run single-shot generation with `--max-tokens 64` | Tokens appear one at a time on stdout (not all at once) |
| Stderr vs stdout | `llmforge run --model openai-community/gpt2 --prompt "Hello" --max-tokens 16 2>/dev/null` | Only generated text on stdout (all status messages on stderr) |
| Stderr isolation | `llmforge run --model openai-community/gpt2 --prompt "Hello" --max-tokens 16 1>/dev/null` | Only status messages visible (downloading, loading, stats, tok/s) |

## 5. Threading

| Test | Command | Expected |
|------|---------|----------|
| Auto threads (default) | `llmforge run --model openai-community/gpt2 --prompt "x" --max-tokens 16 --threads 0` | Uses all available cores (default) |
| Single thread | `llmforge run --model openai-community/gpt2 --prompt "x" --max-tokens 16 --threads 1` | Runs on single thread (slower, but works) |
| Specific thread count | `llmforge run --model openai-community/gpt2 --prompt "x" --max-tokens 16 --threads 4` | Uses 4 threads |

## 6. Model stats output

| Test | Command | Expected |
|------|---------|----------|
| GPT-2 stats | `llmforge run --model openai-community/gpt2 --prompt "x" --max-tokens 1 2>&1 1>/dev/null` | Shows: Parameters ~124.4M, Dimension 768, Layers 12, Heads 12, Vocab 50257, Max seq len 1024 |
| GGUF stats | `llmforge run --model TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --file tinyllama-1.1b-chat-v1.0.Q4_0.gguf --prompt "x" --max-tokens 1 2>&1 1>/dev/null` | Shows: Parameters ~1.1B, Dimension 2048, Layers 22, Heads 32, quant breakdown with Q4_0 |
| Quant breakdown | (GGUF model stderr) | Shows `Quantization Breakdown (N tensors)` with type counts |

---

## See Also

- [Manual Testing Hub](manual_testing.md) — prerequisites and setup
- [Manual CLI Tests](manual_cli_tests.md) — CLI argument and command tests
- [Manual Model Tests](manual_model_tests.md) — model source resolution tests
