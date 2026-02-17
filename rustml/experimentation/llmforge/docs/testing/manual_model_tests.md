# Manual Model Loading Tests

> **TLDR:** Manual test checklist for model source resolution, format detection, and architecture auto-detection.

**Audience**: Developers, QA

**WHAT**: Manual test procedures for model loading paths — local files, directories, and HuggingFace repos
**WHY**: Validates the model_source module correctly resolves the `--model` flag across all input types
**HOW**: Step-by-step test tables with expected outcomes

---

## Table of Contents

- [Local file resolution](#1-local-file-resolution)
- [Local directory resolution](#2-local-directory-resolution)
- [HuggingFace repo resolution](#3-huggingface-repo-resolution)
- [Format detection](#4-format-detection)
- [Architecture auto-detection](#5-architecture-auto-detection)
- [Tokenizer resolution](#6-tokenizer-resolution)
- [Config resolution](#7-config-resolution)

---

## 1. Local file resolution

| Test | Setup | Command | Expected |
|------|-------|---------|----------|
| GGUF file | Have a local `.gguf` file | `llmforge info --model ./model.gguf` | Detects GGUF format, inspects file |
| SafeTensors file | Have a local `.safetensors` file | `llmforge info --model ./model.safetensors` | Detects SafeTensors format, inspects file |
| Absolute path | Have a GGUF file at known absolute path | `llmforge info --model /home/user/models/model.gguf` | Resolves absolute path correctly |
| Relative path | Have a GGUF in a subdirectory | `llmforge info --model ./models/model.gguf` | Resolves relative path correctly |
| Nonexistent file | No file at path | `llmforge info --model ./no_such_file.gguf` | Error: not a file, directory, or HF repo |

## 2. Local directory resolution

| Test | Setup | Command | Expected |
|------|-------|---------|----------|
| Dir with GGUF | Directory containing `something.gguf` | `llmforge info --model ./model_dir/` | Finds and uses the .gguf file |
| Dir with SafeTensors | Directory containing `model.safetensors` | `llmforge info --model ./model_dir/` | Finds and uses model.safetensors |
| Dir with both | Directory containing both .gguf and model.safetensors | `llmforge info --model ./model_dir/` | Prefers GGUF over SafeTensors |
| Empty dir | Empty directory | `llmforge info --model ./empty_dir/` | Error: no .gguf or model.safetensors found |
| Dir with sibling files | Directory containing model.safetensors + config.json + tokenizer.json | `llmforge run --model ./model_dir/ --prompt "Hi" --max-tokens 4` | Auto-detects config.json and tokenizer.json from same directory |

## 3. HuggingFace repo resolution

> Requires internet connection. Files are cached in `~/.cache/huggingface/`.

| Test | Command | Expected |
|------|---------|----------|
| SafeTensors repo | `llmforge run --model openai-community/gpt2 --prompt "x" --max-tokens 1` | Downloads model.safetensors, config.json, tokenizer.json |
| GGUF repo with --file | `llmforge info --model TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --file tinyllama-1.1b-chat-v1.0.Q4_0.gguf` | Downloads specified GGUF file |
| GGUF repo without --file | `llmforge run --model TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF --prompt "x"` | Tries to download `model.safetensors` (default), fails with clear error |
| Nonexistent repo | `llmforge info --model fake-org/no-such-model` | Error: failed to download from HuggingFace |
| Cached repo | Run same command twice | Second run does not re-download, starts immediately |
| Org/name pattern | `llmforge info --model openai-community/gpt2` | Recognized as HF repo (contains exactly one `/`, not a local path) |
| Not HF pattern | `llmforge info --model ./openai-community/gpt2` | Treated as local path (starts with `./`), not HF repo |
| Not HF pattern (abs) | `llmforge info --model /models/gpt2` | Treated as local path (starts with `/`), not HF repo |

## 4. Format detection

| Test | Input | Expected |
|------|-------|----------|
| .gguf extension | `--model file.gguf` | Detected as GGUF |
| .safetensors extension | `--model model.safetensors` | Detected as SafeTensors |
| .gguf via --file | `--model repo/name --file model.Q4_0.gguf` | Detected as GGUF from --file extension |
| Unknown extension | `--model file.bin` | Error: cannot detect format |
| No extension | `--model modelfile` | Error: cannot detect format |

## 5. Architecture auto-detection

> SafeTensors models require a config.json to determine architecture.

| Test | Config contents | Command | Expected |
|------|----------------|---------|----------|
| GPT-2 detected | config.json with `"n_embd": 768` | `llmforge run --model ./model.safetensors --prompt "x" --max-tokens 1` | Stderr shows `Architecture: gpt2` |
| Llama detected | config.json with `"hidden_size": 4096` | `llmforge run --model ./model.safetensors --prompt "x" --max-tokens 1` | Stderr shows `Architecture: llama` |
| Explicit gpt2 | Any config | `llmforge run --model ./model.safetensors --arch gpt2 --prompt "x" --max-tokens 1` | Uses GPT-2 loading path regardless of config |
| Explicit llama | Any config | `llmforge run --model ./model.safetensors --arch llama --prompt "x" --max-tokens 1` | Uses Llama loading path regardless of config |
| No config | SafeTensors file without sibling config.json | `llmforge run --model ./model.safetensors --prompt "x"` | Error: SafeTensors model requires config.json |
| Unknown arch | config.json with neither `n_embd` nor `hidden_size` | `llmforge run --model ./model.safetensors --prompt "x"` | Error: cannot auto-detect architecture |

## 6. Tokenizer resolution

| Test | Setup | Command | Expected |
|------|-------|---------|----------|
| Sibling tokenizer | tokenizer.json next to model file | `llmforge run --model ./model.gguf --prompt "x" --max-tokens 4` | Auto-detects tokenizer.json, stderr shows `Tokenizer:` path |
| Explicit tokenizer | --tokenizer flag | `llmforge run --model ./model.gguf --tokenizer /path/to/tokenizer.json --prompt "x" --max-tokens 4` | Uses specified tokenizer |
| HF tokenizer | HF repo with tokenizer.json | `llmforge run --model openai-community/gpt2 --prompt "x" --max-tokens 4` | Downloads and uses tokenizer from repo |
| No tokenizer | GGUF file with no sibling tokenizer.json | `llmforge run --model ./model.gguf --prompt "x" --max-tokens 4` | Falls back to NaiveTokenizer (byte-level), stderr shows warning |

## 7. Config resolution

| Test | Setup | Command | Expected |
|------|-------|---------|----------|
| Sibling config | config.json next to .safetensors file | `llmforge run --model ./model.safetensors --prompt "x" --max-tokens 1` | Auto-detects config.json |
| Explicit config | --config flag | `llmforge run --model ./model.safetensors --config /path/to/config.json --prompt "x" --max-tokens 1` | Uses specified config path |
| HF config | HF repo with config.json | `llmforge run --model openai-community/gpt2 --prompt "x" --max-tokens 1` | Downloads and uses config from repo |
| GGUF ignores config | GGUF file (config embedded in header) | `llmforge run --model ./model.gguf --prompt "x" --max-tokens 4` | Config extracted from GGUF metadata, no external config.json needed |

---

## See Also

- [Manual Testing Hub](manual_testing.md) — prerequisites and setup
- [Manual CLI Tests](manual_cli_tests.md) — CLI argument and command tests
- [Manual Inference Tests](manual_inference_tests.md) — generation and sampling tests
