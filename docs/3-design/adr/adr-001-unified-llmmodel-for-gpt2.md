# ADR-001: Unified LlmModel for GPT-2 SafeTensors Inference

**Status:** Accepted
**Date:** 2026-02-18

## Context

The inference CLI (`rustml-infer`, `sweai infer`) supported two model formats:

- **GGUF models** (Llama, Gemma, etc.) loaded through `LlmModel`, which has a real KV cache, proper autoregressive decoding (O(n) per step), and support for multiple architectures via `ModelConfig`.
- **SafeTensors models** (GPT-2) loaded through `GptModel`, a standalone implementation with no KV cache. It replayed the full forward pass over the entire token history at every decode step (O(n^2) total), producing correct output but at ~0.64 tokens/sec for GPT-2 small.

Meanwhile, `LlmModel` already had a `from_pretrained_gpt2()` constructor that handled all GPT-2 specifics (fused QKV split, Conv1D transpose, LayerNorm with bias, GELU activation, weight tying), and `map_gpt2_weights()` already remapped HuggingFace weight names to LlmModel internal names. This code was written but not wired into the CLI.

## Decision

Route GPT-2 SafeTensors inference through `LlmModel::from_pretrained_gpt2()` instead of `GptModel::from_hub_weights()`. Retain `GptModel` as a teaching/reference implementation.

### Changes

1. **`infer.rs` (both `rustml-infer` and `sweai infer`):** Changed `run_safetensors()` to:
   - Parse config via `ModelConfig::from_json_value()` instead of `GptConfig::from_hf_config()`
   - Remap weights via `map_gpt2_weights()`
   - Build model via `LlmModel::from_pretrained_gpt2()`
   - Report parameter count and KV cache memory (matching GGUF diagnostic output)

2. **`gpt.rs`:** Updated module and struct documentation to clearly label as educational/reference, with pointers to `LlmModel` for production use.

## Consequences

### Positive

- **~15x faster decoding**: 0.64 tok/s to 9.3 tok/s for GPT-2 small (100 tokens), because `LlmModel` caches K/V projections per-layer and only processes the new token at each step.
- **Unified code path**: Both GGUF and SafeTensors models now flow through the same `LlmModel` + `Generator` pipeline, reducing maintenance surface.
- **Consistent diagnostics**: SafeTensors inference now reports parameter count and KV cache memory, matching GGUF output.
- **Zero new code**: The change was purely wiring; `LlmModel::from_pretrained_gpt2()` and `map_gpt2_weights()` already existed.

### Neutral

- `GptModel` remains in the codebase as a teaching reference. It is still exported from `rustml-nlp` and has unit tests, but is no longer used by any CLI command.

### Negative

- None identified. The `LlmModel` path produces identical logits for the same weights (verified by greedy decoding comparison).

## Alternatives Considered

1. **Add KV cache to `GptModel` directly** — Would duplicate the cache infrastructure already in `LlmModel`/`TransformerBlock`/`MultiHeadAttention`. Rejected to avoid maintaining two cache implementations.

2. **Remove `GptModel` entirely** — The standalone implementation maps 1:1 to the GPT-2 paper and HuggingFace weight names. It serves as a readable reference for understanding the architecture without abstraction layers. Kept for educational value.

## References

- `rustml/nlp/main/src/core/model.rs` — `LlmModel::from_pretrained_gpt2()`, `map_gpt2_weights()`
- `rustml/nlp/main/src/core/gpt.rs` — `GptModel` (reference implementation)
- `rustml/nlp/main/src/bin/infer.rs` — `run_safetensors()` (updated)
- `rustml/cli/src/cmd/infer.rs` — `run_safetensors()` (updated)
