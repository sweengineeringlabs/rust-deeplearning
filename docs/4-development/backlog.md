# Backlog

## Model Format Support

- [ ] **ONNX runtime/loading is not implemented** — No `.onnx` model loading exists in the production crates. SafeTensors and GGUF are supported; ONNX is not. Requires adding an ONNX parser or integrating an ONNX runtime (e.g., `ort` crate) to load and execute ONNX graphs.

## Inference Production Readiness

> Assessment date: 2026-02-18. Current status: **Production-usable** (P0+P1+P2 complete, GPU backend outstanding).

### P0 — Blocks production use

- [x] **Sampling parameter validation** — `--top-k`, `--top-p`, `--temperature`, `--repetition-penalty` are not validated at CLI entry or in `Generator`. Invalid values (top_k=0, temperature<0, top_p>1) silently produce degenerate output. Add `GenerationConfig::validate()` and call it from CLI and `Generator::new()`.
  - Files: `rustml/nlp/main/src/bin/infer.rs`, `rustml/nlp/main/src/core/generator.rs`, `rustml/nlp/main/src/api/types.rs`

- [x] **Beam search panic on empty beams** — `generator.rs:367` calls `.unwrap()` on `beam.tokens.last()` and `:397` indexes `beams[0]` without bounds checks. Either can panic if beams are unexpectedly empty. Replace with proper error returns.
  - File: `rustml/nlp/main/src/core/generator.rs`

- [x] **Silent error suppression in streaming** — `infer.rs:149` uses `if let Ok(piece)` which silently drops tokenizer decode errors. Log a warning to stderr instead.
  - File: `rustml/nlp/main/src/bin/infer.rs`

- [x] **`GptConfig::from_hf_config` uses `unwrap_or` defaults** — `types.rs:82-88` silently falls back to GPT-2 small defaults when JSON fields are missing. Should return errors for required fields.
  - File: `rustml/nlp/main/src/api/types.rs`

### P1 — Important for reliability

- [x] **No request timeout or cancellation** — `generate()` and `generate_stream()` run to completion or `max_tokens` without interrupt. Long generations block indefinitely. Added `with_deadline(Instant)` and `with_timeout(Duration)` to `Generator`, deadline checks in all generation loops, and `--timeout` CLI flag.
  - Files: `rustml/nlp/main/src/core/generator.rs`, `rustml/nlp/main/src/bin/infer.rs`

- [x] **Chat templates are hardcoded** — Only 3 template formats recognized (`<|user|>`, `[INST]`, `<|im_start|>`). Unknown templates silently fall back to plain text with no warning. Added stderr warning on unrecognized template fallback.
  - File: `rustml/nlp/main/src/core/generator.rs`

- [x] **No peak memory reporting** — KV cache grows linearly with context length. No way to monitor or limit memory usage. Added `KVCache::memory_bytes()` and KV cache size reporting on stderr during model load.
  - Files: `rustml/nn/main/src/core/kv_cache.rs`, `rustml/nlp/main/src/bin/infer.rs`

### P2 — Nice to have for production

- [x] **Sequential batch generation** — `generate_batch()` processes prompts one at a time. Added `generate_batch_parallel()` using `rayon` parallel iterators for concurrent prompt processing.
  - File: `rustml/nlp/main/src/core/generator.rs`

- [x] **No structured logging or metrics** — Added generation timing with `Instant`, token counting in streaming mode, and tokens/sec reporting to stderr.
  - File: `rustml/nlp/main/src/bin/infer.rs`

- [x] **No performance benchmarks** — Added `criterion` benchmarks for argmax, top_k, top_p, repetition_penalty, and sample_categorical sampling functions.
  - File: `rustml/nlp/benches/sampling.rs`

- [ ] **CPU-only inference** — No GPU backend (CUDA, Metal). Limited throughput ceiling. Long-term consideration.

- [x] **No prefix caching** — Added `KVCache::snapshot()` and `KVCache::restore_from()` methods with dimension validation, enabling reuse of prefilled KV state across requests with shared prompts.
  - File: `rustml/nn/main/src/core/kv_cache.rs`
