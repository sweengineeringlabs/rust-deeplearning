# Backlog

## Performance Optimization

> Identified via hierarchical performance tracing. See [perf_trace_bottleneck_report.md](../7-operations/audit/perf_trace_bottleneck_report.md) for full profiling data.

- [x] **Gemma 3 lm_head projection is a major bottleneck (262K vocab)** — The final output projection `[1,1152]x[1152,262144]` F32 matmul costs ~100ms/step, consuming 27% of total decode time. Fixed: runtime Q8_0 quantization of lm_head weights reduces memory bandwidth ~4x; parallel F32 gemv path for M=1 decode added to `matmul_inner()`.
  - Files: `rustml/nlp/main/src/core/model.rs`, `rustml/nn/main/src/core/linear.rs`, `rustml/core/main/src/core/tensor/ops.rs`, `rustml/cli/src/cmd/infer.rs`, `rustml/nlp/main/src/bin/infer.rs`

- [x] **FFN linear projections dominate per-layer time (67-93%)** — Feed-forward networks consume the majority of per-layer compute across all models. Fixed: column-parallel quantized matmul for M≤4 decode in all 5 quantized matmul functions. When M=1, parallelizes over out_features (N dimension) instead of rows (M dimension), enabling rayon utilization during decode.
  - Files: `rustml/quant/main/src/core/quantize.rs`

- [x] **GPT-2 lm_head projection overhead (50K vocab)** — The `[1,768]x[768,50257]` matmul costs ~13ms/step, consuming 20% of decode time. Fixed: same lm_head Q8_0 quantization and parallel gemv path as Gemma 3 fix.
  - Files: `rustml/nlp/main/src/core/model.rs`, `rustml/nn/main/src/core/linear.rs`, `rustml/core/main/src/core/tensor/ops.rs`

- [x] **All F32 linear layers remain unquantized at runtime** — Only lm_head was quantized to Q8_0. The 7 linear layers per transformer block (q/k/v/out_proj + up/gate/down_proj) remained F32, dominating per-layer decode time for SafeTensors models. Fixed: `LlmModel::quantize_all_weights()` quantizes every F32 linear layer at load time; CLI wiring replaced lm_head-only path.
  - Files: `rustml/nlp/main/src/core/model.rs`, `rustml/nlp/main/src/bin/infer.rs`, `rustml/cli/src/cmd/infer.rs`

- [x] **Q8_0 SIMD kernel uses scalar i8→f32 conversion** — `dot_q8_block_avx2()` converted i8→f32 via scalar loop + stack array instead of register-based intrinsics. Fixed: AVX2 uses `_mm256_cvtepi8_epi32` + `_mm256_cvtepi32_ps`, new SSE4.1 kernel uses `_mm_cvtepi8_epi32`, NEON uses `vmovl_s8` + `vmovl_s16` + `vcvtq_f32_s32`.
  - Files: `rustml/quant/main/src/core/simd.rs`

- [x] **No LTO in release profile** — No `[profile.release]` section existed in workspace Cargo.toml. The hot cross-crate call chain (`rustml-nn` → `rustml-quant` → simd) could not be inlined. Fixed: thin LTO enabled in release profile.
  - Files: `Cargo.toml`

- [x] **Redundant gate+up matmul dispatch in SwiGLU/GeGLU FFN** — `gate_proj(x)` and `up_proj(x)` are two separate matmuls with identical input and identical dimensions (`[1,1152]×[1152,6144]` Q8_0). Fusing into a single `[1,1152]×[1152,12288]` matmul eliminates one rayon dispatch, one input extraction, and one tensor allocation per layer per step. Fixed: `FeedForward::fuse_gate_up_weights()` concatenates Q8_0 weight bytes at load time; forward splits the fused output via tensor slice.
  - Files: `rustml/nn/main/src/core/feed_forward.rs`, `rustml/nlp/main/src/core/model.rs`, `rustml/nlp/main/src/bin/infer.rs`, `rustml/cli/src/cmd/infer.rs`

- [x] **Redundant Q+K+V matmul dispatch in attention** — `q_proj(x)`, `k_proj(x)`, `v_proj(x)` are three separate matmuls with identical input. Fusing into a single `[1,1152]x[1152,1536]` matmul eliminates two rayon dispatches, two input extractions, and two tensor allocations per layer per step. Fixed: `MultiHeadAttention::fuse_qkv_weights()` concatenates Q8_0 weight bytes at load time; forward splits the fused output via tensor slice.
  - Files: `rustml/nn/main/src/core/attention.rs`, `rustml/nlp/main/src/core/model.rs`, `rustml/nlp/main/src/bin/infer.rs`, `rustml/cli/src/cmd/infer.rs`

- [ ] **lm_head still ~30ms/step (262K vocab floor)** — The `[1,1152]×[1152,262144]` Q8_0 matmul is memory-bandwidth bound. Further reduction requires vocabulary pruning or speculative decoding to avoid the full vocab projection on every step.

- [x] **Cold-start layer-0 latency** — First transformer layer takes ~2× longer than subsequent layers due to rayon thread pool warm-up. Mitigated by `LlmModel::warmup_decode()` which runs a single dummy forward pass at load time.
  - Files: `rustml/nlp/main/src/core/model.rs`, `rustml/nlp/main/src/bin/infer.rs`, `rustml/cli/src/cmd/infer.rs`

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

- [x] **SafeTensors GPT-2 used separate GptModel without KV cache** — The `--safetensors` inference path loaded GPT-2 weights into `GptModel`, a standalone implementation with no KV cache (O(n^2) per generation). Rewired to use `LlmModel::from_pretrained_gpt2()` with `map_gpt2_weights()`, reusing the unified model path that already had KV cache from GGUF work. Result: ~15x faster decoding (0.64 tok/s to 9.3 tok/s). `GptModel` retained as a teaching/reference implementation. See [ADR-001](../3-design/adr/adr-001-unified-llmmodel-for-gpt2.md).
  - Files: `rustml/nlp/main/src/bin/infer.rs`, `rustml/cli/src/cmd/infer.rs`, `rustml/nlp/main/src/core/gpt.rs`

- [ ] **CPU-only inference** — No GPU backend (CUDA, Metal). Limited throughput ceiling. Long-term consideration.

- [x] **No prefix caching** — Added `KVCache::snapshot()` and `KVCache::restore_from()` methods with dimension validation, enabling reuse of prefilled KV state across requests with shared prompts.
  - File: `rustml/nn/main/src/core/kv_cache.rs`

## Testing Infrastructure

- [ ] **`bench_all_optimizations` blocks `cargo test --workspace` for ~76 minutes** — `rustml/nn/tests/bench_optimizations.rs::bench_all_optimizations` runs 2M–500K iterations of SIMD/tensor micro-benchmarks designed for `--release`. In debug mode (standard `cargo test`) it consumed 4,589s (76 min) of a 83-minute full workspace run. Fix options: (a) add `#[ignore]` so it is skipped by default and must be run explicitly with `cargo test --release -p rustml-nn --test bench_optimizations -- --ignored --nocapture`; or (b) scale iteration counts based on `cfg!(debug_assertions)` to keep debug runs under 1s. Option (a) is simpler and matches the test file's own doc comment which already states `--release` is required.
  - File: `rustml/nn/tests/bench_optimizations.rs`
