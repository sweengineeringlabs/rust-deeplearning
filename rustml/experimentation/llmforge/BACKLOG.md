# LLMForge Development Backlog

## Phase 1: Core Infrastructure (Status: Complete)
- [x] N-dimensional Tensor Engine (CPU, faer-backend)
- [x] Basic Matrix Multiplication (MatMul, BatchedMatMul)
- [x] Neural Network Primitives (Linear, LayerNorm, Embedding, Gelu)
- [x] Attention Mechanism (Multi-Head Attention)
- [x] Transformer Architecture (GPT-style Block)
- [x] Model Loader (Custom Binary Format)
- [x] Basic Inference Loop (Greedy Decoding)

## Phase 2: Production Readiness (Status: Complete)

### 1. Configuration Management
- [x] Implement `ModelConfig` struct derived from `serde::Deserialize`.
- [x] Load model hyperparameters (dim, heads, layers, vocab_size) from `config.json`.
- [x] Remove hardcoded constants (e.g., `max_seq_len=1024` in code).

### 2. Weight Loading (SafeTensors)
- [x] Integrate `safetensors` crate.
- [x] Implement `ModelLoader::load_safetensors`.
- [x] Support handling tensor name remapping (HuggingFace vs LLMForge naming) — `WeightMap` + `LlmModel::from_pretrained`.

### 3. Advanced Tokenization
- [x] Integrate `tokenizers` crate (HuggingFace).
- [x] Replace `BPETokenizer::naive` with `NaiveTokenizer`.
- [x] Added `HFTokenizer` wrapper.

### 4. KV-Caching Optimization
- [x] Update `Generator` loop to statefully manage `KVCache`.
- [x] Use `forward_with_cache` in the inference loop instead of stateless `forward`.
- [x] Implement robust `KVCache` management (rotation/eviction if context exceeded, or error handling).

### 5. Backend Optimization
- [x] **Quantization (Q8_0)**: Block-quantized 8-bit storage (32 elements/block, 34 bytes/block). Supports quantize, dequantize, and on-the-fly quantized matmul. ~26.6% of F32 memory. Integrated into `Linear` layer.
- [x] **Quantization (Q4_0)**: 4-bit block quantization (32 elem/block, 18 bytes/block, ~14% of F32). See Phase 5 Performance.
- [x] **SIMD/Parallelism**: `RuntimeConfig` struct with `apply()` to set `faer::set_global_parallelism` and rayon thread pool. Must be called before computation.
- [x] **GPU Support (Investigation)**: Investigated `wgpu` and `cudarc` backends. See findings below.

#### GPU Backend Investigation Findings

- **wgpu** (crate: `wgpu`): Cross-platform WebGPU API. Compute shaders written in WGSL. Works on AMD, NVIDIA, Intel, and Apple GPUs. Moderate integration effort (2-3 weeks for matmul offload). Recommended feature flag: `gpu-wgpu`.
- **cudarc** (crate: `cudarc`): Thin Rust CUDA bindings with cuBLAS for matmul. NVIDIA-only but simpler integration path (1-2 weeks). Recommended feature flag: `gpu-cuda`.
- **Recommendation**: Use `wgpu` for broad hardware compatibility, `cudarc` for maximum NVIDIA performance. Both should be behind feature flags to keep the default build dependency-free. Start with matmul offload, then expand to attention and element-wise ops.

---

## Phase 3: Safety & Correctness Fixes (Status: Complete)

Issues discovered during code review, organized by severity.

### CRITICAL — Memory Safety

- [x] **Replace unsafe `Vec<f32>` to `Vec<u8>` pointer casts** — All 17 `mem::forget`/`Vec::from_raw_parts` blocks replaced with `bytemuck`-based safe conversion (`f32_vec_to_bytes` helper using `try_cast_vec` + `cast_slice` fallback).
- [x] **Fix dtype assumption in `contiguous()`** — `recursive_copy()` now uses `dtype.size()` instead of hardcoded `4`. Q8_0 tensors return an error instead of corrupting memory.
- [x] **Add bounds checking to unsafe pointer arithmetic** — Replaced raw pointer casts in `as_slice_f32()` and `to_f32()` with `bytemuck::try_cast_slice` + safe fallback. Reduced unsafe blocks from 17+ to 4 minimal read-only slice casts.
- [x] **Validate file before memmap** — `loader.rs` now checks file size before memmap (empty file and minimum header size validation).
- [x] **Fix nested view resolution** — View bounds validation added: `view()` returns `Result` and validates max addressable offset fits within storage. Chained `reshape().transpose()` now caught by bounds check.

### CRITICAL — Correctness

- [x] **Implement temperature-based sampling** — `Generator.sample_token()` method: temperature=0 → greedy argmax; temperature>0 → softmax-scaled random sampling. Wired into both prefill and decode loops.
- [x] **Add bounds checking to embedding lookups** — Already implemented (pre-existing `IndexOutOfBounds` check). Verified working.
- [x] **Validate broadcasting shapes** — Added `is_valid_broadcast()` function that checks rhs shape is a valid suffix of lhs shape. `[10,5]+[2]` now correctly errors.

### HIGH — Robustness

- [x] **Add DType support to custom binary loader** — `loader.rs` dtype mapping now supports F32(0), F16(1), BF16(2), I8(3), U8(4) with `UnknownDType` error for invalid bytes.
- [x] **Add checksum/integrity validation to custom binary format** — CRC32 checksum appended by `save_custom_bin()` and verified by `load_custom_bin()`.
- [x] **Add EOS token handling in generator** — `Generator.eos_token_id: Option<u32>` field; generation breaks early when EOS token is produced.
- [x] **Validate config values in `ModelConfig`** — `ModelConfig::validate()` checks dim>0, n_heads>0, dim%n_heads==0, vocab_size>0, n_layers>0, max_seq_len>0. Called from `load()`.
- [x] **Fix Linear layer broadcasting for batched input** — Already handled by `matmul()`'s >2D broadcasting path which collapses batch dims.
- [x] **Add KVCache overflow diagnostics** — KVCache `update()` now returns `SequenceLengthExceeded { max, actual }` with context instead of generic `OutOfMemory`.
- [x] **Validate KVCache head_dim matches attention head_dim** — `KVCache::head_dim()` accessor added; `forward_with_cache()` validates cache head_dim matches attention head_dim.

### MEDIUM — Code Quality

- [x] **Break `tensor.rs` into submodules** — Split into `tensor/mod.rs`, `tensor/dtype.rs`, `tensor/tensor.rs`, `tensor/ops.rs`, `tensor/views.rs`.
- [x] **Change `Device` from `String` to enum** — `Device` enum with `Cpu` variant replaces `String` field.
- [x] **Change `Shape` from `Vec<usize>` to `SmallVec`** — `Shape = SmallVec<[usize; 4]>` avoids heap allocation for tensors with ≤4 dimensions.
- [x] **Improve HFTokenizer error handling** — Error messages now include file path and operation context.
- [x] **Fix NaiveTokenizer UTF-8 handling** — Now iterates chars (not bytes); multi-byte characters encode/decode correctly. `decode()` uses `char::from_u32()` with error handling.
- [x] **Document all unsafe code invariants** — `// SAFETY:` comments added to all 6 unsafe blocks in `tensor.rs` and `loader/mod.rs`.
- [x] **Remove dead `Cuda` error variant** — Removed. Added `InvalidConfig`, `SequenceLengthExceeded`, `UnknownDType` variants.
- [x] **Add activation function abstraction to FFN** — `Activation` enum (Gelu, Silu, Relu) with `FeedForward::with_activation()` constructor. `silu()` and `relu()` added to Tensor.

### Additional changes (Phase 3)

- [x] **`MultiHeadAttention::new()` and `from_weights()` return `Result`** — Replaced `assert_eq!` panics with `Result<Self>` + `InvalidConfig` error. Updated all call sites.
- [x] **`TransformerBlock::new()` returns `Result`** — Propagates `MultiHeadAttention::new()` error.
- [x] **`LlmModel::new()` returns `Result`** — Propagates `TransformerBlock::new()` error.
- [x] **New safety tests** — 15 tests covering broadcasting validation, config validation, tokenizer UTF-8 round-trip, MHA divisibility checks, temperature determinism, EOS field, f32 round-trip.

---

## Phase 4: Test Coverage (Status: Complete)

Current state: 215 tests across 21 test files, all passing.

### Unit Tests — Tensor Operations
- [x] **Matmul numerical correctness** — 2×2, 2×3×3×2, shape mismatch, 3D broadcast, batched 3D. (`tensor_ops_test.rs`)
- [x] **Broadcasting edge cases** — Valid broadcasts (`[3,4] + [4]`), invalid broadcasts (`[10,5] + [2]`, `[6,4] + [3]`), scalar broadcasts.
- [x] **Reshape with non-contiguous tensors** — Transpose then reshape; verify data layout. (`tensor_views_test.rs`)
- [x] **Transpose and permute correctness** — 2D swap, 3D inner-dim swap, out-of-bounds error, permute reorder, wrong-length error. (`tensor_views_test.rs`)
- [x] **Softmax numerical stability** — Uniform input, large values (no NaN), negative values, single element. (`tensor_ops_test.rs`)
- [x] **Layer norm correctness** — Zero-mean/unit-variance with identity weight, custom weight+bias. (`tensor_ops_test.rs`)
- [x] **DType conversions** — F32↔BF16, F32↔F16 round-trips, F32 identity, I8→F32 NotImplemented error. (`dtype_conversion_test.rs`)
- [x] **Contiguous() for strided tensors** — Non-contiguous views produce correct contiguous copy. (`tensor_views_test.rs`)

### Unit Tests — Neural Network Layers
- [x] **Linear forward correctness** — Known weight/input/output triples, 3D input. (`nn_layers_test.rs`)
- [x] **Linear bias addition** — With and without bias; verify values and zero-init. (`nn_layers_test.rs`)
- [x] **Embedding out-of-bounds handling** — Index == vocab_size, index >> vocab_size. (`nn_layers_test.rs`)
- [x] **Embedding lookup correctness** — Indices [0,2,1] return correct weight rows. (`nn_layers_test.rs`)
- [x] **LayerNorm output statistics** — Per-row mean < 1e-4, variance ~1.0. (`nn_layers_test.rs`)

### Unit Tests — Attention
- [x] **Attention weight correctness** — Deterministic output check, cached forward shapes. (`attention_test.rs`)
- [x] **KVCache update semantics** — Sequential updates accumulate correctly, get_view returns expected shapes. (`attention_test.rs`)
- [x] **KVCache overflow behavior** — SequenceLengthExceeded on exceeding max_seq_len. (`attention_test.rs`)
- [x] **Multi-head dimension splitting** — d_model splits correctly; non-divisible errors. (`attention_test.rs`)
- [x] **KVCache head_dim mismatch** — Mismatched head_dim between MHA and cache errors. (`attention_test.rs`)

### Unit Tests — Loader
- [x] **SafeTensors with real files** — BF16 and F32 tensors loaded from generated safetensors files. (`loader_test.rs`)
- [x] **Different dtypes in SafeTensors** — BF16, F32 loading verified. (`loader_test.rs`)
- [x] **Multiple tensors in SafeTensors** — 3 tensors with different shapes all load correctly. (`loader_test.rs`)
- [x] **Corrupted file handling** — Truncated safetensors, invalid header, CRC32 mismatch. (`loader_test.rs`)
- [x] **Custom binary format round-trip** — F32 + BF16 tensors survive save/load cycle. (`loader_test.rs`)
- [x] **Empty file handling** — Empty custom binary file returns error. (`loader_test.rs`)

### Integration Tests
- [x] **End-to-end generation correctness** — temp=0 greedy generation is deterministic. (`integration_test.rs`)
- [x] **Temperature sampling distribution** — temperature=0 produces deterministic output (greedy). Tested.
- [x] **Forward pass output shape** — Model output shape = [batch, seq, vocab_size]. (`integration_test.rs`)
- [x] **Long context generation** — Cache fills then errors; within-limit generation succeeds. (`integration_test.rs`)
- [x] **Error propagation** — dim=0 config error, missing weight map error, exceeding max_seq_len error. (`integration_test.rs`)

---

## Phase 5: Feature Enhancements

### Sampling & Generation (Status: Complete)
- [x] **Top-k sampling** — Select from top-k highest probability tokens. Builder: `with_top_k(k)`.
- [x] **Nucleus (top-p) sampling** — Select from smallest set exceeding cumulative probability p. Builder: `with_top_p(p)`.
- [x] **Repetition penalty** — Penalize tokens that appear in recent context (HF convention: divide positive, multiply negative). Builder: `with_repetition_penalty(penalty)`.
- [x] **Beam search** — `generate_beam(prompt, max_tokens, beam_width)` with per-beam `KVCache::deep_clone()`.
- [x] **Streaming generation** — `generate_stream(prompt, max_tokens, callback)` with `FnMut(u32) -> bool` closure.
- [x] **Batch generation** — `generate_batch(prompts, max_tokens)` sequential per-prompt generation.

### Model Architecture (Status: Complete)
- [x] **RoPE positional encoding** — `RoPEFreqs` precomputes cos/sin tables; applied to Q/K before cache update. Configurable via `PositionEncoding::RoPE` and `rope_theta`.
- [x] **ALiBi positional bias** — Additive bias with geometric slopes `2^(-8h/H)`. Naturally includes causal masking. Configurable via `PositionEncoding::ALiBi`.
- [x] **SwiGLU activation** — `Activation::SwiGLU` variant + `gate_proj: Option<Linear>` on `FeedForward`. Forward: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
- [x] **Grouped-Query Attention (GQA)** — K/V projections sized to `n_kv_heads * head_dim`; `repeat_kv` expands K/V heads before matmul. Supports MQA (n_kv_heads=1).
- [x] **Causal attention mask support** — `Tensor::causal_mask(seq_len, total_len)` produces `[1,1,S,T]` additive mask. Controlled by `causal: bool`. Skipped for seq_len=1.
- [x] **Cross-attention support** — `CrossAttention` struct in `attention/cross.rs`. Q from decoder, K/V from encoder. Optional GQA. No causal mask.
- [x] **Parameter freezing API** — `Freezable` trait on Linear/Embedding/LayerNorm. `LlmModel::freeze_embeddings()` and `parameter_count() -> (total, frozen)`.

### Model Compatibility
- [x] **HuggingFace tensor name remapping** — `WeightMap::llama2()` maps HF weight names to LLMForge internal names.
- [x] **Llama-2 model loading** — HF config parser (`from_hf_llama2`), RoPE/GQA settings, pos embedding loading from weights, gate_proj mapping.
- [x] **GPT-2 model loading** — HF config parser (`from_hf_gpt2`), fused QKV splitting, tied embeddings, LayerNorm with bias, `from_pretrained_gpt2`.
- [x] **GGUF format support** — Hand-written binary parser for llama.cpp ecosystem. Supports F32, F16, Q8_0, Q4_0 tensor types plus k-quant types (Q2_K through Q8_K, dequantized to F32 on load). Mmap-based tensor loading.

### Performance
- [x] **Quantization (Q8_0)** — 8-bit block quantization (~26.6% memory). Quantize/dequantize + on-the-fly quantized matmul.
- [x] **Quantization (Q4_0)** — 4-bit block quantization (32 elem/block, 18 bytes/block, ~14% of F32). Quantize/dequantize + on-the-fly Q4 matmul. Integrated into `Linear` layer.
- [x] **SIMD-accelerated operations** — `std::arch` intrinsics: AVX2 (8 f32/cycle), SSE2 (4 f32/cycle), NEON (aarch64), scalar fallback. Runtime dispatch for Q8_0 and Q4_0 dot products.
- [x] **Tune faer parallelism** — `RuntimeConfig::apply()` sets `faer::set_global_parallelism` + rayon thread pool.
- [x] **Memory pooling/arena allocator** — `TensorPool` with best-fit buffer reuse, capacity limits, and `Tensor::into_bytes()` for buffer extraction.
- [x] **Cache-aware memory layout** — Tiled matmul (TILE_M=4, TILE_N=8) for cache locality. `Tensor::new_aligned()` for 64-byte aligned allocation.
- [ ] **Native quantized matmul (Q4_0×Q8_0)** — Implemented but not yet default. `quantized_matmul_q4_native` quantizes F32 activations to Q8_0 on-the-fly, performs i8×i8→i32 integer dot products per block, scales once per block. AVX2 kernel uses `maddubs` trick (1.65x faster per block); SSE2/NEON/scalar fallbacks. However, per-row Q8_0 quantization overhead (~11µs/row) makes single-token decode 2–9% slower than the dequant-to-F32 path. Prefill (batch≥32) is ~12% faster. Awaiting end-to-end generation quality comparison before switching default.
- [ ] **GPU backend (wgpu)** — WebGPU-based compute shaders for hardware acceleration. Feature flag: `gpu-wgpu`. (Investigated)
- [ ] **GPU backend (cudarc)** — CUDA backend for NVIDIA GPUs. Feature flag: `gpu-cuda`. (Investigated)

### Infrastructure
- [ ] **Benchmarking suite** — Automated benchmarks for matmul, attention, full forward pass with regression tracking.
- [ ] **CI pipeline** — Automated build, test, and lint on push.
- [ ] **Logging/tracing** — Add `tracing` crate for structured logging of forward pass timing and memory usage.
- [x] **Model download utility** — `hf-hub` crate integration for automatic model downloading from HuggingFace Hub. Used in `gpt2_inference` and `gguf_inference` examples. Caches downloads in `~/.cache/huggingface/hub/`.

---

## Phase 6: End-to-End Inference Examples (Status: Complete)

### Examples
- [x] **GPT-2 inference** (`examples/gpt2_inference.rs`) — Auto-downloads GPT-2 124M from `openai-community/gpt2` via `hf-hub`. Pipeline: config.json → `ModelConfig::from_hf_gpt2()`, model.safetensors → `ModelLoader::load_safetensors()` → `WeightMap::gpt2().remap()` → `LlmModel::from_pretrained_gpt2()`, tokenizer.json → `HFTokenizer`. Interactive REPL with streaming token output.
- [x] **GGUF inference** (`examples/gguf_inference.rs`) — Auto-downloads TinyLlama-1.1B-Chat Q4_0 (~670MB) from `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` via `hf-hub`, or loads a local GGUF file. Prints quantization breakdown. Falls back to `NaiveTokenizer` when no tokenizer provided. CLI: `[gguf_path] [tokenizer_path] [--temp N] [--max-tokens N]`.

### Bug Fixes (discovered during end-to-end testing)
- [x] **Mmap alignment safety** — `to_f32()` for F32 dtype now detects misaligned mmap data and copies to owned storage. BF16/F16 paths use safe byte-level reads instead of unsafe `from_raw_parts`. `as_slice_f32()` returns proper error for unaligned data.
- [x] **GPT-2 Conv1D weight transposition** — GPT-2 stores Conv1D weights as `[In, Out]` (not `[Out, In]`). `from_pretrained_gpt2` now transposes c_attn, out_proj, up_proj, and down_proj weights before use.
- [x] **K-quant dequantization** — GGUF loader now supports Q2_K through Q8_K quantization types, dequantized to F32 during loading.
- [x] **Q4_0/Q8_0 → F32 dequantization** — `Tensor::to_f32()` now supports Q4_0 (18 bytes/block, nibble unpacking) and Q8_0 (34 bytes/block, i8 scaling) for tensors that require F32 (embeddings, layer norms).

---

## Phase 7: Performance & Correctness Investigation (Status: Complete)

> Baseline benchmark (2026-02-13, AMD Ryzen 5 7520U, 8 threads, 6.6GB RAM, WSL2):
> GPT-2 124M F32 = 13.5 tok/s | TinyLlama 1.1B Q4_0 = 2.6 tok/s

### Correctness — GGUF Loader
- [x] **Parameter count mismatch** — **BUG FIXED**: `LlmModel::parameter_count()` skipped `self.layers` entirely — the transformer blocks containing ~90% of parameters were never counted. Added `parameter_count() -> (usize, usize)` methods to `Linear`, `LayerNorm`, `MultiHeadAttention`, `CrossAttention`, `FeedForward`, and `TransformerBlock`. TinyLlama now correctly reports ~1.1B parameters.

### Correctness — Quantization Dequantization (CRITICAL)
- [x] **Q4_0 nibble ordering** — **BUG FIXED**: Our Q4_0 dequantization interleaved lo/hi nibbles (lo→pos 0, hi→pos 1, lo→pos 2...) instead of llama.cpp's correct ordering (lo nibbles→positions 0-15, hi nibbles→positions 16-31). Fixed in `to_f32()` (tensor.rs) and all SIMD dot product functions (scalar, AVX2, SSE2, NEON in simd.rs). Verified via correlation test: llama.cpp order vs HF reference = 0.995, our old order vs HF = -0.048.
- [x] **Q6_K element ordering** — **BUG FIXED**: Our Q6_K dequantization used a flat `l in 0..128` loop with incorrect qh indexing (`qh[l/2]`), producing 252/256 elements in wrong positions. Rewritten to match llama.cpp's two-pass structure: two halves of 128 elements, each producing 4 outputs per inner iteration at positions `[l, l+32, l+64, l+96]` with pointer advancement `ql+=64, qh+=32, sc+=8` between halves. This was the primary cause of TinyLlama's garbage output — the output.weight (Q6K) projection was completely scrambled.
- [x] **Q4_K element ordering** — **BUG FIXED**: Interleaved lo/hi nibbles instead of llama.cpp's 64-element chunks (first 32 = lo nibbles, next 32 = hi nibbles). Rewritten to match `dequantize_row_q4_K`.
- [x] **Q5_K element ordering** — **BUG FIXED**: Same interleaving issue as Q4_K plus incorrect qh bit mask rotation. Rewritten with `u1/u2` bit mask rotation matching llama.cpp.
- [x] **Q3_K element ordering + scale unpacking** — **BUG FIXED**: Used flat `j/4` byte indexing instead of llama.cpp's two-pass structure with shift-based 2-bit extraction. Scale unpacking used simple nibble extraction instead of llama.cpp's `aux/kmask` method. Both fully rewritten.
- [x] **Q2_K element ordering** — **BUG FIXED**: Same flat indexing issue. Rewritten with two 128-element halves and shift-based 2-bit extraction matching `dequantize_row_q2_K`.

### Correctness — Tokenizer & CLI
- [x] **BOS/EOS token wiring** — Added `bos_token_id` and `eos_token_id` to `ModelConfig`. GGUF loader reads from metadata. CLI wires into `Generator` builder. Llama models now correctly prepend BOS token.
- [x] **SentencePiece streaming decode** — Single-token decode stripped leading space markers (▁) from SentencePiece tokens. Fixed with incremental buffer decode: maintain running token list, decode full sequence, diff against previous text.

### Performance — Inference Pipeline
- [x] **KV cache not used during generation** — **Already working**: `Generator::generate_stream()` uses `make_cache()` + `forward_with_cache()`. No fix needed.
- [x] **Q4_0 SIMD dispatch** — **Already working**: Runtime `is_x86_feature_detected!("avx2")` dispatch in `dot_q4_block()`. Added verification logging in `RuntimeConfig::apply()` to confirm SIMD capability at startup.
- [x] **Verify Rayon parallelism is active** — **Already working**: `quantized_matmul_q4()` uses `par_chunks_mut()` over output rows. Added verification logging in `RuntimeConfig::apply()` to confirm thread count at startup.

### Chat Template Support
- [x] **Chat template auto-detection** — GGUF loader reads `tokenizer.chat_template` from metadata and stores it in `ModelConfig::chat_template`. `Generator::with_chat_template()` applies the template before encoding. Supports `<|user|>`/`<|assistant|>` (TinyLlama), `[INST]` (Llama-2-Chat, Mistral), and ChatML (`<|im_start|>`) formats. Raw prompts are passed through unchanged when no template is set.

### Logit Comparison Diagnostic
- [ ] **Logit comparison vs llama.cpp** — `examples/logit_dump.rs` dumps top-20 logits (prefill + 3 decode steps) as JSON. `scripts/logit_reference.sh` produces matching output via llama-server's OpenAI-compatible API (curl + jq, no Python). Compare top-1 predictions to determine whether generation degradation is inherent to the model/quantization/greedy setup or a forward-pass bug.

### Known Limitations
- TinyLlama Q4_0 generation degrades into repetitive output after ~10-20 coherent tokens with greedy decoding. Chat template wrapping (`<|system|>...<|user|>...<|assistant|>`) does NOT improve quality — in some cases it makes output worse. Root cause under investigation via logit comparison against llama.cpp reference implementation.
- A native integer matmul (`quantized_matmul_q4_native`) is implemented and available but not yet default — benchmarks show single-token decode is 2–9% slower due to per-row activation quantization overhead.

---

## Phase 8: Gemma 3 Architecture Support (Status: Planned)

> Goal: Load and run Gemma 3 3B (and Gemma 2) GGUF models via the existing inference pipeline.
> Reference: google/gemma-3-3b-it, available as GGUF from bartowski/gemma-3-3b-it-GGUF

### GGUF Loader — Multi-Architecture Metadata
- [ ] **Architecture-aware GGUF config extraction** — `to_model_config()` currently hardcodes `llama.*` metadata keys. Add architecture detection via `general.architecture` metadata field and dispatch to architecture-specific config extractors. Support `gemma2` prefix (Gemma 3 uses `gemma2.*` keys in GGUF) alongside existing `llama` prefix.
- [ ] **Gemma metadata key mapping** — Map `gemma2.embedding_length`, `gemma2.block_count`, `gemma2.attention.head_count`, `gemma2.attention.head_count_kv`, `gemma2.feed_forward_length`, `gemma2.context_length`, `gemma2.attention.layer_norm_rms_epsilon`, `gemma2.vocab_size` to `ModelConfig` fields.

### Model Config — Gemma-Specific Fields
- [ ] **Embedding scale factor** — Gemma scales token embeddings by `sqrt(dim)` after lookup. Add `embedding_scale: Option<f32>` to `ModelConfig`. Set to `sqrt(dim)` for Gemma, `None` for Llama/GPT-2.
- [ ] **RMSNorm weight offset** — Gemma adds +1.0 to RMSNorm weights before applying (i.e., `(1 + weight) * normalized`). Add `norm_weight_offset: Option<f32>` to `ModelConfig` or a `NormVariant` enum. Set to `1.0` for Gemma, `None`/`0.0` for Llama.
- [ ] **Logit soft-capping** — Gemma 2/3 applies `tanh(logits / cap) * cap` to attention logits (cap=50.0) and final output logits (cap=30.0). Add `attn_logit_cap: Option<f32>` and `final_logit_cap: Option<f32>` to `ModelConfig`.
- [ ] **Sliding window attention config** — Gemma 3 alternates between global attention and local sliding window attention across layers. Add `sliding_window_size: Option<usize>` and `sliding_window_pattern: Option<Vec<bool>>` (or a layer-stride rule) to `ModelConfig`.

### Model Forward Pass — Gemma Features
- [ ] **Embedding scaling in forward pass** — Apply `x_emb = x_emb * embedding_scale` after token embedding lookup when `embedding_scale` is set. Modify `LlmModel::forward()` and `forward_with_cache()`.
- [ ] **RMSNorm +1 offset** — Modify `RMSNorm::forward()` (or add a `GemmaRMSNorm` variant) to add the configured offset to weights before multiplication.
- [ ] **Attention logit soft-capping** — In `MultiHeadAttention::forward()`, after computing `Q @ K^T / sqrt(d)`, apply `tanh(scores / cap) * cap` when `attn_logit_cap` is configured.
- [ ] **Final logit soft-capping** — In `LlmModel::forward()`, after the output projection, apply `tanh(logits / cap) * cap` when `final_logit_cap` is configured.
- [ ] **Sliding window causal mask** — Modify `Tensor::causal_mask()` or add `Tensor::sliding_window_mask(seq_len, total_len, window_size)` that masks positions outside `[pos - window_size, pos]`. Apply per-layer based on the sliding window pattern.

### Weight Mapping
- [ ] **Gemma GGUF weight name mapping** — Gemma GGUF files from llama.cpp use the same `blk.{i}.attn_q.weight` naming convention as Llama. Verify existing `gguf_weight_map()` works for Gemma weights. Add any Gemma-specific weight names if needed (e.g., different norm layer names).

### Testing & Validation
- [ ] **Gemma config extraction test** — Unit test: construct mock GGUF metadata with `gemma2.*` keys, verify `to_model_config()` produces correct `ModelConfig`.
- [ ] **RMSNorm +1 offset test** — Unit test: verify `GemmaRMSNorm` output differs from standard `RMSNorm` by exactly the offset contribution.
- [ ] **Logit soft-capping test** — Unit test: verify `tanh(x/cap)*cap` applied correctly, preserves shape, caps extreme values.
- [ ] **Sliding window mask test** — Unit test: verify mask blocks positions outside window, allows positions inside window.
- [ ] **Embedding scaling test** — Unit test: verify output is `sqrt(dim)` times the unscaled embedding.
- [ ] **End-to-end Gemma 3 3B inference** — Integration test: load `bartowski/gemma-3-3b-it-GGUF` Q4_0, run greedy generation on a simple prompt, verify non-degenerate output.
