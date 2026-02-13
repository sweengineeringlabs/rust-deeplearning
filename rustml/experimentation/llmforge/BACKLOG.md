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
- [ ] **Quantization (Q4_0)**: 4-bit quantization for further memory reduction.
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
- [x] **GGUF format support** — Hand-written binary parser for llama.cpp ecosystem. Supports F32, F16, Q8_0, Q4_0 tensor types. Mmap-based tensor loading.

### Performance
- [x] **Quantization (Q8_0)** — 8-bit block quantization (~26.6% memory). Quantize/dequantize + on-the-fly quantized matmul.
- [x] **Quantization (Q4_0)** — 4-bit block quantization (32 elem/block, 18 bytes/block, ~14% of F32). Quantize/dequantize + on-the-fly Q4 matmul. Integrated into `Linear` layer.
- [x] **SIMD-accelerated operations** — `std::arch` intrinsics: AVX2 (8 f32/cycle), SSE2 (4 f32/cycle), NEON (aarch64), scalar fallback. Runtime dispatch for Q8_0 and Q4_0 dot products.
- [x] **Tune faer parallelism** — `RuntimeConfig::apply()` sets `faer::set_global_parallelism` + rayon thread pool.
- [x] **Memory pooling/arena allocator** — `TensorPool` with best-fit buffer reuse, capacity limits, and `Tensor::into_bytes()` for buffer extraction.
- [x] **Cache-aware memory layout** — Tiled matmul (TILE_M=4, TILE_N=8) for cache locality. `Tensor::new_aligned()` for 64-byte aligned allocation.
- [ ] **GPU backend (wgpu)** — WebGPU-based compute shaders for hardware acceleration. Feature flag: `gpu-wgpu`. (Investigated)
- [ ] **GPU backend (cudarc)** — CUDA backend for NVIDIA GPUs. Feature flag: `gpu-cuda`. (Investigated)

### Infrastructure
- [ ] **Benchmarking suite** — Automated benchmarks for matmul, attention, full forward pass with regression tracking.
- [ ] **CI pipeline** — Automated build, test, and lint on push.
- [ ] **Logging/tracing** — Add `tracing` crate for structured logging of forward pass timing and memory usage.
- [ ] **Model download utility** — Download models from HuggingFace Hub directly.
