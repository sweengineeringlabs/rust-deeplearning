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

## Phase 4: Test Coverage

Current state: 38 tests (21 existing + 15 new safety + 2 parallelism), all passing.

### Unit Tests — Tensor Operations
- [ ] **Matmul numerical correctness** — Test against known matrix products, not just output shape.
- [x] **Broadcasting edge cases** — Valid broadcasts (`[3,4] + [4]`), invalid broadcasts (`[10,5] + [2]`, `[6,4] + [3]`), scalar broadcasts.
- [ ] **Reshape with non-contiguous tensors** — Transpose then reshape; verify data layout.
- [ ] **Transpose and permute correctness** — Verify element access after dimension swaps.
- [ ] **Softmax numerical stability** — Large values, negative values, uniform input, single-element input.
- [ ] **Layer norm correctness** — Compare against reference implementation with different epsilon values.
- [ ] **DType conversions** — F32 to F16/BF16 round-trip; verify precision bounds.
- [ ] **Contiguous() for strided tensors** — Non-contiguous views produce correct contiguous copy.

### Unit Tests — Neural Network Layers
- [ ] **Linear forward correctness** — Known weight/input/output triples.
- [ ] **Linear bias addition** — With and without bias; verify values.
- [ ] **Embedding out-of-bounds handling** — Indices at boundary, negative (after cast), beyond vocab_size.
- [ ] **Embedding lookup correctness** — Verify returned vectors match weight rows.
- [ ] **LayerNorm output statistics** — Output should have mean near 0, variance near 1.

### Unit Tests — Attention
- [ ] **Attention weight correctness** — Known Q/K/V with expected attention output.
- [ ] **KVCache update semantics** — Sequential updates produce correct accumulated cache.
- [ ] **KVCache overflow behavior** — Verify error on exceeding max_seq_len.
- [ ] **Multi-head dimension splitting** — Verify heads attend independently.

### Unit Tests — Loader
- [ ] **SafeTensors with real model files** — Small test model with known weights.
- [ ] **Different dtypes in SafeTensors** — F16, BF16, F32 loading.
- [ ] **Corrupted file handling** — Truncated files, wrong headers, invalid shapes.
- [ ] **Custom binary format round-trip** — Write then read; verify exact match.

### Integration Tests
- [ ] **End-to-end generation correctness** — Fixed seed + known tiny model = deterministic expected output.
- [x] **Temperature sampling distribution** — temperature=0 produces deterministic output (greedy). Tested.
- [ ] **Long context generation** — Generate beyond initial KV cache size; verify cache management.
- [ ] **Error propagation** — Invalid config, missing files, shape mismatches produce correct error types.

---

## Phase 5: Feature Enhancements

### Sampling & Generation
- [ ] **Top-k sampling** — Select from top-k highest probability tokens.
- [ ] **Nucleus (top-p) sampling** — Select from smallest set exceeding cumulative probability p.
- [ ] **Repetition penalty** — Penalize tokens that appear in recent context.
- [ ] **Beam search** — Maintain k candidate sequences; return highest scoring.
- [ ] **Streaming generation** — Yield tokens as they are generated (callback or async iterator).
- [ ] **Batch generation** — Process multiple prompts in parallel.

### Model Architecture
- [ ] **RoPE positional encoding** — Replace absolute positional embeddings for better length extrapolation.
- [ ] **ALiBi positional bias** — Alternative to RoPE; no learned parameters.
- [ ] **SwiGLU activation** — Required for Llama-family model compatibility.
- [ ] **Grouped-Query Attention (GQA)** — Wire `n_kv_heads` config into attention; share K/V heads across Q heads.
- [ ] **Causal attention mask support** — Explicit mask tensor instead of implicit full attention.
- [ ] **Cross-attention support** — For encoder-decoder architectures.
- [ ] **Parameter freezing API** — Freeze embeddings or early layers for fine-tuning.

### Model Compatibility
- [x] **HuggingFace tensor name remapping** — `WeightMap::llama2()` maps HF weight names to LLMForge internal names.
- [ ] **Llama-2 model loading** — End-to-end loading and inference of Llama-2 7B.
- [ ] **GPT-2 model loading** — Validate with smaller, publicly available model.
- [ ] **GGUF format support** — Load quantized models from llama.cpp ecosystem.

### Performance
- [x] **Quantization (Q8_0)** — 8-bit block quantization (~26.6% memory). Quantize/dequantize + on-the-fly quantized matmul.
- [ ] **Quantization (Q4_0)** — 4-bit integer quantization for ~4x memory reduction.
- [ ] **SIMD-accelerated operations** — Explicit SIMD intrinsics for dot products and element-wise ops.
- [x] **Tune faer parallelism** — `RuntimeConfig::apply()` sets `faer::set_global_parallelism` + rayon thread pool.
- [ ] **Memory pooling/arena allocator** — Reuse tensor allocations across forward passes instead of fresh allocations.
- [ ] **Cache-aware memory layout** — Optimize stride patterns for CPU cache line utilization.
- [ ] **GPU backend (wgpu)** — WebGPU-based compute shaders for hardware acceleration. Feature flag: `gpu-wgpu`. (Investigated)
- [ ] **GPU backend (cudarc)** — CUDA backend for NVIDIA GPUs. Feature flag: `gpu-cuda`. (Investigated)

### Infrastructure
- [ ] **Benchmarking suite** — Automated benchmarks for matmul, attention, full forward pass with regression tracking.
- [ ] **CI pipeline** — Automated build, test, and lint on push.
- [ ] **Logging/tracing** — Add `tracing` crate for structured logging of forward pass timing and memory usage.
- [ ] **Model download utility** — Download models from HuggingFace Hub directly.
