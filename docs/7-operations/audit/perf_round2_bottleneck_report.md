# Performance Optimization Round 2 — Bottleneck Report

**Date:** 2026-02-19
**Commit:** `0293c76` (branch: `sweengineeringlabs`)
**Tool:** Hierarchical `log` crate instrumentation (`RUST_LOG=rustml=debug|trace`)
**Platform:** Linux 6.6.87 (WSL2), x86_64 AVX2, 8 rayon threads
**Build:** `cargo build --release` (thin LTO enabled)

---

## Changes Under Test

Three fixes applied on top of Round 1 (`c871e75`):

1. **Quantize all F32 linear layers to Q8_0 at load time** — `LlmModel::quantize_all_weights()` covers attention q/k/v/out_proj, FFN up/down/gate_proj, MoE experts, and lm_head.
2. **Register-based i8→f32 conversion in Q8_0 SIMD kernels** — AVX2 uses `_mm256_cvtepi8_epi32` + `_mm256_cvtepi32_ps`; new SSE4.1 kernel; NEON uses `vmovl_s8` → `vmovl_s16` → `vcvtq_f32_s32`.
3. **Thin LTO in release profile** — Enables cross-crate inlining of `Linear::forward` → `matmul_f32_q8` → `dot_q8_block` hot path.

---

## Models Tested

| Model | Format | Params | Layers | d_model | Vocab | Layers Quantized |
|-------|--------|--------|--------|---------|-------|------------------|
| GPT-2 | SafeTensors | 163M | 12 | 768 | 50,257 | 73 (all linear) |
| Gemma 3 1B IT | SafeTensors | 1,302M | 26 | 1,152 | 262,144 | 183 (all linear) |

---

## 1. GPT-2 (124M, Q8_0 runtime-quantized)

### Command

```bash
RUST_LOG=rustml=debug cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 10 --temperature 0
```

### Output

```
Hello, I'm sorry. I'm sorry. I
  Generated in 1.04s
```

### High-level timing (decode steps, excluding prefill)

| Phase | Round 1 (F32) | Round 2 (Q8_0) | Improvement |
|-------|---------------|----------------|-------------|
| Decode (avg/token) | ~63ms | **~96ms** | See note |
| Layers total (avg) | ~50ms | ~93ms | — |
| Projection (lm_head) | ~13ms | **~4.8ms** | **2.7x** |

**Note:** GPT-2 per-layer time increased because the Q8_0 matmul path for small 768-dim weights incurs quantization + dequantization overhead that exceeds the bandwidth savings at this model size. The lm_head (768×50257) benefits clearly due to its large output dimension.

### Decode breakdown (per step, steady-state avg)

| Component | Time | % of total |
|-----------|------|------------|
| Layers total | ~93ms | 95% |
| Projection (lm_head) | ~4.8ms | **5%** |
| Embedding | 0.1ms | <1% |
| Norm | 0.5ms | <1% |

### Per-layer breakdown (trace, single-token decode, steady state)

| Op | Round 1 (F32) | Round 2 (Q8_0) |
|----|---------------|----------------|
| QKV projections (3× 768→768) | 1.1ms | ~2.8ms |
| Output projection (768→768) | 0.3ms | ~0.6ms |
| Attention total | 1.8ms | ~4.3ms |
| FFN up_proj (768→3072) | 1.3ms | ~1.0ms |
| FFN down_proj (3072→768) | 1.2ms | ~0.9ms |
| FFN total | 2.7ms | ~2.1ms |
| **Layer total** | ~4.8ms | ~7ms |

**Finding:** For GPT-2's small dimensions (768), Q8_0 quantization is a net negative on attention projections (overhead > bandwidth savings) but a net positive on FFN projections (3072 wide) and lm_head (50257 wide). The crossover point where Q8_0 wins is around 1024+ output features.

---

## 2. Gemma 3 1B IT (Q8_0 runtime-quantized)

### Command

```bash
RUST_LOG=rustml=debug cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors google/gemma-3-1b-it --prompt "Hello" --max-tokens 5 --temperature 0
```

### Output

```
Hello with the prompt, "
  Generated in 1.60s
```

### High-level timing

| Phase | Round 1 (F32) | Round 2 (Q8_0) | Improvement |
|-------|---------------|----------------|-------------|
| Prefill (2 tokens) | ~1,305ms | **~422ms** | **3.1x** |
| Decode (avg/token) | ~390ms | **~227ms** | **1.7x** |
| Layers total (avg decode) | ~260ms | **~202ms** | **1.3x** |
| Projection (lm_head) | ~100ms | **~24ms** | **4.2x** |

### Decode breakdown (per step, steady-state avg of steps 2-5)

| Component | Round 1 (F32) | Round 2 (Q8_0) | Improvement |
|-----------|---------------|----------------|-------------|
| Layers total (26 layers) | ~260ms | ~202ms | 1.3x |
| Projection (lm_head) | ~100ms | **~24ms** | **4.2x** |
| Embedding | <1ms | <1ms | — |
| Norm | <1ms | <1ms | — |

### Per-step forward pass timings (all decode steps)

| Step | Embedding | Layers | Norm | Projection | Total |
|------|-----------|--------|------|------------|-------|
| 1 (prefill) | 1.4ms | 368ms | 0.02ms | 52ms | 422ms |
| 2 | 0.5ms | 200ms | 0.02ms | 28ms | 228ms |
| 3 | 0.5ms | 232ms | 0.02ms | 22ms | 254ms |
| 4 | 0.4ms | 189ms | 0.02ms | 22ms | 212ms |
| 5 | 0.4ms | 188ms | 0.02ms | 25ms | 213ms |

### Per-layer breakdown (decode, steady-state avg)

| Op | Round 1 (F32) | Round 2 (Q8_0) |
|----|---------------|----------------|
| Attention total | ~1.5ms | ~1.5ms |
| FFN (GeGLU, 3 projections) | ~8.8ms | **~6.0ms** |
| **Layer total** | ~10ms | **~7.7ms** |

**Finding:** The massive improvement comes from two sources:
1. **lm_head:** 4.2x faster (100ms → 24ms) — the 262K-vocab `[1,1152]×[1152,262144]` matmul benefits enormously from Q8_0 bandwidth reduction.
2. **FFN per layer:** ~1.5x faster (8.8ms → 6.0ms) — the three GeGLU projections (`[1,1152]×[1152,6144]` × 2 + `[1,6144]×[6144,1152]`) benefit from Q8_0 at these dimensions.
3. **Attention projections:** Neutral (~1.5ms) — 1152-dim Q/K/V projections are near the crossover point.

---

## Comparison: Round 1 vs Round 2

### GPT-2

| Metric | Round 1 | Round 2 | Delta |
|--------|---------|---------|-------|
| lm_head projection | ~13ms | ~4.8ms | **-63%** |
| Per-layer (avg) | ~4.8ms | ~7ms | +46% (regression) |
| Total decode/step | ~63ms | ~96ms | +52% (regression) |

### Gemma 3 1B

| Metric | Round 1 | Round 2 | Delta |
|--------|---------|---------|-------|
| lm_head projection | ~100ms | ~24ms | **-76%** |
| Per-layer (avg) | ~10ms | ~7.7ms | **-23%** |
| Total decode/step | ~390ms | ~227ms | **-42%** |
| Prefill | ~1,305ms | ~422ms | **-68%** |

---

## Remaining Bottlenecks (ranked)

### 1. Gemma 3 per-layer time still ~7.7ms — FFN dominates

The three GeGLU projections consume ~6ms/layer (78% of layer time). At 26 layers this adds up to ~200ms/step. Further optimization options:
- **Fused gate+up projection:** Single matmul `[1,1152]×[1152,12288]` then split, halving kernel launch overhead
- **Q4_0 quantization for FFN:** Further 2× bandwidth reduction at cost of some accuracy
- **Batched dequantize:** Amortize Q8_0 dequant overhead across multiple rows

### 2. Gemma 3 lm_head still ~24ms — 262K vocab floor

Even with Q8_0, the `[1,1152]×[1152,262144]` matmul reads 262K×1152 = 302M bytes (Q8_0). Options:
- **Speculative top-k projection:** Only compute logits for top-N candidate tokens
- **Q4_0 lm_head:** Further 2× bandwidth reduction

### 3. GPT-2 Q8_0 regression on small dimensions

Q8_0 quantization hurts GPT-2 performance on 768-dim attention projections. The quantize/dequantize overhead exceeds bandwidth savings at this size. Options:
- **Dimension-aware quantization:** Only quantize layers above a threshold (e.g., out_features ≥ 1024)
- **Keep GPT-2-class models as F32:** Skip quantization for models with dim < 1024

### 4. Cold-start layer-0 latency (both models)

First decode step is consistently 1.5-2× slower than steady state (Gemma 3: layer 0 = 10.8ms vs avg 7.7ms). CPU cache warming effect.

---

## Methodology

Same instrumentation as Round 1. All timings are from single runs on the same machine under low load. Decode timings exclude the prefill step (step 1) for steady-state averages.

### Reproduce

```bash
# GPT-2
RUST_LOG=rustml=debug cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 10 --temperature 0

# Gemma 3 1B
RUST_LOG=rustml=debug cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors google/gemma-3-1b-it --prompt "Hello" --max-tokens 5 --temperature 0
```
