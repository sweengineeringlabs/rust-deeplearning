# Performance Trace Bottleneck Report

**Date:** 2026-02-19
**Tool:** Hierarchical `log` crate instrumentation (`RUST_LOG=rustml=debug|trace`)
**Platform:** Linux 6.6.87 (WSL2), x86_64 AVX2, 8 rayon threads
**Build:** `cargo build --release`

---

## Models Tested

| Model | Format | Params | Layers | d_model | Vocab | Quantization |
|-------|--------|--------|--------|---------|-------|--------------|
| GPT-2 | SafeTensors | 163M | 12 | 768 | 50,257 | F32 |
| TinyLlama 1.1B Chat | GGUF | 1,100M | 22 | 2,048 | 32,000 | Q4_0 |
| Gemma 3 1B IT | SafeTensors | 1,302M | 26 | 1,152 | 262,144 | F32 |

---

## 1. GPT-2 (124M, F32 SafeTensors)

### High-level timing

| Phase | Time | Notes |
|-------|------|-------|
| Prefill (5 tokens) | 306ms | ~61ms/token |
| Decode (avg/token) | **63ms** | ~16 tok/s |

### Decode breakdown (per step)

| Component | Time | % of total |
|-----------|------|------------|
| Layers total | ~50ms | 77% |
| Projection (lm_head) | ~13ms | **20%** |
| Embedding | 0.07ms | <1% |
| Norm | 0.5ms | <1% |
| Sample | 0.04ms | <1% |

### Per-layer breakdown (trace, layer 2, single-token decode)

| Op | Time | Notes |
|----|------|-------|
| QKV projections (3x linear 768->768) | 1.1ms | 3 matmuls |
| Output projection (linear 768->768) | 0.3ms | |
| Batched matmul (Q@K^T + attn@V) | 0.1ms | Tiny for seq_len=1 |
| Softmax | 0.004ms | |
| KV cache update+view | 0.04ms | |
| **Attention total** | **1.8ms** | |
| FFN up_proj (768->3072) | 1.3ms | |
| FFN down_proj (3072->768) | 1.2ms | |
| **FFN total** | **2.7ms** | |

**Finding:** Linear projections dominate (~95% of per-layer time). The lm_head
projection (768->50257) is disproportionately expensive at ~13ms/step (20% of
decode time).

---

## 2. TinyLlama 1.1B (Q4_0 GGUF)

### High-level timing

| Phase | Time | Notes |
|-------|------|-------|
| Prefill (6 tokens) | 984ms | ~164ms/token |
| Decode (avg/token) | **370ms** | ~2.7 tok/s |

### Decode breakdown (per step)

| Component | Time | % of total |
|-----------|------|------------|
| Layers total | ~345ms | **93%** |
| Projection (lm_head) | ~24ms | 6% |
| Embedding + norm | <1ms | <1% |

### Attn vs FFN (decode, per layer avg)

| Component | Time | % of layer |
|-----------|------|------------|
| Attention | ~3.8ms | 23% |
| FFN (SwiGLU) | ~12.5ms | **77%** |

**Finding:** FFN is 3.3x more expensive than attention during decode. The Q4_0
quantized SwiGLU (3 projections: gate+up+down, each 2048->5632) dominates.
This is expected -- SwiGLU has 3 weight matrices and the hidden dim (5632) is
much larger than d_model.

### Prefill vs decode attn/FFN comparison (layer avg)

| Phase | Attn | FFN | Ratio (FFN/Attn) |
|-------|------|-----|-------------------|
| Prefill | 12.0ms | 29.3ms | 2.4x |
| Decode | 3.8ms | 12.5ms | 3.3x |

---

## 3. Gemma 3 1B IT (F32 SafeTensors)

### High-level timing

| Phase | Time | Notes |
|-------|------|-------|
| Prefill (8 tokens) | 1,305ms | ~163ms/token |
| Decode (avg/token) | **390ms** | ~2.6 tok/s |

### Decode breakdown (per step)

| Component | Time | % of total |
|-----------|------|------------|
| Layers total | ~260ms | **67%** |
| Projection (lm_head) | **~100ms** | **27%** |
| Embedding + norm | <1ms | <1% |

### Attn vs FFN (decode, per layer avg)

| Component | Time | % of layer |
|-----------|------|------------|
| Attention | ~1.5ms | 14% |
| FFN | ~8.8ms | **86%** |

**Finding:** The lm_head projection is a massive bottleneck. Gemma 3's 262K
vocabulary makes the final `[1,1152]x[1152,262144]` matmul cost ~100ms,
consuming 27% of total decode time.

---

## Top Bottlenecks (ranked by impact)

### 1. Gemma 3 lm_head (262K vocab) -- 100ms/step, 27% of decode

The `[1,1152]x[1152,262144]` F32 matmul for the output projection is the
single most expensive operation. Mitigation options:
- **Top-k approximate projection:** Only compute logits for likely tokens
- **Quantize lm_head weights:** Q8_0 or Q4_0 would reduce memory bandwidth
- **Vocabulary pruning:** Skip padding/unused token ranges

### 2. FFN linear projections (all models) -- 67-93% of layer time

The feed-forward network dominates every model's per-layer budget. For SwiGLU
models (TinyLlama, Gemma 3) this is 3 projections per layer. Mitigation:
- **Better Q4 kernels:** Current dequant-then-dot path has room for SIMD
  optimization in the inner loop
- **Tiled/blocked matmul:** Improve cache locality for large F32 matmuls
- **Mixed precision:** F16 accumulation where acceptable

### 3. GPT-2 lm_head (50K vocab) -- 13ms/step, 20% of decode

Same root cause as #1 but with smaller vocab. The `[1,768]x[768,50257]` matmul
is still expensive relative to the 12-layer model.

### 4. Cold-start layer-0 latency

The first layer in each forward pass consistently runs 2-3x slower than steady
state across all models (e.g., GPT-2 layer 0: 20ms vs layer 2: 5.7ms). This
is likely CPU cache warming and branch prediction effects.

---

## Methodology

Instrumentation was added using the `log` crate at two granularity levels:

- **`RUST_LOG=rustml=debug`** -- Layer-level summaries: prefill/decode timing,
  per-layer totals, attn/FFN split, forward-pass phase breakdown
  (embedding/layers/norm/projection)
- **`RUST_LOG=rustml=trace`** -- Op-level detail: individual linear projections
  with dtype, softmax, matmul, batched_matmul, RoPE, embedding lookup, KV cache
  update/get_view, quantized matmul entry points

Without `RUST_LOG` set, instrumentation has zero overhead (single atomic load +
branch per call site, optimized away in release builds).

### Log format

```
[perf] component::op [shape/details] elapsed_ms
```

### Example output

```
# debug level
[perf] generator::prefill tokens=5 306.532ms
[perf] model::forward embedding=0.375ms layers=274.149ms norm=2.074ms projection=28.417ms total=305.423ms
[perf] transformer[0]::forward attn=20.938ms ffn=13.034ms
[perf] generator::decode_step token=6 model=64.694ms sample=0.037ms total=64.731ms

# trace level (adds within each layer)
[perf] linear::forward [1, 1, 768]->[768,3072] F32 4.956ms
[perf] batched_matmul [1, 12, 1, 64]x[1, 12, 64, 1] 0.041ms
[perf] softmax [1, 12, 1, 1] 0.049ms
[perf] kv_cache::update layer=0 pos=0 0.012ms
[perf] quant::matmul_f32_q4 [1x2048]x[5632x2048] 12.3ms
```

### Instrumented files (19 total)

| Crate | File | Level |
|-------|------|-------|
| rustml-nlp | generator.rs | debug |
| rustml-nlp | model.rs | debug |
| rustml-nn | transformer_block.rs | debug |
| rustml-nn | attention.rs | debug |
| rustml-nn | feed_forward.rs | debug |
| rustml-nn | linear.rs | trace |
| rustml-nn | rope.rs | trace |
| rustml-nn | embedding.rs | trace |
| rustml-nn | kv_cache.rs | trace |
| rustml-core | ops.rs (softmax, matmul, batched_matmul, rms_norm) | trace |
| rustml-quant | quantize.rs (5 matmul entry points) | trace |
