# rustml Performance Optimization Backlog

> Generated from profiling session 2026-02-19. Based on GPT-2 and Gemma 3 1B inference with `RUST_LOG=rustml=debug`.

## Profiling Summary

### GPT-2 (163M params)
- **Config**: 12 layers, 768 dim, 25 Q8_0 layers (attention F32 - below threshold)
- **Fusions**: None (768 < 1024, biased projections)

| Component | Steady-State | Variance | % of Total |
|-----------|--------------|----------|------------|
| Embedding | ~0.1ms | 0.09-0.16ms | <1% |
| Layers (12) | ~46ms | 40-56ms | **~90%** |
| Norm | ~0.8ms | 0.8-1.6ms | ~2% |
| Projection | ~3ms | 2.0-3.9ms | ~6% |
| **Total** | ~50ms | 44-61ms | 100% |

**Per-layer**: Attention ~0.9ms (50%), FFN ~0.8ms (50%)

### Gemma 3 1B (1.3B params)
- **Config**: 26 layers, 1152 dim, 183 Q8_0 layers (all quantized)
- **Fusions**: 26 gate+up pairs, 26 QKV triples

| Component | Steady-State | Variance | % of Total |
|-----------|--------------|----------|------------|
| Embedding | ~0.3ms | 0.2-0.5ms | <1% |
| Layers (26) | ~175ms | 168-185ms | **~87%** |
| Norm | ~0.02ms | 0.01-0.03ms | <1% |
| Projection | ~22ms | 17-64ms | ~11% |
| **Total** | ~198ms | 185-206ms | 100% |

**Per-layer**: Attention ~1.7ms (15%), FFN ~10ms (85%)

---

## Comparative Analysis

| Metric | GPT-2 | Gemma 3 |
|--------|-------|---------|
| Attention:FFN ratio | 50:50 | 15:85 |
| Step jitter | 1.4x | 1.1x |
| Projection jitter | 2x | **3.7x** |
| QKV quantized | No | Yes |
| QKV fused | No | Yes |
| Gate+up fused | N/A | Yes |

**Key insight**: Fusions work well on Gemma 3 (attention is fast). The bottleneck shifts to FFN for larger models.

---

## Tasks

### 1. Add trace-level profiling for matmul operations
**Priority**: High | **Status**: Done | **Blocks**: #2, #4, #6

Add `RUST_LOG=rustml=trace` output for each matmul call with dimensions, timing, and memory bandwidth.

**Expected output format**:
```
[TRACE] matmul Q_proj [1,768]x[768,768] 0.15ms (12.5 GB/s)
```

**Files**: `rustml/quant/main/src/core/quantize.rs`, `rustml/nn/main/src/core/linear.rs`

---

### 2. Investigate attention quantization threshold for smaller models
**Priority**: High | **Status**: Pending | **Blocked by**: #1

Current Q8_0 threshold (min_dim=1024) skips all 48 attention projections in GPT-2 (768 dim).

**Analysis needed**:
- Profile attention matmul with Q8_0 at 768 dim vs F32
- Determine if lower threshold (512 or 768) provides net benefit
- Consider accuracy impact

**Observed**: GPT-2 attention is 50% of layer time with F32 projections. Gemma 3 attention is only 15% with Q8_0+fusion.

**Files**: `rustml/nlp/main/src/core/model.rs`, `rustml/quant/`

---

### 3. Reduce timing jitter from rayon scheduling
**Priority**: Medium | **Status**: Pending

GPT-2 layer times vary 1.4x within same run. Gemma 3 shows lower jitter (1.1x).

**Investigation**:
1. Add thread pool warm-up at model load
2. Profile cache misses with `perf`/ETW
3. Test with `RAYON_NUM_THREADS=1` to isolate
4. Compare jitter patterns between models

**Files**: `rustml/nn/main/src/core/feed_forward.rs`, `rustml/quant/`

**Success criteria**: Reduce GPT-2 jitter to <1.2x variance

---

### 4. Profile and optimize attention forward pass
**Priority**: Medium | **Status**: Pending | **Blocked by**: #1

Attention optimization is model-dependent:
- GPT-2: 50% of layer time (F32 projections, no fusion)
- Gemma 3: 15% of layer time (Q8_0 + QKV fusion working well)

**Profiling needed**:
- Q/K/V projection time breakdown
- QK^T matmul time
- Softmax time
- Attention*V time
- Output projection time

**Optimization candidates**:
- Flash attention pattern (fused QK^T + softmax + AV)
- Better KV cache memory access
- SIMD softmax

**Files**: `rustml/nn/main/src/core/attention.rs`, `rustml/nn/main/src/core/kv_cache.rs`

---

### 5. Fix lm_head projection variance
**Priority**: **High** | **Status**: Pending

Projection times show significant variance, especially on larger models:
- GPT-2: 2.0-3.9ms (2x variance)
- Gemma 3: 17-64ms (**3.7x variance!**)

**Root cause hypothesis**: Cold cache on first access, memory bandwidth saturation.

**Solutions**:
1. Add warm-up pass through lm_head during model load
2. Prefetch weights before first decode
3. Investigate memory layout for cache efficiency

**Files**: `rustml/nlp/main/src/core/model.rs`, `rustml/nn/main/src/core/linear.rs`

**Success criteria**:
- GPT-2: Consistent ~3ms
- Gemma 3: Consistent ~20ms (<1.5x variance)

---

### 6. Optimize FFN for large models
**Priority**: High | **Status**: Pending | **Blocked by**: #1

FFN dominates layer time on Gemma 3 (85% vs 15% attention). Gate+up fusion helps but FFN is still the bottleneck.

**Gemma 3 FFN profile**:
- Per-layer: ~10ms (fused gate+up)
- Dimensions: 1152 -> 6912 -> 1152 (6x expansion)
- Total: 26 layers x 10ms = 260ms

**Optimization candidates**:
1. Improve Q8_0 matmul memory access patterns
2. Tune rayon chunk sizes for 6912-wide rows
3. Investigate SIMD utilization at these dimensions
4. Consider down_proj fusion with activation

**Files**: `rustml/nn/main/src/core/feed_forward.rs`, `rustml/quant/main/src/core/quantize.rs`

**Success criteria**: Reduce FFN to <8ms/layer on Gemma 3 (20% improvement)

---

### 7. Prefill optimization for longer contexts
**Priority**: Low | **Status**: Pending

Prefill is slower than decode on a per-token basis:
- Gemma 3 prefill: 401ms for 2 tokens (200ms/token)
- Gemma 3 decode: ~200ms for 1 token

**Investigation**:
1. Profile prefill vs decode matmul patterns
2. Check batch dimension handling in attention
3. Identify opportunities for parallel token processing

**Files**: `rustml/nlp/main/src/core/generator.rs`, `rustml/nn/main/src/core/attention.rs`

---

## Recommended Order

1. **#1** - Foundation for data-driven decisions
2. **#5** - Quick win, huge impact on Gemma 3 (3.7x variance)
3. **#6** - FFN is 85% of Gemma 3 layer time
4. **#2** - Enable Q8_0 for GPT-2 attention
5. **#3** - Reduce timing jitter
6. **#4** - Attention optimizations (lower priority since fusion works)
7. **#7** - Prefill optimization (context-dependent)

## Expected Impact

| Task | Model | Expected Improvement |
|------|-------|---------------------|
| #1 Trace profiling | All | Enables data-driven optimization |
| #2 Lower Q8_0 threshold | GPT-2 | 10-20% layer speedup |
| #3 Jitter fix | GPT-2 | More predictable latency |
| #4 Attention optimization | GPT-2 | 5-10% layer speedup |
| #5 lm_head warmup | Gemma 3 | ~40ms faster early steps |
| #6 FFN optimization | Gemma 3 | 20% layer speedup (~35ms/step) |
| #7 Prefill optimization | All | Faster time-to-first-token |

## Test Commands

```bash
# GPT-2 profiling
RUST_LOG=rustml=debug cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 10 --temperature 0

# Gemma 3 1B profiling (requires HF_TOKEN)
HF_TOKEN=xxx RUST_LOG=rustml=debug cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors google/gemma-3-1b-it --prompt "Hello" --max-tokens 5 --temperature 0
```
