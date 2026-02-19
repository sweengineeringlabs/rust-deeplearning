# Performance Audit Round 7: Jitter Analysis

> Date: 2026-02-19
> After implementing `RuntimeConfig::warmup_thread_pool()` in Task #3

## Summary

Thread pool warmup reduced GPT-2 layer jitter from 5.1x to 2.3x. This report captures steady-state performance metrics for bottleneck assessment.

---

## GPT-2 (163M params)

**Config**: 12 layers, 768 dim, 73 Q8_0 layers, no fusions (biased projections)

### Decode Step Timings

| Token | Layers | Projection | Total |
|-------|--------|------------|-------|
| 2 | 32.7ms | 3.2ms | 37.4ms |
| 3 | 43.1ms | 3.2ms | 48.0ms |
| 4 | 39.8ms | 3.6ms | 44.6ms |
| 5 | 36.7ms | 3.7ms | 41.5ms |
| 6 | 45.4ms | 3.6ms | 50.2ms |
| 7 | 43.6ms | 2.9ms | 47.6ms |
| 8 | 43.6ms | 3.3ms | 48.1ms |
| 9 | 46.1ms | 3.6ms | 51.2ms |
| 10 | 39.7ms | 2.0ms | 42.8ms |

**Step jitter**: 37-51ms (1.4x variance)

### Layer-Level Analysis

| Metric | Value |
|--------|-------|
| Layer min | 2.3ms |
| Layer max | 5.2ms |
| Layer variance | **2.3x** (was 5.1x before warmup) |
| Attention range | 0.35-1.6ms |
| FFN range | 0.43-2.0ms |
| Attention:FFN ratio | ~50:50 |

### Component Breakdown

| Component | Time | % of Total |
|-----------|------|------------|
| Embedding | 0.1ms | <1% |
| Layers (12) | 37-46ms | ~88% |
| Norm | 0.7-1.3ms | ~2% |
| Projection | 2.0-3.7ms | ~8% |

**Bottleneck**: Layers dominate. Attention and FFN are balanced (50:50).

---

## Gemma 3 1B (1.3B params)

**Config**: 26 layers, 1152 dim, 183 Q8_0 layers, 26 QKV fusions, 26 gate+up fusions

### Warmup and Prefill

| Phase | Time | Notes |
|-------|------|-------|
| Cold warmup | 1012ms | First forward pass |
| Prefill (2 tokens) | 425ms | 213ms/token |

### Decode Step Timings

| Token | Layers | Projection | Total |
|-------|--------|------------|-------|
| 3 | 239.8ms | 50.4ms | 290.5ms |
| 4 | ~240ms | ~55ms | ~295ms |
| 5 | ~235ms | ~52ms | ~287ms |

**Step jitter**: 240-295ms (1.2x variance)

### Layer-Level Analysis

| Metric | Value |
|--------|-------|
| Layer min | 5.9ms |
| Layer max | 18.1ms |
| Layer variance | **3.1x** |

### Component Breakdown

| Component | Time | % of Total |
|-----------|------|------------|
| Embedding | 0.3-0.4ms | <1% |
| Layers (26) | 235-240ms | ~82% |
| Norm | 0.02-0.03ms | <1% |
| Projection | 50-63ms | **~17-21%** |

**Bottleneck**: lm_head projection is significant (262K vocab × 1152 dim = 302M params).

---

## Comparative Analysis

| Metric | GPT-2 | Gemma 3 |
|--------|-------|---------|
| Params | 163M | 1.3B |
| Layers | 12 | 26 |
| Vocab size | 50,257 | 262,144 |
| Step time | 37-51ms | 240-295ms |
| Step jitter | 1.4x | 1.2x |
| Layer jitter | 2.3x | 3.1x |
| lm_head % | ~8% | **~20%** |
| Attention:FFN | 50:50 | 15:85 |

---

## Bottleneck Summary

### GPT-2
- **Status**: Well-balanced after warmup fix
- **Primary bottleneck**: None dominant
- **Optimization potential**: Low (already efficient)

### Gemma 3
- **Status**: lm_head projection is significant overhead
- **Primary bottleneck**: lm_head (20% of decode time)
- **Secondary**: Layer jitter (3.1x variance)
- **Optimization potential**: Medium

---

## Remaining Optimization Tasks

| Task | Target | Expected Impact |
|------|--------|-----------------|
| #4 Attention optimizations | Both | 5-10% layer speedup |
| #7 Prefill optimization | Gemma 3 | Faster time-to-first-token |
| lm_head optimization | Gemma 3 | 10-15% decode speedup |

### lm_head Analysis (Gemma 3)

The lm_head projection is:
- **Dimensions**: 1152 → 262,144 (302M weights)
- **Time**: 50-63ms per decode step
- **Bandwidth**: ~5-6 GB/s (below system peak)

Potential optimizations:
1. Q8_0 quantization of lm_head (currently F32?)
2. Vocabulary pruning for specific use cases
3. Speculative decoding to amortize cost

---

## Test Commands

```bash
# GPT-2 profiling
RUST_LOG=rustml=debug cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 10 --temperature 0

# Gemma 3 1B profiling
RUST_LOG=rustml=debug cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors google/gemma-3-1b-it --prompt "Hello" --max-tokens 5 --temperature 0
```
