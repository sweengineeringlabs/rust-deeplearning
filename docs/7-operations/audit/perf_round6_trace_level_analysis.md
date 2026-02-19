# Performance Optimization Round 6 — Trace-Level Matmul Analysis

**Date:** 2026-02-19
**Commit:** `2a4e49b` (branch: `sweengineeringlabs`)
**Tool:** `RUST_LOG=rustml=trace` with memory bandwidth instrumentation (Task #1)
**Platform:** Windows 11, x86_64 AVX2, 8 rayon threads
**Build:** `cargo build --release` (thin LTO enabled)

---

## Summary

This report provides trace-level matmul profiling data to guide optimization decisions for Tasks #2, #4, #5, and #6 in the backlog.

---

## 1. GPT-2 (163M params)

### Configuration
- **Layers:** 12
- **d_model:** 768
- **Attention:** F32 (below 1024 threshold)
- **FFN:** Q8_0 (up: 768→3072, down: 3072→768)

### Matmul Breakdown

| Operation | Dimensions | Count | Avg Time | Total | Avg BW | % Layer |
|-----------|------------|-------|----------|-------|--------|---------|
| Attention Q/K/V (F32) | [1x768]x[768x768] | 48/step | 0.176ms | 8.4ms | 13.9 GB/s | 47% |
| Attention O (F32) | [1x768]x[768x768] | 12/step | ~0.18ms | 2.1ms | ~14 GB/s | 12% |
| FFN up_proj (Q8_0) | [1x768]x[3072x768] | 12/step | 0.378ms | 4.5ms | 7.3 GB/s | 25% |
| FFN down_proj (Q8_0) | [1x3072]x[768x3072] | 12/step | 0.294ms | 3.5ms | 9.7 GB/s | 19% |
| **Layer total** | — | — | — | **18.5ms** | — | **100%** |

### Observations

1. **F32 attention achieves higher bandwidth** (13.9 GB/s) than Q8_0 FFN (7.3-9.7 GB/s)
2. **Attention dominates** at 59% of matmul time (Task #2 target)
3. **FFN is efficient** with Q8_0 quantization
4. **up_proj is slower than down_proj** despite same element count — memory access pattern difference

### Bandwidth Analysis

| Format | Operation | Bandwidth | Notes |
|--------|-----------|-----------|-------|
| F32 | Attention | 13.9 GB/s | Memory-bound, good utilization |
| Q8_0 | up_proj | 7.3 GB/s | Lower due to dequantization overhead |
| Q8_0 | down_proj | 9.7 GB/s | Better due to wider input (3072) |

---

## 2. Gemma 3 1B (1,302M params)

### Configuration
- **Layers:** 26
- **d_model:** 1152
- **Attention:** Q8_0 + QKV fused (1152→1536, output: 1024→1152)
- **FFN:** Q8_0 + Gate+Up fused (1152→13824), Down (6912→1152)

### Matmul Breakdown

| Operation | Dimensions | Count | Avg Time | Min | Max | Variance | Avg BW |
|-----------|------------|-------|----------|-----|-----|----------|--------|
| QKV Fused | [1x1152]x[1536x1152] | 26/step | 0.257ms | 0.13ms | 2.18ms | **16.9x** | 8.8 GB/s |
| Output Proj | [1x1024]x[1152x1024] | 26/step | 0.158ms | 0.09ms | 0.30ms | 3.3x | 8.6 GB/s |
| Gate+Up Fused | [1x1152]x[13824x1152] | 26/step | 1.299ms | 0.84ms | 2.60ms | 3.1x | 13.9 GB/s |
| Down Proj | [1x6912]x[1152x6912] | 26/step | 0.794ms | 0.50ms | 1.37ms | 2.8x | 11.1 GB/s |
| **lm_head** | [1x1152]x[262144x1152] | 1/step | 17.8ms | — | — | — | 18.0 GB/s |

### Per-Layer Time Distribution

| Component | Time/Layer | % of Layer |
|-----------|------------|------------|
| Attention (QKV + QKT + O) | ~0.46ms | **18%** |
| FFN (Gate+Up + Down) | ~2.09ms | **82%** |
| **Total matmul** | ~2.55ms | 100% |

### lm_head Analysis (Task #5 Target)

| Phase | Dimensions | Time | Bandwidth |
|-------|------------|------|-----------|
| Warmup | [1x1152]x[262144x1152] | 20.5ms | 15.7 GB/s |
| Prefill (2 tokens) | [2x1152]x[262144x1152] | 41.8ms | 7.7 GB/s |
| Decode | [1x1152]x[262144x1152] | 17.8ms | 18.0 GB/s |

**Finding:** Decode lm_head shows low variance (17.8ms) and high bandwidth (18 GB/s). The 3.7x variance from debug logs was between prefill (multi-token) and decode (single-token).

### Observations

1. **QKV fusion has high variance** (16.9x) — first call is 2.2ms, subsequent ~0.2ms (cold cache)
2. **Gate+Up achieves highest bandwidth** (13.9 GB/s avg, up to 20.3 GB/s peak)
3. **FFN dominates** at 82% of layer matmul time (Task #6 target)
4. **lm_head is efficient** at 18 GB/s when warm — variance is cold vs warm, not inherent

---

## 3. Comparative Analysis

### Bandwidth Comparison

| Model | Operation | Bandwidth | Notes |
|-------|-----------|-----------|-------|
| GPT-2 | F32 Attention | 13.9 GB/s | Highest for small model |
| GPT-2 | Q8_0 FFN | 7.3-9.7 GB/s | Dequantization overhead |
| Gemma 3 | Q8_0 QKV | 8.8 GB/s | Good for fused operation |
| Gemma 3 | Q8_0 Gate+Up | 13.9 GB/s | Best Q8_0 bandwidth |
| Gemma 3 | Q8_0 lm_head | 18.0 GB/s | Excellent when warm |

### Variance Analysis

| Model | Operation | Variance | Cause |
|-------|-----------|----------|-------|
| GPT-2 | All | ~1.5x | Rayon scheduling |
| Gemma 3 | QKV Fused | **16.9x** | Cold cache on first call |
| Gemma 3 | Gate+Up | 3.1x | Memory bandwidth saturation |
| Gemma 3 | Down | 2.8x | Memory bandwidth saturation |

---

## 4. Optimization Recommendations

### Task #2: Lower Q8_0 Threshold for GPT-2

**Data-driven decision:**
- GPT-2 F32 attention: 13.9 GB/s, 0.176ms avg
- Gemma 3 Q8_0 attention: 8.8 GB/s, 0.257ms avg

At 768 dim, F32 is competitive with Q8_0. However, the memory savings from Q8_0 (4x less weight storage) may help cache utilization.

**Recommendation:** Test Q8_0 at 768 dim threshold. If bandwidth stays >10 GB/s, adopt.

### Task #5: lm_head Variance

**Finding:** Variance is cold-start related, not inherent to operation. The existing warmup pass appears to help.

**Recommendation:** Verify warmup pass touches lm_head weights. If already done, task may be resolved.

### Task #6: FFN Optimization

**Data:**
- Gate+Up: 1.299ms @ 13.9 GB/s (good)
- Down: 0.794ms @ 11.1 GB/s (room for improvement)

**Bottleneck:** Down projection has lower bandwidth. Possible causes:
1. Non-contiguous memory access pattern
2. Suboptimal rayon chunk sizes for 6912-wide rows

**Recommendation:**
1. Profile down_proj memory access pattern
2. Tune rayon chunk size for 6912 elements (current may be too small)

---

## 5. Raw Data Summary

### GPT-2 Totals (per decode step)

| Component | Matmul Time | % |
|-----------|-------------|---|
| Attention (F32) | 10.5ms | 56% |
| FFN (Q8_0) | 8.0ms | 44% |
| **Total** | 18.5ms | 100% |

### Gemma 3 Totals (per decode step)

| Component | Matmul Time | % |
|-----------|-------------|---|
| Attention (QKV+O) | 10.8ms | 16% |
| FFN (Gate+Up+Down) | 54.4ms | 84% |
| lm_head | 17.8ms | — |
| **Layer matmul total** | 65.2ms | 100% |

---

## Appendix: Test Commands

```bash
# GPT-2 trace profiling
RUST_LOG=rustml=trace cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 10 --temperature 0

# Gemma 3 trace profiling
HF_TOKEN=xxx RUST_LOG=rustml=trace cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors google/gemma-3-1b-it --prompt "Hello" --max-tokens 5 --temperature 0
```
