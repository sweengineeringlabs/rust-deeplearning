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
**Priority**: High | **Status**: ✓ Done

Lowered Q8_0 threshold from 1024 → 768 to enable attention quantization for GPT-2.

**Results (2026-02-19)**:
| Metric | F32 (before) | Q8_0 (after) | Change |
|--------|--------------|--------------|--------|
| Layers quantized | 25 | 73 | +48 |
| Generation time | 0.54s | 0.49s | **-10%** |
| Output quality | ✓ | ✓ | Identical |
| Attention memory | 4x | 1x | **-75%** |

**Conclusion**: Q8_0 at 768 dim is beneficial — faster inference and lower memory with no accuracy loss.

**Files**: `rustml/nlp/main/src/core/model.rs`

---

### 3. Reduce timing jitter from rayon scheduling
**Priority**: Medium | **Status**: ✓ Improved

GPT-2 layer times vary 1.4x within same run. Gemma 3 shows lower jitter (1.1x).

**Solution implemented (2026-02-19)**:
Added `RuntimeConfig::warmup_thread_pool()` called from `warmup_decode()`:
- Forces all rayon threads to spawn and do work before timed inference
- Warms instruction cache with SIMD code paths
- Touches memory to populate TLB entries

**Results**:
| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Layer max time | 13.2ms | 5.4ms | **-59%** |
| Layer variance | 5.1x | 2.5x | **-51%** |
| Step variance | 1.9x | 1.4x | **-26%** |
| Step range | 39-75ms | 35-49ms | Tighter |

**Remaining jitter** (~1.4x) is due to OS scheduling and cache effects that cannot be eliminated in software.

**Files**: `rustml/core/main/src/core/runtime.rs`, `rustml/nlp/main/src/core/model.rs`

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
**Priority**: Medium | **Status**: Resolved (trace analysis)

Projection times show significant variance, especially on larger models:
- GPT-2: 2.0-3.9ms (2x variance)
- Gemma 3: 17-64ms (**3.7x variance!**)

**Root cause (confirmed via trace)**: Variance is between prefill (multi-token) and decode (single-token), not cold cache.

**Trace-level findings**:
- Warmup: 20.5ms @ 15.7 GB/s
- Prefill (2 tokens): 41.8ms @ 7.7 GB/s (expected: ~2x for 2 tokens)
- Decode: 17.8ms @ 18.0 GB/s (excellent bandwidth)

**Conclusion**: No optimization needed. The 3.7x variance was comparing 2-token prefill vs 1-token decode. Single-token decode is consistent at ~18ms with excellent bandwidth (18 GB/s).

**Files**: `rustml/nlp/main/src/core/model.rs`, `rustml/nn/main/src/core/linear.rs`

---

### 6. Optimize FFN for large models
**Priority**: Low | **Status**: Investigated — Near-optimal

FFN dominates layer time on Gemma 3 (85% vs 15% attention). Gate+up fusion helps but FFN is still the bottleneck.

**Trace-level findings (per layer)**:
| Operation | Avg Time | Bandwidth | Utilization |
|-----------|----------|-----------|-------------|
| Gate+Up fused | 1.299ms | 13.9 GB/s | 99% |
| Down proj | 0.794ms | 8.9-11.1 GB/s | 64-79% |
| **Total FFN** | **2.09ms** | — | — |

**Investigation results (2026-02-19)**:
- Tested block tiling to improve L1 cache reuse → **Failed** (2x slower due to vec allocation overhead)
- Root cause: Down_proj input (27KB) doesn't fit in L1 (32KB), requires L2 access
- Gate+up input (4.5KB) fits in L1, hence higher utilization
- This is a fundamental cache size constraint, not code inefficiency

**Remaining optimization paths** (diminishing returns):
1. Hardware prefetching (CPUs already do this automatically)
2. Weight layout restructuring (invasive, would affect all models)
3. GPU acceleration (different architecture)

**Conclusion**: Current implementation is near-optimal for CPU. The 20-35% gap from Gate+Up is due to L1 vs L2 cache access patterns inherent to the different input sizes.

**Files**: `rustml/nn/main/src/core/feed_forward.rs`, `rustml/quant/main/src/core/quantize.rs`

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

1. ~~**#1** - Foundation for data-driven decisions~~ ✓ Done
2. ~~**#5** - Quick win, huge impact on Gemma 3~~ ✓ Resolved (no action needed)
3. ~~**#6** - FFN optimization~~ ✓ Investigated (near-optimal)
4. ~~**#2** - Enable Q8_0 for GPT-2 attention~~ ✓ Done (10% speedup)
5. ~~**#3** - Reduce timing jitter~~ ✓ Improved (51% layer jitter reduction)
6. **#4** - Attention optimizations (lower priority since fusion works)
7. **#7** - Prefill optimization (context-dependent)

## Expected Impact

| Task | Model | Status | Expected Improvement |
|------|-------|--------|---------------------|
| #1 Trace profiling | All | ✓ Done | Enables data-driven optimization |
| #2 Lower Q8_0 threshold | GPT-2 | ✓ Done | 10% faster, 75% less memory |
| #3 Jitter fix | GPT-2 | ✓ Improved | 51% layer jitter reduction |
| #4 Attention optimization | GPT-2 | Pending | 5-10% layer speedup |
| #5 lm_head variance | Gemma 3 | ✓ Resolved | N/A (was measurement artifact) |
| #6 FFN optimization | Gemma 3 | Near-optimal | N/A (L1 cache constraint) |
| #7 Prefill optimization | All | Pending | Faster time-to-first-token |

## Test Commands

```bash
# GPT-2 profiling
RUST_LOG=rustml=debug cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 10 --temperature 0

# Gemma 3 1B profiling (requires HF_TOKEN)
HF_TOKEN=xxx RUST_LOG=rustml=debug cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors google/gemma-3-1b-it --prompt "Hello" --max-tokens 5 --temperature 0
```
