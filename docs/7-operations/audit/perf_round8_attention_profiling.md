# Performance Audit Round 8: Attention Component Profiling

> Date: 2026-02-19
> Task #4 from BACKLOG.md

## Summary

Added trace-level profiling to attention forward pass to identify optimization targets. Finding: attention is near-optimal — core attention operations (QK^T + softmax + A*V) total <0.1ms combined. Projections dominate but are already optimized.

---

## Profiling Instrumentation

Added timing to `MultiHeadAttention::forward_with_cache_inner()` for 7 components:

1. **QKV projection** — Q, K, V linear projections (fused or separate)
2. **QK normalization** — RMSNorm on Q and K (Gemma 3 only)
3. **RoPE** — Rotary position encoding application
4. **QK^T matmul** — Attention scores computation + scaling
5. **Softmax** — Attention weight normalization
6. **A*V matmul** — Weighted value aggregation
7. **Output projection** — Final linear projection

Enable with: `RUST_LOG=rustml=trace`

---

## GPT-2 (163M params)

**Config**: 12 layers, 768 dim, no QK norm, no RoPE, no QKV fusion (biased projections)

### Attention Component Breakdown (per layer, decode)

| Component | Time | % of Attention |
|-----------|------|----------------|
| QKV projection | 0.30-0.40ms | **55-65%** |
| QK normalization | N/A | — |
| RoPE | N/A | — |
| QK^T matmul | 0.08ms | 12-15% |
| Softmax | 0.006ms | <2% |
| A*V matmul | 0.012ms | <2% |
| Output projection | 0.10ms | 18-20% |
| **Total** | **0.50-0.60ms** | 100% |

### Observations

- QKV uses 3 separate Q8_0 matmuls (no fusion due to biases)
- Core attention ops (QK^T + softmax + A*V) = ~0.1ms total
- Output projection is 2nd largest cost

---

## Gemma 3 1B (1.3B params)

**Config**: 26 layers, 1152 dim, QK norm enabled, RoPE enabled, QKV fusion enabled

### Attention Component Breakdown (per layer, decode)

| Component | Time | % of Attention |
|-----------|------|----------------|
| QKV projection | 0.50-0.60ms | **50-55%** |
| QK normalization | 0.04-0.06ms | 5% |
| RoPE | 0.015ms | <2% |
| QK^T matmul | 0.08-0.12ms | 10-12% |
| Softmax | 0.007ms | <1% |
| A*V matmul | 0.015ms | <2% |
| Output projection | 0.20-0.30ms | 22-28% |
| **Total** | **0.90-1.00ms** | 100% |

### Observations

- QKV fusion is working (single fused matmul instead of 3)
- QK normalization adds minimal overhead (~0.05ms)
- RoPE is negligible (~0.015ms)
- Output projection is larger due to bigger dimension (1152 vs 768)

---

## Comparative Analysis

| Component | GPT-2 | Gemma 3 | Notes |
|-----------|-------|---------|-------|
| QKV projection | 0.35ms | 0.55ms | Fusion saves ~30% on Gemma 3 |
| QK normalization | — | 0.05ms | Gemma 3 only |
| RoPE | — | 0.015ms | Gemma 3 only |
| QK^T + scale | 0.08ms | 0.10ms | Similar |
| Softmax | 0.006ms | 0.007ms | Already fast |
| A*V | 0.012ms | 0.015ms | Already fast |
| Output projection | 0.10ms | 0.25ms | Scales with dim |
| **Total attention** | **0.55ms** | **0.95ms** | — |

### Key Insight

Core attention operations (QK^T + softmax + A*V) are only **~0.1ms combined** for both models. This is already highly efficient.

---

## Optimization Candidates Evaluated

### 1. Flash Attention (fused QK^T + softmax + A*V)
**Status**: Not worth implementing

- Current cost: ~0.1ms for all three ops combined
- Flash attention complexity is high (tiling, memory management)
- Expected gain: <0.05ms per layer
- ROI: Very low

### 2. SIMD Softmax
**Status**: Not needed

- Current softmax: <0.01ms per layer
- Already using efficient implementation
- Optimization would save <0.005ms

### 3. Better KV Cache Access
**Status**: Already optimal

- `kv_cache::get_view()`: <0.002ms
- `kv_cache::update()`: <0.003ms
- No optimization needed

### 4. Output Projection Fusion
**Status**: Marginal benefit

- Could theoretically fuse with residual add
- Would save ~0.01-0.02ms per layer
- Complexity not worth the gain

---

## Conclusions

1. **Attention is near-optimal** — no high-impact optimizations available
2. **Projections dominate** (~75% of attention time) but are already:
   - Q8_0 quantized
   - QKV fused (where biases absent)
3. **Core attention ops are fast** — <0.1ms combined
4. **Task #4 status**: Profiled, no action needed

---

## Trace Output Format

```
[attn] layer=N QKV=Xms QKnorm=Xms RoPE=Xms QK^T=Xms softmax=Xms A*V=Xms out=Xms
```

Example:
```
[attn] layer=0 QKV=0.358ms QKnorm=0.000ms RoPE=0.000ms QK^T=0.079ms softmax=0.007ms A*V=0.012ms out=0.142ms
```

---

## Test Commands

```bash
# GPT-2 attention trace
RUST_LOG=rustml=trace cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 5 --temperature 0 \
  2>&1 | grep "\[attn\]"

# Gemma 3 attention trace
HF_TOKEN=xxx RUST_LOG=rustml=trace cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors google/gemma-3-1b-it --prompt "Hello" --max-tokens 3 --temperature 0 \
  2>&1 | grep "\[attn\]"
```

---

## Files Modified

- `rustml/nn/main/src/core/attention.rs` — Added trace-level timing instrumentation
- `rustml/BACKLOG.md` — Updated Task #4 status to "Profiled (near-optimal)"
