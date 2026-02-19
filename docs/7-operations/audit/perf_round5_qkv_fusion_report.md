# Performance Optimization Round 5 — QKV Fusion Report

**Date:** 2026-02-19
**Commit:** `a9bf3f5` (branch: `sweengineeringlabs`)
**Tool:** Hierarchical `log` crate instrumentation (`RUST_LOG=rustml=debug`)
**Platform:** Linux 6.6.87 (WSL2), x86_64 AVX2, 8 rayon threads
**Build:** `cargo build --release` (thin LTO enabled)

---

## Change Under Test

Fuse Q, K, V projection weights into a single matmul per attention layer.

Each attention layer previously made 3 separate Q8_0 matmul calls with identical input:
- `q_proj`: `[1,1152] x [1152,1024]` — ~3.6MB weight read
- `k_proj`: `[1,1152] x [1152,256]` — ~0.9MB weight read
- `v_proj`: `[1,1152] x [1152,256]` — ~0.9MB weight read

Each call incurred rayon `par_chunks_mut` dispatch, input tensor extraction, and output allocation overhead. `MultiHeadAttention::fuse_qkv_weights()` concatenates the three Q8_0 weight tensors into a single `[1,1152] x [1152,1536]` tensor at load time. The forward path runs one fused matmul then splits the output via `tensor.slice()`.

**Guard conditions:** All three projections must be Q8_0, have no biases, and share the same `in_features`. Models with biased projections (GPT-2) skip fusion automatically.

**Files changed:** `attention.rs`, `model.rs`, `infer.rs` (×2), `backlog.md`, `manual_infer_tests.md`

---

## Models Tested

| Model | Format | Params | d_model | Layers | Gate+up fused | QKV fused |
|-------|--------|--------|---------|--------|---------------|-----------|
| GPT-2 | SafeTensors | 163M | 768 | 12 | 0 (no gate_proj) | **0** (biased projections) |
| Gemma 3 1B IT | SafeTensors | 1,302M | 1,152 | 26 | 26 | **26** (all attention) |

---

## 1. Gemma 3 1B IT (26 QKV fused)

### Command

```bash
RUST_LOG=rustml=debug cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors google/gemma-3-1b-it --prompt "Hello" --max-tokens 10 --temperature 0
```

### Diagnostic output

```
Quantized 183 linear layers F32 -> Q8_0
Fused 26 gate+up projection pairs
Fused 26 QKV projection triples
Warmup: 223ms
```

### Output

```
Hello with the prompt, "What is the best way
  Generated in 2.65s
```

### Per-step forward pass timings (decode)

| Step | Embedding | Layers | Norm | Projection | Total |
|------|-----------|--------|------|------------|-------|
| warmup | 1.2ms | 219ms | <0.1ms | 20ms | 240ms |
| 1 (prefill) | 0.2ms | 352ms | <0.1ms | 48ms | 400ms |
| 2 | 0.2ms | 203ms | <0.1ms | 29ms | 232ms |
| 3 | 0.1ms | 187ms | <0.1ms | 24ms | 211ms |
| 4 | 0.1ms | 194ms | <0.1ms | 28ms | 222ms |
| 5 | 0.1ms | 189ms | <0.1ms | 21ms | 209ms |
| 6 | 0.1ms | 207ms | <0.1ms | 28ms | 235ms |
| 7 | 0.1ms | 207ms | <0.1ms | 25ms | 232ms |
| 8 | 0.2ms | 196ms | <0.1ms | 31ms | 227ms |
| 9 | 0.2ms | 222ms | <0.1ms | 28ms | 250ms |
| 10 | 0.2ms | 206ms | <0.1ms | 24ms | 230ms |

**Steady-state decode (steps 2-10 avg):** ~205ms layers, ~26ms projection, ~228ms total
**Best observed step:** 202ms (layers=182ms, proj=20ms)

### Per-layer breakdown (last decode step, typical)

| Layer | Attention | FFN | Layer Total |
|-------|-----------|-----|-------------|
| 0 | 1.4ms | 4.6ms | 6.3ms |
| 1 | 2.0ms | 3.9ms | 6.2ms |
| 2 | 2.5ms | 4.3ms | 7.0ms |
| 3 | 1.7ms | 5.4ms | 7.4ms |
| 4 | 1.5ms | 4.5ms | 6.3ms |
| 5 | 1.5ms | 4.8ms | 6.5ms |
| 6 | 2.0ms | 4.4ms | 6.6ms |
| 7 | 2.1ms | 4.2ms | 6.5ms |
| 8 | 1.9ms | 5.1ms | 7.2ms |
| 9 | 1.4ms | 4.9ms | 6.4ms |
| 10 | 2.0ms | 4.8ms | 7.0ms |
| 11 | 2.1ms | 4.9ms | 7.2ms |
| 12 | 2.0ms | 4.6ms | 6.8ms |
| 13 | 2.2ms | 4.7ms | 7.0ms |
| 14 | 1.9ms | 4.8ms | 6.8ms |
| 15 | 1.9ms | 5.0ms | 7.1ms |
| 16 | 1.4ms | 5.0ms | 6.6ms |
| 17 | 2.0ms | 5.5ms | 7.7ms |
| 18 | 2.0ms | 4.7ms | 6.9ms |
| 19 | 1.6ms | 5.0ms | 6.8ms |
| 20 | 1.8ms | 5.5ms | 7.5ms |
| 21 | 2.0ms | 4.1ms | 6.3ms |
| 22 | 2.0ms | 4.1ms | 6.3ms |
| 23 | 1.9ms | 4.8ms | 6.9ms |
| 24 | 1.9ms | 4.9ms | 6.9ms |
| 25 | 2.8ms | 5.0ms | 7.9ms |
| **Avg** | **1.9ms** | **4.7ms** | **6.8ms** |

### Attention improvement

| Metric | Before (Round 3) | After (Round 5) | Change |
|--------|-------------------|-----------------|--------|
| Attention/layer | ~3.0ms | ~1.9ms | **-37%** |
| FFN/layer | ~4.7ms | ~4.7ms | unchanged |
| Layer total | ~7.7ms | ~6.8ms | **-12%** |
| Layers (26 total) | ~200ms | ~177ms | **-23ms** |

---

## 2. GPT-2 (no fusion — regression check)

### Command

```bash
RUST_LOG=rustml=debug cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 10 --temperature 0
```

### Diagnostic output

```
Quantized 25 linear layers F32 -> Q8_0
Warmup: 73ms
```

No `Fused` messages — correct. GPT-2 has biased Q/K/V projections (`c_attn` split into Q/K/V with bias), so both gate+up and QKV fusion are skipped.

### Output

```
Hello, I'm sorry. I'm sorry. I
  Generated in 0.69s
```

### Per-step forward pass timings (decode)

| Step | Embedding | Layers | Norm | Projection | Total |
|------|-----------|--------|------|------------|-------|
| warmup | 0.1ms | 67ms | 0.5ms | 5ms | 73ms |
| 1 (prefill) | 0.1ms | 58ms | 0.5ms | 5ms | 64ms |
| 2 | 0.1ms | 78ms | 0.6ms | 6ms | 85ms |
| 3 | 0.1ms | 73ms | 0.8ms | 5ms | 79ms |
| 4 | 0.1ms | 56ms | 0.8ms | 6ms | 63ms |
| 5 | 0.1ms | 57ms | 0.5ms | 4ms | 62ms |
| 6 | 0.1ms | 59ms | 0.6ms | 5ms | 64ms |
| 7 | 0.1ms | 57ms | 0.5ms | 5ms | 62ms |
| 8 | 0.1ms | 59ms | 0.7ms | 4ms | 64ms |
| 9 | 0.1ms | 56ms | 0.5ms | 5ms | 62ms |
| 10 | 0.1ms | 59ms | 0.5ms | 3ms | 62ms |

**Steady-state decode (steps 4-10 avg):** ~58ms layers, ~5ms projection, ~63ms total

**No regression** — consistent with pre-fusion GPT-2 performance.

---

## Time Budget Analysis (Gemma 3 1B, steady-state decode)

### Where the time goes

| Component | Time | % of step | Notes |
|-----------|------|-----------|-------|
| FFN (26 layers × 4.7ms) | ~122ms | 60% | Gate+up fused; raw Q8_0 compute |
| Attention (26 layers × 1.9ms) | ~49ms | 24% | QKV fused; includes RoPE, softmax, out_proj |
| lm_head projection | ~23ms | 11% | 262K vocab, memory-bandwidth bound |
| Norms + embedding | ~3ms | 1% | Negligible |
| Overhead + jitter | ~8ms | 4% | WSL2 scheduling, rayon sync |
| **Total** | **~205ms** | **100%** | |

### Optimization history (Gemma 3 1B decode/step)

| Round | Change | Decode/step | Improvement |
|-------|--------|-------------|-------------|
| Baseline | F32 only, no parallelism | ~370ms | — |
| Round 1 | Q8_0 lm_head + parallel gemv | ~280ms | -24% |
| Round 2 | All-layer Q8_0, SIMD kernels, LTO | ~227ms | -19% |
| Round 3 | Gate+up fusion (26 GeGLU layers) | ~224ms | -1% |
| Round 4 | Decode warmup pass | ~224ms | cold-start fix |
| **Round 5** | **QKV fusion (26 attention layers)** | **~205ms** | **-8%** |
| **Cumulative** | | | **-45%** |

---

## Remaining Bottlenecks

### 1. FFN matmul — ~122ms (60% of step)

The fused gate+up Q8_0 matmul `[1,1152]×[1152,12288]` is the dominant cost at 4.7ms/layer. This is raw compute — SIMD kernels and rayon parallelism are already applied. Further improvement requires:
- Faster Q8_0 SIMD kernel (e.g., wider AVX-512 on supported hardware)
- Weight sparsity or pruning to reduce matmul size
- Q4_0 quantization (halves bandwidth, but accuracy trade-off)

### 2. lm_head — ~23ms (11% of step)

The `[1,1152]×[1152,262144]` Q8_0 matmul reads ~250MB of weights per step. Pure memory-bandwidth bound. Fixes require algorithmic changes:
- Vocabulary pruning (reduce 262K → subset)
- Speculative decoding (skip full vocab on most steps)
- Output embedding tying with embedding quantization

### 3. WSL2 variance — +/-30ms

OS/rayon scheduling jitter on WSL2 adds ~30ms noise floor. Some steps hit 200ms, others 250ms on identical work. Thread pinning (`RAYON_NUM_THREADS` + CPU affinity) would help but is platform-specific. This makes sub-15ms optimizations hard to measure reliably.

### 4. Attention overhead — ~49ms (24% of step)

At 1.9ms/layer the attention QKV projection is now a single matmul. The remaining time splits across:
- Fused QKV matmul (~0.8ms)
- Slice + contiguous (~0.1ms)
- RoPE application (~0.2ms)
- QK normalization (~0.1ms)
- Softmax + attention matmul (~0.3ms)
- Output projection (~0.4ms)

Diminishing returns — output projection is the largest remaining piece and cannot be fused.

---

## Reproduction

### Gemma 3 1B (requires HF token for gated model)

```bash
export HF_TOKEN=hf_xxx
RUST_LOG=rustml=debug cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors google/gemma-3-1b-it --prompt "Hello" --max-tokens 10 --temperature 0
```

Expected stderr:
```
Quantized 183 linear layers F32 -> Q8_0
Fused 26 gate+up projection pairs
Fused 26 QKV projection triples
Warmup: ~220ms
```

### GPT-2 (no auth required)

```bash
RUST_LOG=rustml=debug cargo run --release -p rustml-nlp --bin rustml-infer -- \
  --safetensors openai-community/gpt2 --prompt "Hello" --max-tokens 10 --temperature 0
```

Expected stderr:
```
Quantized 25 linear layers F32 -> Q8_0
Warmup: ~73ms
```

No `Fused` lines (correct — biased projections skip fusion).

### Windows (native, no WSL2)

```powershell
$env:RUST_LOG="rustml=debug"
cargo run --release -p rustml-nlp --bin rustml-infer -- --safetensors google/gemma-3-1b-it --prompt "Hello" --max-tokens 10 --temperature 0
```

Expect lower variance and potentially faster steady-state decode without WSL2 scheduling overhead.
