# Inference Dataflow

> **TLDR:** How a text prompt flows through the crate stack and transformer layers to produce generated text — tensor shapes, KV cache mechanics, and optimization control points.

**Audience**: Developers, ML Engineers

**WHAT**: End-to-end data flow for autoregressive text generation
**WHY**: Understanding tensor shapes, crate boundaries, and optimization control points is essential for debugging, profiling, and extending the inference pipeline
**HOW**: Traces data from CLI input through model loading, prefill, decode, and sampling — with shapes at every stage

---

## Table of Contents

- [Crate Dependency Flow](#crate-dependency-flow)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [Model Loading](#model-loading)
- [Generation Loop](#generation-loop)
- [Per-Token Forward Pass](#per-token-forward-pass)
- [Multi-Head Attention](#multi-head-attention)
- [KV Cache](#kv-cache)
- [Sampling Pipeline](#sampling-pipeline)
- [Tensor Shape Reference](#tensor-shape-reference)
- [Architecture Variants](#architecture-variants)
- [Optimization Profiles](#optimization-profiles)
- [Runtime Optimizations](#runtime-optimizations)

---

## Crate Dependency Flow

Data flows left-to-right through four crates. Each crate boundary is a clean API surface:

```
rustml-cli          rustml-nlp            rustml-nn              rustml-core
(CLI parsing)       (Generation,          (Attention,            (Tensor ops,
                     LlmModel,            TransformerBlock,       softmax,
                     Sampling)            FeedForward,            matmul,
                                          KVCache, RoPE)          RMSNorm SIMD)
                                                │
                                                ▼
                                          rustml-quant
                                          (Q4_0/Q4_1/Q8_0
                                           SIMD dot kernels)
```

| Crate | Responsibility | Key Types |
|-------|---------------|-----------|
| `rustml-cli` | Parse `--prompt`, `--opt-profile`, sampling flags; dispatch to GGUF or SafeTensors path | `InferArgs`, `run_generation()` |
| `rustml-nlp` | Orchestrate generation: tokenize, prefill, decode loop, sample | `LlmModel`, `Generator`, `SamplingBuffer` |
| `rustml-nn` | One transformer layer: norm → attention → residual → FFN → residual | `TransformerBlock`, `MultiHeadAttention`, `FeedForward`, `KVCache` |
| `rustml-core` | Dense math: matmul, softmax, add, RMSNorm; runtime config atomics | `Tensor`, `RuntimeConfig`, `OptProfile` |
| `rustml-quant` | SIMD kernels for quantized weight dot products | `dot_q4q8_block`, `dot_q4_1_q8_block` |

---

## End-to-End Pipeline

```
CLI Input (--prompt "Hello", --temperature 0.7, --top-k 50, --opt-profile optimized)
  │
  ▼
Load Model ──────────────────────────────────────────────── [1]
  │  GGUF: parse binary → extract config + tokenizer + weights
  │  SafeTensors: download from HF → read config.json → load .safetensors
  │  Dispatch by arch: gpt2, llama, gemma3, falcon, mixtral, ...
  │
  ▼
Apply OptProfile ────────────────────────────────────────── [2]
  │  Set global atomics (rayon thresholds)
  │  Set per-layer flags (use_inplace_ops, use_inplace_scaling)
  │  Set generator flag (use_buffered_sampling)
  │
  ▼
Create Generator (temperature, top_k, top_p, repetition_penalty)
  │
  ▼
Encode Prompt ───────────────────────────────────────────── [3]
  │  tokenizer.encode("Hello") → [17534]
  │  Optional: prepend BOS, apply chat template
  │
  ▼
Allocate KV Cache ───────────────────────────────────────── [4]
  │  Size: num_layers × 2 × max_seq × num_kv_heads × head_dim
  │  Auto-sized: min(prompt_len + max_tokens + margin, model_max_seq)
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ PREFILL ─────────────────────────────────────────────── [5] │
│                                                             │
│   model.forward_with_cache([all prompt tokens], cache)      │
│   Input:  Tensor [1, seq_len]                               │
│   Output: logits [1, seq_len, vocab_size]                   │
│   Cache:  populated with K/V for all prompt positions       │
│                                                             │
│   → Extract last-position logits → sample first token       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ DECODE LOOP (repeat until EOS or max_tokens) ────────  [6] │
│                                                             │
│   model.forward_with_cache([1 token], cache)                │
│   Input:  Tensor [1, 1]                                     │
│   Output: logits [1, 1, vocab_size]                         │
│   Cache:  appends 1 K/V entry per layer                     │
│                                                             │
│   → sample_token(logits, past_tokens)                       │
│   → tokenizer.decode(token_id) → print (if streaming)      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
Output: "Hello world! How are you today?"
```

---

## Model Loading

Architecture dispatch happens at model construction. All paths produce a unified `LlmModel`:

```
GGUF file ──parse──▶ GgufConfig ──▶ match arch {
                                       "llama"   → from_pretrained()
                                       "gemma3"  → from_pretrained_gemma3()
                                       "falcon"  → from_pretrained_falcon()
                                       "gpt2"    → from_pretrained_gpt2()
                                       "mixtral" → from_pretrained_mixtral()
                                     }

SafeTensors ──download──▶ config.json ──▶ match model_type {
                                             "gpt2"    → map_gpt2_weights() + from_pretrained_gpt2()
                                             "llama"   → map_llama_weights() + from_pretrained()
                                             "gemma3"  → map_gemma3_weights() + from_pretrained_gemma3()
                                             ...
                                           }
```

### LlmModel Structure

```
LlmModel
├── token_embedding: Embedding [vocab_size, d_model]
├── pos_embedding: Option<Embedding> [max_seq, d_model]   (learned, GPT-2 only)
├── layers: Vec<TransformerBlock>                          (N layers)
│     ├── attention: MultiHeadAttention
│     │     ├── q_proj: Linear [d_model, d_model]
│     │     ├── k_proj: Linear [d_model, kv_dim]
│     │     ├── v_proj: Linear [d_model, kv_dim]
│     │     ├── out_proj: Linear [d_model, d_model]
│     │     ├── rope: Option<RoPEFreqs>
│     │     ├── q_norm, k_norm: Option<RMSNorm>            (Gemma 3)
│     │     └── use_inplace_scaling: bool                   (OptProfile)
│     ├── feed_forward: FeedForward
│     │     ├── gate_proj: Linear [d_model, ffn_dim]
│     │     ├── up_proj: Linear [d_model, ffn_dim]
│     │     └── down_proj: Linear [ffn_dim, d_model]
│     ├── attention_norm: RMSNorm | LayerNorm [d_model]
│     ├── ffn_norm: RMSNorm | LayerNorm [d_model]
│     ├── post_attention_norm: Option (Gemma 3 sandwich)
│     ├── post_ffn_norm: Option (Gemma 3 sandwich)
│     ├── moe: Option<MoeLayer>                             (Mixtral)
│     └── use_inplace_ops: bool                             (OptProfile)
├── norm: RMSNorm | LayerNorm [d_model]                    (final)
└── output: Linear [d_model, vocab_size]                   (tied to embedding in some models)
```

---

## Generation Loop

`Generator.generate()` orchestrates the full autoregressive pipeline:

```
generate(prompt, max_tokens)
  │
  ├── encode_prompt(prompt) → token_ids: Vec<u32>
  │     ├── apply chat template (if configured)
  │     ├── tokenizer.encode(text) → ids
  │     └── prepend BOS (if model requires it)
  │
  ├── make_cache() → KVCache
  │     └── pre-allocate [layers × 2 × max_seq × kv_heads × head_dim]
  │
  ├── SamplingBuffer::new(vocab_size)
  │     └── pre-allocate logit + index buffers (reused across steps)
  │
  ├── prefill(token_ids, cache)
  │     ├── model.forward_with_cache(token_ids, cache)
  │     ├── cache.advance(seq_len)
  │     └── return logits[-1] (last position only)
  │
  └── decode loop:
        ├── token_id = sample_token(logits, past_tokens, sampling_buf)
        ├── past_tokens.push(token_id)
        ├── if token_id == eos_token: break
        ├── logits = model.forward_with_cache([token_id], cache)
        ├── cache.advance(1)
        └── if streaming: tokenizer.decode(token_id) → print
```

---

## Per-Token Forward Pass

Inside `LlmModel.forward_with_cache()`, data flows through the embedding, layer stack, and output projection:

```
token_ids [1, S]
  │
  │  token_embedding.forward()
  ▼
hidden [1, S, d_model]                                     ← embedding lookup
  │
  │  + pos_embedding (GPT-2: learned positions)
  │  × embedding_scale (Gemma: sqrt(d_model))
  ▼
hidden [1, S, d_model]
  │
  │  ┌──────────────────────────────────────────────────┐
  │  │ Layer 0..N-1: TransformerBlock.forward_with_cache │
  │  │                                                    │
  │  │   residual = hidden.clone()                        │
  │  │                                                    │
  │  │   hidden = attention_norm(hidden)                  │
  │  │         ▼                                          │
  │  │   hidden = attention(hidden, cache, layer_idx)     │
  │  │         ▼                                          │
  │  │   hidden = post_attention_norm(hidden)  (if any)   │
  │  │         ▼                                          │
  │  │   hidden += residual           (residual add)      │
  │  │         ▼                                          │
  │  │   residual = hidden.clone()                        │
  │  │                                                    │
  │  │   hidden = ffn_norm(hidden)                        │
  │  │         ▼                                          │
  │  │   hidden = feed_forward(hidden)                    │
  │  │         ▼                                          │
  │  │   hidden = post_ffn_norm(hidden)    (if any)       │
  │  │         ▼                                          │
  │  │   hidden += residual           (residual add)      │
  │  │                                                    │
  │  │   Output: hidden [1, S, d_model]                   │
  │  └──────────────────────────────────────────────────┘
  │    ... repeated N times ...
  ▼
hidden [1, S, d_model]
  │
  │  final norm (RMSNorm or LayerNorm)
  ▼
hidden [1, S, d_model]
  │
  │  output.forward() (linear: hidden @ W^T)
  ▼
logits [1, S, vocab_size]
```

---

## Multi-Head Attention

The attention module computes scaled dot-product attention with optional GQA, RoPE, sliding window, and logit capping:

```
input [1, S_new, d_model]
  │
  │  ┌─ If fused_qkv available (Q8_0, no biases):
  │  │    fused_qkv(input) → [q|k|v], then slice
  │  │
  │  └─ Otherwise (separate projections):
  ├── q_proj(input) → q [1, S_new, d_model]
  ├── k_proj(input) → k [1, S_new, kv_dim]
  └── v_proj(input) → v [1, S_new, kv_dim]
        │
        │  reshape to heads + transpose
        ▼
      q [1, num_heads, S_new, head_dim]
      k [1, num_kv_heads, S_new, head_dim]
      v [1, num_kv_heads, S_new, head_dim]
        │
        │  QK normalization (Gemma 3: per-head RMSNorm)
        │  RoPE rotation (encode absolute position)
        ▼
      q [1, num_heads, S_new, head_dim]       (rotated)
      k [1, num_kv_heads, S_new, head_dim]    (rotated)
        │
        │  cache.update(layer_idx, k, v)   ← store new K/V
        │  cache.get_view(layer_idx)       ← retrieve full history
        ▼
      k_full [1, num_kv_heads, total_len, head_dim]
      v_full [1, num_kv_heads, total_len, head_dim]
        │
        │  GQA expansion: repeat K/V to match Q heads
        │  n_rep = num_heads / num_kv_heads
        ▼
      k_full [1, num_heads, total_len, head_dim]
      v_full [1, num_heads, total_len, head_dim]
        │
        │  scores = q @ k_full^T
        ▼
      scores [1, num_heads, S_new, total_len]
        │
        │  scale: scores *= 1/sqrt(head_dim)    (or /= sqrt(head_dim))
        │  logit cap: cap * tanh(scores / cap)  (Gemma 2 only)
        │  mask: + causal_mask or sliding_window_mask or ALiBi bias
        ▼
      scores [1, num_heads, S_new, total_len]   (masked)
        │
        │  softmax(dim=-1)
        ▼
      attn_weights [1, num_heads, S_new, total_len]
        │
        │  context = attn_weights @ v_full
        ▼
      context [1, num_heads, S_new, head_dim]
        │
        │  transpose(1,2) → reshape → [1, S_new, d_model]
        │  out_proj(context)
        ▼
      output [1, S_new, d_model]
```

### Decode Step (S_new=1)

During autoregressive decoding, only the new token attends to the full cached history:

```
q [1, H, 1, head_dim]  ×  k_full [1, H, T, head_dim]^T  →  scores [1, H, 1, T]
                                                              → softmax
                                                              → @ v_full [1, H, T, head_dim]
                                                              → context [1, H, 1, head_dim]
```

This is O(T) per decode step instead of O(T^2) — the KV cache avoids recomputing K/V for past positions.

---

## KV Cache

Pre-allocated buffer that stores K and V tensors from all previous positions, eliminating redundant computation:

```
KVCache {
  k_cache: [num_layers][max_seq_len][num_kv_heads][head_dim]
  v_cache: [num_layers][max_seq_len][num_kv_heads][head_dim]
  current_len: usize      ← number of positions filled so far
}
```

### Cache Operations

```
Prefill (seq_len = 5):
  ┌───────────────────────────────────────────────┐
  │ k_cache[layer]:                               │
  │ pos: [0] [1] [2] [3] [4] [5] [6] ... [max]   │
  │      ███ ███ ███ ███ ███                       │  ← filled
  │ current_len = 5                                │
  └───────────────────────────────────────────────┘

Decode step 1 (1 token):
  ┌───────────────────────────────────────────────┐
  │ pos: [0] [1] [2] [3] [4] [5] [6] ... [max]   │
  │      ███ ███ ███ ███ ███ ███                   │  ← append 1
  │ current_len = 6                                │
  └───────────────────────────────────────────────┘

Decode step 2 (1 token):
  ┌───────────────────────────────────────────────┐
  │ pos: [0] [1] [2] [3] [4] [5] [6] ... [max]   │
  │      ███ ███ ███ ███ ███ ███ ███               │  ← append 1
  │ current_len = 7                                │
  └───────────────────────────────────────────────┘
```

### Memory Budget

```
KV cache bytes = num_layers × 2 × max_seq × num_kv_heads × head_dim × 4 (f32)

Example (Gemma 3 1B, 26 layers, 4 KV heads, head_dim=256, max_seq=2048):
  = 26 × 2 × 2048 × 4 × 256 × 4 = ~436 MB
```

---

## Sampling Pipeline

Converts raw logits into a token ID. Applied once per decode step:

```
logits [vocab_size]
  │
  │  temperature == 0 ?
  ├── YES → argmax(logits) → token_id                     (greedy)
  │
  ├── NO ↓
  │
  │  repetition_penalty(logits, past_tokens)
  │    for each tok in past_tokens:
  │      logits[tok] > 0 → logits[tok] /= penalty
  │      logits[tok] ≤ 0 → logits[tok] *= penalty
  ▼
logits [vocab_size]                                         (penalized)
  │
  │  logits /= temperature
  ▼
logits [vocab_size]                                         (scaled)
  │
  │  top_k(logits, k)
  │    keep top k values, set rest to -inf
  ▼
logits [vocab_size]                                         (top-k filtered)
  │
  │  top_p(logits, p)  (nucleus sampling)
  │    sort descending → softmax → cumsum
  │    keep smallest set where cumsum ≥ p, set rest to -inf
  ▼
logits [vocab_size]                                         (top-p filtered)
  │
  │  softmax(logits) → probabilities
  │  categorical_sample(probabilities) → token_id
  ▼
token_id: u32
```

### Buffered vs Allocating Sampling

When `use_buffered_sampling=true` (default), the sampling pipeline reuses pre-allocated buffers:

```
SamplingBuffer {
  logits: Vec<f32>     [vocab_size]   ← reused every step
  indices: Vec<usize>  [vocab_size]   ← reused every step
}
```

When `false` (baseline profile), each step allocates fresh `Vec`s — useful for correctness comparison.

---

## Tensor Shape Reference

### GPT-2 (d=768, 12 heads, 12 layers, vocab=50257)

| Stage | Prefill (S=32) | Decode (S=1) |
|-------|---------------|-------------|
| Input token IDs | `[1, 32]` | `[1, 1]` |
| After embedding | `[1, 32, 768]` | `[1, 1, 768]` |
| Q projection | `[1, 12, 32, 64]` | `[1, 12, 1, 64]` |
| K/V projection | `[1, 12, 32, 64]` | `[1, 12, 1, 64]` |
| K/V from cache | — | `[1, 12, T, 64]` |
| Attention scores | `[1, 12, 32, 32]` | `[1, 12, 1, T]` |
| Attention output | `[1, 32, 768]` | `[1, 1, 768]` |
| FFN hidden | `[1, 32, 3072]` | `[1, 1, 3072]` |
| FFN output | `[1, 32, 768]` | `[1, 1, 768]` |
| Logits | `[1, 32, 50257]` | `[1, 1, 50257]` |

### Gemma 3 1B (d=1152, 4 heads, head_dim=256, 26 layers, vocab=262144)

| Stage | Prefill (S=32) | Decode (S=1) |
|-------|---------------|-------------|
| Input token IDs | `[1, 32]` | `[1, 1]` |
| After embedding (×sqrt(1152)) | `[1, 32, 1152]` | `[1, 1, 1152]` |
| Q projection | `[1, 4, 32, 256]` | `[1, 4, 1, 256]` |
| K/V (GQA, 1 KV head) | `[1, 1, 32, 256]` | `[1, 1, 1, 256]` |
| K/V after GQA expand | `[1, 4, 32, 256]` | `[1, 4, T, 256]` |
| Attention scores | `[1, 4, 32, 32]` | `[1, 4, 1, T]` |
| FFN hidden (GeGLU) | `[1, 32, 6144]` | `[1, 1, 6144]` |
| Logits | `[1, 32, 262144]` | `[1, 1, 262144]` |

---

## Architecture Variants

All architectures produce a `LlmModel` with the same forward path. Differences are in configuration:

| Feature | GPT-2 | Llama/Mistral | Falcon | Mixtral | Gemma 3 |
|---------|-------|---------------|--------|---------|---------|
| Position encoding | Learned | RoPE | RoPE or ALiBi | RoPE | RoPE (dual local/global) |
| Normalization | LayerNorm + bias | RMSNorm | LayerNorm | RMSNorm | RMSNorm (sandwich: 4 per block) |
| FFN activation | GELU | SwiGLU | GELU | SwiGLU | GeGLU |
| KV heads (GQA) | H (no GQA) | H_kv ≤ H | H_kv ≤ H | H_kv ≤ H | H_kv ≤ H |
| QK norms | No | No | No | No | Yes (per-head RMSNorm) |
| Sliding window | No | No (Mistral: Yes) | No | No | Yes (local layers) |
| Logit cap | No | No | No | No | Yes (Gemma 2) |
| Residual pattern | Standard | Standard | Parallel | Standard | Standard (sandwich) |
| MoE | No | No | No | Yes (8 experts) | No |
| Weight tying | Yes (output=embed) | No | No | No | Yes |
| Embedding scale | No | No | No | No | Yes (sqrt(d_model)) |
| Attention bias | Yes | No | Yes | No | No (Qwen 2: optional) |

---

## Optimization Profiles

Three profiles control the runtime behavior of the inference pipeline. Set via `--opt-profile`:

```
OptProfile::Optimized (default)          OptProfile::Baseline
├── rayon thresholds: 4096               ├── rayon thresholds: MAX (always sequential)
├── use_inplace_ops: true                ├── use_inplace_ops: false (always allocate)
├── use_inplace_scaling: true            ├── use_inplace_scaling: false
└── use_buffered_sampling: true          └── use_buffered_sampling: false

OptProfile::Aggressive
├── rayon thresholds: 1024 (parallelize earlier)
├── use_inplace_ops: true
├── use_inplace_scaling: true
└── use_buffered_sampling: true
```

### Where Optimizations Apply

```
Generator.sample_token()
  │  use_buffered_sampling ──▶ reuse SamplingBuffer vs allocate Vec
  │
  ▼
LlmModel.forward_with_cache()
  │
  ├── TransformerBlock.forward_with_cache()
  │     │  use_inplace_ops ──▶ clone+forward_inplace+add_inplace vs forward()+add()
  │     │
  │     ├── MultiHeadAttention.forward_with_cache()
  │     │     │  use_inplace_scaling ──▶ mul_scalar_inplace(1/scale) vs div_scalar(scale)
  │     │     │
  │     │     ├── Tensor.softmax()
  │     │     │     │  SOFTMAX_PAR_THRESHOLD ──▶ sequential vs rayon parallel
  │     │     │
  │     │     └── Tensor.batched_matmul()
  │     │           │  BATCHED_MATMUL_PAR_THRESHOLD ──▶ sequential vs rayon parallel
  │     │
  │     └── FeedForward.forward()
  │
  └── ... repeated for N layers
```

All profiles produce numerically identical output under greedy decoding (temperature=0).

---

## Runtime Optimizations

Beyond optimization profiles, several runtime optimizations reduce inference latency:

### Q8_0 Quantization

Linear layers are automatically quantized from F32 to Q8_0 at model load time. The dimension threshold determines which layers qualify:

```
Quantization rule:
  max(in_features, out_features) >= 768  →  quantize to Q8_0
  max(in_features, out_features) < 768   →  keep F32
```

| Model | d_model | Layers Quantized | Memory Reduction |
|-------|---------|------------------|------------------|
| GPT-2 | 768 | 73 (all linear) | ~75% |
| Gemma 3 1B | 1152 | 183 (all linear) | ~75% |

### Weight Fusion

After quantization, compatible weight pairs are fused to reduce matmul dispatch overhead:

**Gate+Up Fusion (SwiGLU/GeGLU layers):**
```
Before: gate_proj(x), up_proj(x)    →  2 matmul dispatches
After:  fused_gate_up(x)            →  1 matmul dispatch, slice output
```

**QKV Fusion (Attention layers, requires no bias):**
```
Before: q_proj(x), k_proj(x), v_proj(x)  →  3 matmul dispatches
After:  fused_qkv(x)                      →  1 matmul dispatch, slice output
```

| Model | Gate+Up Fused | QKV Fused | Reason |
|-------|---------------|-----------|--------|
| GPT-2 | 0 | 0 | No gate_proj; biased Q/K/V |
| Gemma 3 1B | 26 | 26 | GeGLU; no biases |

### Thread Pool Warmup

Before timed inference, `RuntimeConfig::warmup_thread_pool()` pre-warms the rayon thread pool:

```
Warmup actions:
1. Spawn all rayon threads (avoid first-use latency)
2. Exercise SIMD code paths (warm instruction cache)
3. Touch memory buffers (populate TLB entries)
```

**Impact on jitter:**
| Metric | Before Warmup | After Warmup | Change |
|--------|---------------|--------------|--------|
| Layer variance | 5.1x | 2.3x | -55% |
| Step variance | 1.9x | 1.4x | -26% |

### Attention Component Timing

Trace-level profiling (`RUST_LOG=rustml=trace`) breaks down attention into 7 components:

```
[attn] layer=N QKV=<T>ms QKnorm=<T>ms RoPE=<T>ms QK^T=<T>ms softmax=<T>ms A*V=<T>ms out=<T>ms
```

| Component | Description | Typical % |
|-----------|-------------|-----------|
| QKV | Q/K/V projections (fused or separate) | 50-60% |
| QKnorm | QK normalization (Gemma 3 only) | 5% |
| RoPE | Rotary position encoding | <2% |
| QK^T | Attention scores + scaling | 10-15% |
| softmax | Attention weights | <2% |
| A*V | Weighted value aggregation | <2% |
| out | Output projection | 18-25% |

**Key insight:** Core attention ops (QK^T + softmax + A*V) total <0.1ms per layer. Projections dominate.

### Prefill vs Decode Efficiency

Prefill processes multiple tokens in one forward pass; decode processes one token at a time:

| Model | Prefill (per-token) | Decode | Efficiency |
|-------|---------------------|--------|------------|
| GPT-2 (10 tokens) | ~38ms | ~50ms | Prefill 24% faster |
| Gemma 3 (10 tokens) | ~160ms | ~178ms | Prefill 10% faster |

Prefill is more efficient due to better memory bandwidth utilization and cache locality.

---

## See Also

- [Tokenizer–Weight Integration](guides/tokenizer_weight_integration.md) — how tokenization connects to model weights through the embedding matrix
- [Project Structure](project_structure.md) — crate layout and SEA layering conventions
- [ADR-001: Unified LlmModel](adr/adr-001-unified-llmmodel-for-gpt2.md) — why GGUF and SafeTensors share one model path
- [Model Verification Guide](../4-development/guides/model-verification.md) — verifying correctness across the pipeline
- [Manual Inference Tests](../5-testing/manual_infer_tests.md) — test procedures including optimization profiles and profiling methodology
- [FFN Architectures](guides/ffn_architectures.md) — Standard FFN vs SwiGLU/GeGLU, parameter math, gate+up fusion
