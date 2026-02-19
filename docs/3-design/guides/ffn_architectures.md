# FFN Architectures and Layer Counting

> How feed-forward networks differ across transformer models and how to count quantizable layers.

## Standard FFN (GPT-2, BERT)

The original transformer FFN uses two linear projections with an activation in between:

```
x → up_proj → activation → down_proj → output
```

**Computation**:
```
output = down_proj( ReLU( up_proj(x) ) )
```

**Matrices** (2 per layer):
| Projection | Dimensions | Purpose |
|------------|------------|---------|
| up_proj | dim → 4×dim | Expand to higher dimension |
| down_proj | 4×dim → dim | Project back to model dimension |

**Example** (GPT-2, dim=768):
- up_proj: 768 → 3072
- down_proj: 3072 → 768

---

## SwiGLU FFN (LLaMA, Gemma, Mistral)

**SwiGLU** = **Swi**sh + **G**ated **L**inear **U**nit

Introduced by Noam Shazeer (Google, 2020). Adds a gating mechanism that controls information flow.

```
x → gate_proj → swish ─┐
                       ├─ element-wise multiply → down_proj → output
x → up_proj ───────────┘
```

**Computation**:
```
output = down_proj( swish(gate_proj(x)) * up_proj(x) )
```

Where `swish(x) = x * sigmoid(x)`

**Matrices** (3 per layer):
| Projection | Dimensions | Purpose |
|------------|------------|---------|
| gate_proj | dim → hidden | Controls what passes through |
| up_proj | dim → hidden | Value to be gated |
| down_proj | hidden → dim | Project back to model dimension |

**Example** (Gemma 3 1B, dim=1152, hidden=4×1152=4608):
- gate_proj: 1152 → 4608
- up_proj: 1152 → 4608
- down_proj: 4608 → 1152

---

## Parameter Count Math

For a model with dimension `d` and FFN hidden size `h = 4d` (typical expansion factor):

**Standard FFN**:
```
up_proj:   d × h = d × 4d = 4d²
down_proj: h × d = 4d × d = 4d²
─────────────────────────────────
Total:                      8d²
```

**SwiGLU FFN**:
```
gate_proj: d × h = d × 4d = 4d²
up_proj:   d × h = d × 4d = 4d²
down_proj: h × d = 4d × d = 4d²
─────────────────────────────────
Total:                     12d²
```

**Example** (GPT-2, d=768):
- Standard FFN: 8 × 768² = 8 × 589,824 = **4.7M parameters per layer**

**Example** (Gemma 3 1B, d=1152):
- SwiGLU FFN: 12 × 1152² = 12 × 1,327,104 = **15.9M parameters per layer**

---

## Why SwiGLU?

| Aspect | Standard FFN | SwiGLU |
|--------|--------------|--------|
| Matrices per layer | 2 | 3 |
| Parameters | 8d² | 12d² |
| Gradient flow | ReLU can "die" | Swish is smooth |
| Gating | None | Learned |
| Quality | Baseline | +1-2% on benchmarks |

The gating mechanism allows the model to learn which features to amplify or suppress, improving expressiveness at the cost of 50% more FFN parameters.

---

## Counting Quantizable Layers

The `quantize_all_weights` function iterates through all `Linear` layers and counts those that meet the dimension threshold.

### Transformer Layer Structure

```
TransformerBlock
├── Attention
│   ├── q_proj    (Linear)
│   ├── k_proj    (Linear)
│   ├── v_proj    (Linear)
│   └── out_proj  (Linear)
├── FFN (Standard)
│   ├── up_proj   (Linear)
│   └── down_proj (Linear)
└── FFN (SwiGLU)
    ├── gate_proj (Linear)
    ├── up_proj   (Linear)
    └── down_proj (Linear)

Model
├── embedding     (not quantized)
├── layers[]      (N transformer blocks)
└── output        (Linear, lm_head)
```

### Quantization Threshold

A layer is quantized if:
```rust
max(in_features, out_features) >= min_dim
```

### Example: GPT-2 (12 layers, dim=768)

| Layer | Dimensions | max() | Quantized (min=768)? |
|-------|------------|-------|----------------------|
| q_proj | 768→768 | 768 | Yes |
| k_proj | 768→768 | 768 | Yes |
| v_proj | 768→768 | 768 | Yes |
| out_proj | 768→768 | 768 | Yes |
| up_proj | 768→3072 | 3072 | Yes |
| down_proj | 3072→768 | 3072 | Yes |
| output | 768→50257 | 50257 | Yes |

**Total**: (4 attention + 2 FFN) × 12 layers + 1 output = **73 layers**

### Example: Gemma 3 1B (26 layers, dim=1152)

| Layer | Dimensions | max() | Quantized (min=1152)? |
|-------|------------|-------|----------------------|
| q_proj | 1152→1152 | 1152 | Yes |
| k_proj | 1152→1152 | 1152 | Yes |
| v_proj | 1152→1152 | 1152 | Yes |
| out_proj | 1152→1152 | 1152 | Yes |
| gate_proj | 1152→4608 | 4608 | Yes |
| up_proj | 1152→4608 | 4608 | Yes |
| down_proj | 4608→1152 | 4608 | Yes |
| output | 1152→vocab | vocab | Yes |

**Total**: (4 attention + 3 FFN) × 26 layers + 1 output = **183 layers**

### Formula

```
quantized_layers = (attention_projs + ffn_projs) × n_layers + output_layers

Where:
  attention_projs = 4 (Q, K, V, out)
  ffn_projs       = 2 (standard) or 3 (SwiGLU)
  output_layers   = 1 (lm_head)
```

---

## References

- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) - Shazeer, 2020
- [LLaMA: Open Foundation Models](https://arxiv.org/abs/2302.13971) - Touvron et al., 2023
