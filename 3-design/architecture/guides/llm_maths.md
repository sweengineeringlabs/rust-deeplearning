# LLM Inference Memory Math

## KV Cache Sizing

The KV cache stores key and value projections for every layer, enabling O(1)
per-token decode instead of re-computing attention over the full sequence.

### Formula

```
KV cache bytes = 2 * n_layers * n_kv_heads * seq_len * head_dim * sizeof(f32)
                 ^                                                 ^
                 K + V tensors                                     4 bytes
```

### Worked Example: google/gemma-3-1b-it

| Parameter     | Value |
|---------------|-------|
| `n_layers`    | 26    |
| `n_kv_heads`  | 1     |
| `head_dim`    | 256   |
| `max_seq_len` | 32768 |

**Full context window (no auto-sizing):**

```
2 x 26 x 1 x 32768 x 256 x 4 = 1,744,830,464 bytes = 1664 MB
```

Per-layer breakdown:
- One K or V tensor: `1 x 32768 x 256 x 4 = 32 MB`
- K + V per layer: `64 MB`
- All 26 layers: `26 x 64 = 1664 MB`

**Auto-sized for `--max-tokens 20` with a short prompt (~6 tokens):**

Effective context = `prompt_tokens + max_tokens + margin = 6 + 20 + 64 = 90`

```
2 x 26 x 1 x 90 x 256 x 4 = 4,792,320 bytes ~ 4.6 MB
```

Reduction: **1664 MB -> 4.6 MB** (362x smaller).

## Why Auto-Sizing Matters

Models declare `max_position_embeddings` for their full context window (e.g.
32768 for Gemma-3, 131072 for Llama-3). Pre-allocating the full window wastes
memory and slows startup when generating only a handful of tokens.

The `--context-len` flag (or automatic `prompt + max_tokens + margin` sizing)
allocates only what is needed, capped at the model maximum.

## Sampling Hot Path Allocations

For a 262K-token vocabulary (Gemma-3), each `f32` vector over the vocab is ~1 MB.

| Function              | Before (allocs/call) | After          | Savings        |
|-----------------------|----------------------|----------------|----------------|
| `sample_categorical`  | 2 Vecs (~2 MB)       | 0 (streaming)  | ~2 MB / token  |
| `apply_top_p`         | 4 Vecs (~4 MB)       | 1 Vec (~2 MB)  | ~2 MB / token  |
| `apply_top_k` (k<=1K) | 1 Vec (~1 MB)        | 1 Vec (~8 KB)  | ~1 MB / token  |
| `decode_step`         | 2 Vecs (~2 MB)       | 1 Vec (~1 MB)  | ~1 MB / token  |
