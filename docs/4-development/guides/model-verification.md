# Model Verification Guide

How to verify that a GGUF model is loaded correctly and produces the right output. Three complementary approaches, ordered by coverage.

## 1. Token ID Verification

**What it catches:** Tokenizer bugs (wrong encoding, missing merges, normalization errors).

**Why it matters:** Wrong token IDs produce coherent but semantically wrong output. The model doesn't crash — it silently gives you nonsense because it looks up entirely different embedding vectors.

**Approach:** Compare `GgufTokenizer` output against a known-good reference (HF tokenizer or hardcoded expected IDs).

**Tools:**
- `tokenizer_verifier` CLI: `cargo run --example tokenizer_verifier -- model.gguf tokenizer.json`
- `tokenizer_verifier_test` integration test (env-gated with `GGUF_MODEL_PATH` + `HF_TOKENIZER_PATH`)

**Example:**
```
Input:  "The capital of France is"
Expect: [818, 5279, 529, 7001, 563]
Wrong:  [669, 270, 5279, 270, 529, 270, 7001, 270, 563]  ← broken tokenizer
```

**When to run:** After any change to `GgufTokenizer`, `from_gguf()`, or the encode/decode pipeline.

## 2. Weight Spot-Checks

**What it catches:** Weight loading bugs (wrong tensor mapping, broken dequantization, incorrect dtype handling, transposition errors).

**Why it matters:** A wrong weight row in the embedding table or attention matrix will shift model output. Quantization bugs (e.g., Q4_1 dequantized incorrectly) produce subtly wrong values across millions of parameters.

**Approach:** Load the same GGUF model in a reference implementation (Python + llama-cpp-python, or HF transformers), extract specific weight values, and compare against our loaded weights.

### Generating Reference Values (Python)

```python
from llama_cpp import Llama

model = Llama("gemma-3-1b-it-Q4_0.gguf", n_ctx=32, n_gpu_layers=0, verbose=False)

# Extract specific tensor values
# (API depends on the binding — may need ctypes access to raw weights)
```

Or with HuggingFace (for non-quantized models):

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype=torch.float32)
emb = model.model.embed_tokens.weight

# First 5 values of embedding row 818 ("The")
print(emb[818, :5].tolist())
# → [0.0234, -0.0156, 0.0078, ...]
```

### Rust Test

```rust
#[test]
fn verify_embedding_weights() {
    let gguf = GGUFFile::parse_header("model.gguf").unwrap();
    let tensors = gguf.load_and_remap_gemma3("model.gguf", 26).unwrap();
    let tensors = convert_tensors(tensors);

    let emb = &tensors["token_embedding"];
    assert_eq!(emb.shape(), &[262144, 1152]);

    // Spot-check: first 5 values of row 818 (token "The")
    let row_818: Vec<f32> = emb.slice(0, 818, 819).unwrap().iter().take(5).collect();
    let expected = [0.0234, -0.0156, 0.0078, ...]; // from Python
    for (a, b) in row_818.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-3, "weight mismatch: {} vs {}", a, b);
    }
}
```

### What to Spot-Check

| Tensor | Why |
|---|---|
| `token_embedding` row for a common token | Catches embedding load errors |
| `blk.0.attn_q.weight` corner values | Catches attention weight mapping |
| `blk.0.ffn_down.weight` (Q4_1 in Gemma 3) | Catches quantization dequant bugs |
| `output_norm.weight` | Catches norm weight loading |
| Last layer attention weights | Catches off-by-one in layer indexing |

**When to run:** After changes to weight loading (`from_pretrained_gemma3`, `get_weight`, `WeightMap`), GGUF parsing, or dequantization.

## 3. Logit Comparison (End-to-End)

**What it catches:** Everything — tokenizer bugs, weight loading bugs, dequantization errors, attention implementation bugs, normalization bugs, RoPE bugs, KV cache bugs. If the logits match, the entire pipeline is correct.

**Why it matters:** This is the single strongest verification. A wrong value anywhere in the pipeline (tokenizer, embeddings, attention, FFN, norms, RoPE) will produce measurably different logits.

**Approach:** Feed a known prompt through both our model and a reference implementation, compare the output logit vector (or top-K predictions).

### Generating Reference Logits (Python)

```python
from llama_cpp import Llama
import numpy as np

model = Llama("gemma-3-1b-it-Q4_0.gguf", n_ctx=128, n_gpu_layers=0, logits_all=True)

prompt = "The capital of France is"
tokens = model.tokenize(prompt.encode(), add_bos=True)
model.eval(tokens)

# Get logits for the last token position
logits = np.array(model.scores[len(tokens) - 1])

# Top-5 predictions
top5 = np.argsort(logits)[-5:][::-1]
for idx in top5:
    print(f"  id={idx:<6} logit={logits[idx]:>8.3f}")

# Save full logit vector for comparison
np.save("reference_logits.npy", logits)
```

### Rust Test

```rust
#[test]
fn verify_logits_match_reference() {
    // Build model
    let gguf = GGUFFile::parse_header("model.gguf").unwrap();
    let config = ...;
    let model = LlmModel::from_pretrained_gemma3(&config, tensors).unwrap();

    // Same prompt, same token IDs
    let input_ids = vec![2, 818, 5279, 529, 7001, 563]; // BOS + "The capital of France is"
    let input = Tensor::from_vec(
        input_ids.iter().map(|&id| id as f32).collect(),
        vec![1, input_ids.len()],
    ).unwrap();

    let logits = model.forward(&input).unwrap();
    let last_logits: Vec<f32> = logits
        .slice(1, input_ids.len() - 1, input_ids.len()).unwrap()
        .iter().collect();

    // Compare top-5 predictions (order should match exactly)
    let mut indexed: Vec<(usize, f32)> = last_logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let expected_top5 = [235269, 148, 6481, 235248, 51586]; // from Python
    for (i, &expected_id) in expected_top5.iter().enumerate() {
        assert_eq!(indexed[i].0, expected_id,
            "Top-{} mismatch: got id={}, expected id={}", i+1, indexed[i].0, expected_id);
    }

    // Optional: compare full logit vector within tolerance
    // let reference: Vec<f32> = load_npy("reference_logits.npy");
    // let max_diff = last_logits.iter().zip(reference.iter())
    //     .map(|(a, b)| (a - b).abs())
    //     .fold(0.0f32, f32::max);
    // assert!(max_diff < 0.1, "max logit diff: {}", max_diff);
}
```

### Tolerance

Quantized models won't match float32 reference exactly. Expected tolerances:

| Comparison | Tolerance |
|---|---|
| Top-1 prediction ID | Exact match |
| Top-5 prediction IDs | Exact match (order may vary for close logits) |
| Logit values (Q4_0 vs Q4_0) | < 0.01 (same quant, different impl) |
| Logit values (Q4_0 vs F32) | < 1.0 (quantization noise) |
| Logit values (Q4_0 vs F16) | < 0.5 |

### Practical Shortcut: Top-K Hardcoded Test

If setting up Python reference is too heavy, hardcode the expected top-5 token IDs for a few known prompts. These are stable across runs for greedy decoding:

```rust
const VERIFICATION_CASES: &[(&str, &[u32], &[u32])] = &[
    // (prompt, input_token_ids, expected_top5_output_ids)
    ("The capital of France is", &[2, 818, 5279, 529, 7001, 563], &[...]),
    ("1 + 1 =", &[2, 235274, 963, 235248, 235274, 589], &[...]),
];
```

**When to run:** After any change to the forward pass, attention, FFN, normalization, RoPE, KV cache, weight loading, or quantized matmul.

## 4. SafeTensors / GPT-2 Verification

The same three approaches apply to SafeTensors models loaded through `LlmModel::from_pretrained_gpt2()`. Key differences from GGUF:

- **Tokenizer**: BPE tokenizer loaded from `vocab.json` + `merges.txt` (not embedded in the model file)
- **Weight mapping**: HuggingFace names are remapped via `map_gpt2_weights()` to LlmModel internal names (e.g., `transformer.h.0.attn.c_attn.weight` becomes `layers.0.attention.c_attn.weight`)
- **Conv1D transpose**: GPT-2 stores linear weights as `[in, out]`; `from_pretrained_gpt2()` transposes them to `[out, in]`
- **Fused QKV**: The single `c_attn.weight` is split into separate Q, K, V projections
- **No quantization**: SafeTensors weights are F32, so logit comparison tolerances should be tighter (< 0.001)

### Generating Reference Logits (Python, GPT-2)

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

inputs = tokenizer("The capital of France is", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits[0, -1]  # last position

top5 = torch.topk(logits, 5)
for idx, val in zip(top5.indices, top5.values):
    print(f"  id={idx:<6} logit={val:>8.3f}")
```

> **Note:** Both `rustml-infer --safetensors` and `sweai infer --safetensors` now use `LlmModel` with KV cache (not `GptModel`). The `GptModel` implementation is retained as a teaching reference only. See [ADR-001](../../3-design/adr/adr-001-unified-llmmodel-for-gpt2.md).

## Summary

| Approach | Coverage | Dependencies | Speed |
|---|---|---|---|
| Token ID verification | Tokenizer only | GGUF file + reference IDs | Fast (ms) |
| Weight spot-checks | Weight loading + dequant | GGUF file + reference values | Fast (seconds) |
| Logit comparison | Full pipeline | GGUF file + reference logits | Slow (model load + forward pass) |

Use all three. Token IDs catch tokenizer regressions instantly. Weight spot-checks catch loading bugs without running inference. Logit comparison is the final proof that everything works end-to-end.
