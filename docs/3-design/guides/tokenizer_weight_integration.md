# Tokenizer–Weight Integration Architecture

> **TLDR:** How tokenization and model weights connect through the embedding matrix — the critical bridge in the inference pipeline.

**Audience**: Developers, ML Engineers

**WHAT**: Documents the data-flow relationship between tokenizer vocabulary and model weights
**WHY**: Misalignment between tokenizer and weights causes silent errors (wrong tokens, garbage output, panics)
**HOW**: Pipeline, dataflow, and sequence diagrams showing each stage of inference

---

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Component Responsibilities](#component-responsibilities)
- [The Embedding Bridge](#the-embedding-bridge)
- [Dataflow Diagram](#dataflow-diagram)
- [Sequence Diagram](#sequence-diagram)
- [Vocabulary Agreement Contract](#vocabulary-agreement-contract)
- [CLI Verification Matrix](#cli-verification-matrix)
- [Failure Modes](#failure-modes)

---

## Pipeline Overview

The end-to-end inference pipeline has two domains: **tokenization** (text ↔ IDs) and **computation** (IDs ↔ vectors ↔ output). The embedding matrix bridges them.

```
                     TOKENIZATION                          COMPUTATION
               ┌──────────────────────┐    ┌─────────────────────────────────────────┐
               │                      │    │                                         │
 "Hello" ──▶  │  Tokenizer (encode)  │ ──▶│  Embedding ──▶ Transformer ──▶ LM Head  │
               │  text → token IDs    │    │  IDs → vecs    vecs → vecs    vecs → logits
               │                      │    │                                         │
               └──────────────────────┘    └──────────────────────┬──────────────────┘
                                                                  │
               ┌──────────────────────┐                           │
               │                      │                           │
 "world" ◀──  │  Tokenizer (decode)  │ ◀── argmax/sample ◀──────┘
               │  token IDs → text    │         logits → ID
               │                      │
               └──────────────────────┘
```

Each stage in detail:

| Stage | Input | Output | Owned By |
|-------|-------|--------|----------|
| **Encode** | Raw text (`"Hello world"`) | Token IDs (`[17534, 2134]`) | Tokenizer |
| **Embed** | Token IDs | Dense vectors (dim=`embedding_length`) | `token_embd.weight` |
| **Transform** | Vectors | Contextualized vectors | Attention + FFN weights |
| **Project** | Final hidden state | Logits (vocab-sized) | `output.weight` (or tied) |
| **Sample** | Logits | Next token ID | Sampling logic |
| **Decode** | Token ID | Text piece | Tokenizer |

---

## Component Responsibilities

### Tokenizer

Manages the mapping between text and integer token IDs.

```
Tokenizer
├── Vocabulary:   token_id ↔ token_string   (e.g., 17534 ↔ "Hello")
├── Merge rules:  BPE merge priority table   (SentencePiece or GPT-2 style)
├── Special tokens: BOS, EOS, PAD, UNK       (control tokens with reserved IDs)
└── Normalization: Unicode NFKC, whitespace   (preprocessing before tokenization)
```

**Does NOT** know about vectors, weights, or model architecture.

### Model Weights

Transform token IDs into probability distributions over the vocabulary.

```
Model Weights
├── token_embd.weight     [vocab_size × dim]       ← THE BRIDGE
├── blk.N.attn_q.weight   [dim × dim]              ← attention
├── blk.N.attn_k.weight   [dim × kv_dim]
├── blk.N.attn_v.weight   [dim × kv_dim]
├── blk.N.attn_output     [dim × dim]
├── blk.N.ffn_gate        [dim × ffn_dim]           ← feed-forward
├── blk.N.ffn_up          [dim × ffn_dim]
├── blk.N.ffn_down        [ffn_dim × dim]
├── blk.N.attn_norm       [dim]                     ← layer norms
├── blk.N.ffn_norm        [dim]
└── output.weight          [vocab_size × dim]        ← logit projection
```

**Does NOT** know about text, merge rules, or encoding logic.

---

## The Embedding Bridge

The embedding matrix `token_embd.weight` is the single point where tokenization meets computation:

```
                    token_embd.weight
                   ┌─────────────────┐
         row 0  →  │  0.12  -0.34 ...│  ← vector for token 0 ("<unk>")
         row 1  →  │  0.56   0.78 ...│  ← vector for token 1 ("<s>")
         row 2  →  │ -0.91   0.23 ...│  ← vector for token 2 ("</s>")
            ⋮       │       ⋮         │
  row 17534  →  │  0.44  -0.67 ...│  ← vector for token 17534 ("Hello")
            ⋮       │       ⋮         │
  row V-1    →  │  0.33   0.11 ...│  ← vector for token V-1
                   └─────────────────┘
                    shape: [V × D]
                    V = vocab_size
                    D = embedding_length
```

**The contract**: Row `i` of the embedding matrix corresponds to token `i` in the tokenizer vocabulary. If the tokenizer says `"Hello"` = ID `17534`, then `token_embd.weight[17534]` must be the learned vector for `"Hello"`.

---

## Dataflow Diagram

Complete data transformation through the inference pipeline:

```
┌─────────────┐
│  Raw Text   │  "The capital of France is"
└──────┬──────┘
       │ tokenizer.encode()
       ▼
┌─────────────┐
│  Token IDs  │  [651, 6421, 302, 6556, 338]
└──────┬──────┘
       │ embedding lookup: token_embd.weight[id]
       ▼
┌─────────────────────┐
│  Embedding Vectors  │  5 vectors × dim (e.g., 2048)
│  shape: [5 × 2048]  │
└──────┬──────────────┘
       │
       ▼  ×N transformer layers
┌─────────────────────────────────────────────────┐
│  For each layer:                                │
│                                                 │
│  ┌───────────┐    ┌───────────┐    ┌─────────┐ │
│  │ RMS Norm  │ ─▶ │ Attention │ ─▶ │ + Resid │ │
│  └───────────┘    │  Q, K, V  │    └────┬────┘ │
│                   │  Rotary   │         │      │
│                   │  Softmax  │         ▼      │
│                   └───────────┘    ┌─────────┐ │
│                                    │ RMS Norm│ │
│                                    └────┬────┘ │
│                                         ▼      │
│                                    ┌─────────┐ │
│                                    │   FFN   │ │
│                                    │ SwiGLU  │ │
│                                    └────┬────┘ │
│                                         ▼      │
│                                    ┌─────────┐ │
│                                    │ + Resid │ │
│                                    └─────────┘ │
└──────────────────────────┬──────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Final Hidden State │  shape: [5 × 2048]
└──────┬──────────────┘
       │ last position only → output.weight matmul
       ▼
┌─────────────────────┐
│  Logits             │  shape: [vocab_size] (e.g., 262144)
│  raw scores for     │
│  every token in     │
│  the vocabulary     │
└──────┬──────────────┘
       │ temperature → top-k → top-p → sample
       ▼
┌─────────────┐
│  Next Token │  ID: 3681
└──────┬──────┘
       │ tokenizer.decode()
       ▼
┌─────────────┐
│  Text Piece │  "Paris"
└─────────────┘
```

---

## Sequence Diagram

Token-by-token autoregressive generation:

```
 User          Tokenizer         Embedding        Transformer        Sampler
  │                │                 │                 │                │
  │  "Hello"       │                 │                 │                │
  │───────────────▶│                 │                 │                │
  │                │  encode         │                 │                │
  │                │  [17534]        │                 │                │
  │                │────────────────▶│                 │                │
  │                │                 │  lookup row     │                │
  │                │                 │  17534          │                │
  │                │                 │────────────────▶│                │
  │                │                 │                 │  forward pass  │
  │                │                 │                 │  N layers      │
  │                │                 │                 │───────────────▶│
  │                │                 │                 │                │ sample
  │                │                 │                 │                │ logits
  │                │                 │                 │   token 2134 ◀─┤
  │                │                 │                 │                │
  │                │◀────────────────────────────────────────── 2134 ──┤
  │                │  decode(2134)   │                 │                │
  │◀─" world"─────│                 │                 │                │
  │                │                 │                 │                │
  │ ─ ─ ─ ─ ─ ─ NEXT ITERATION: input = [17534, 2134] ─ ─ ─ ─ ─ ─ ─ │
  │                │                 │                 │                │
  │                │────────────────▶│                 │                │
  │                │                 │  lookup rows    │                │
  │                │                 │  17534, 2134    │                │
  │                │                 │────────────────▶│                │
  │                │                 │                 │  forward pass  │
  │                │                 │                 │───────────────▶│
  │                │                 │                 │   token 528  ◀─┤
  │                │◀──────────────────────────────────────── 528 ─────┤
  │◀─"!" ─────────│                 │                 │                │
  │                │                 │                 │                │
  │ (repeat until EOS or max_tokens)                                   │
```

---

## Vocabulary Agreement Contract

The tokenizer and model weights must agree on a shared vocabulary. This is not enforced by file format — it is a **semantic contract**.

### What Must Match

| Property | Tokenizer Side | Weight Side | Failure If Mismatched |
|----------|---------------|-------------|----------------------|
| **Vocab size** | Number of tokens in vocabulary | Rows in `token_embd.weight` | Index-out-of-bounds or dimension mismatch |
| **Token-to-ID mapping** | `"Hello"` → `17534` | Row `17534` = learned vector for `"Hello"` | Correct IDs retrieve wrong vectors; garbage output |
| **Special token IDs** | BOS=`2`, EOS=`1` | Training used same IDs for start/end signals | Model doesn't recognize sequence boundaries |

### Where Agreement Lives in GGUF

GGUF bundles both tokenizer and weights in one file, which helps maintain agreement:

```
┌─────────────────────────────────────┐
│              GGUF File              │
│                                     │
│  ┌───────────────────────────────┐  │
│  │         Metadata              │  │
│  │  tokenizer.ggml.tokens  ─────│──│── Vocabulary (token strings)
│  │  tokenizer.ggml.scores  ─────│──│── Merge priorities
│  │  tokenizer.ggml.bos_token_id │  │── Special token IDs
│  │  tokenizer.ggml.eos_token_id │  │
│  │  {arch}.embedding_length     │  │── Model dimensions
│  │  {arch}.block_count          │  │
│  └───────────────────────────────┘  │
│                                     │
│  ┌───────────────────────────────┐  │
│  │         Tensors               │  │
│  │  token_embd.weight ──────────│──│── Embedding matrix [V × D]
│  │  blk.0.attn_q.weight        │  │── Transformer weights
│  │  ...                          │  │
│  │  output.weight ──────────────│──│── Output projection [V × D]
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘

Agreement: len(tokenizer.ggml.tokens) == token_embd.weight.shape[0]
```

---

## CLI Verification Matrix

Each RustML CLI tool verifies a different part of the pipeline:

```
                  rustml-tokenizer        rustml-gguf-inspect         rustml-infer
                  ─────────────────       ───────────────────         ────────────
 Text ↔ IDs           ✓                                                   ✓
 Vocabulary            ✓ (info)                ✓ (meta)                    ✓
 Merge rules           ✓ (encode)                                         ✓
 Special tokens        ✓ (info --lookup)       ✓ (meta --key)
 Embedding shape                               ✓ (tensors)                ✓
 Weight dtypes                                 ✓ (tensors)
 Model config                                  ✓ (info)                   ✓
 Forward pass                                                             ✓
 Sampling                                                                 ✓
 End-to-end                                                               ✓
```

### Verification Workflow

```
Step 1: Inspect the model file
    $ rustml-gguf-inspect info model.gguf
    → Confirm architecture, dimensions, vocab size, tensor count

Step 2: Verify tokenizer vocabulary
    $ rustml-gguf-inspect meta model.gguf --key tokenizer.ggml.tokens
    → Confirm vocabulary is present and sized correctly

Step 3: Verify embedding dimensions match vocab
    $ rustml-gguf-inspect tensors model.gguf --filter token_embd
    → Confirm token_embd.weight shape is [vocab_size × embedding_length]

Step 4: Test tokenizer encode/decode
    $ rustml-tokenizer --gguf model.gguf encode "Hello world"
    → Confirm IDs are within [0, vocab_size)
    $ rustml-tokenizer --gguf model.gguf decode <IDs>
    → Confirm round-trip recovers original text

Step 5: End-to-end inference
    $ rustml-infer model.gguf --prompt "Hello" --max-tokens 10
    → Confirm coherent text generation
```

---

## Failure Modes

What happens when the tokenizer–weight contract is broken:

| Failure | Symptom | Root Cause | Detection |
|---------|---------|------------|-----------|
| Vocab size mismatch | Panic: index out of bounds | Tokenizer has more tokens than embedding rows | `gguf-inspect info` vs `gguf-inspect tensors --filter token_embd` |
| Wrong tokenizer model | Coherent-looking but semantically wrong text | Token IDs map to wrong embedding vectors | Compare `tokenizer encode` output against reference |
| Corrupted merges | Tokens split incorrectly | BPE merge table damaged or incomplete | `tokenizer encode` produces unexpected IDs |
| Missing special tokens | Model ignores turn boundaries | BOS/EOS IDs not matching training config | `tokenizer info --lookup "<bos>"` + `gguf-inspect meta --key tokenizer.ggml.bos_token_id` |
| Embedding weight corruption | Random/noisy output | Quantization error or file corruption | Compare output against reference implementation |

---

## See Also

- [Architecture](../architecture.md) — project structure and crate layout
- [Manual Testing Guide](../../5-testing/manual_testing.md) — CLI test procedures
- [Manual Tokenizer Tests](../../5-testing/manual_tokenizer_tests.md) — tokenizer verification tests
- [Manual GGUF Inspector Tests](../../5-testing/manual_gguf_inspect_tests.md) — weight/metadata inspection tests
