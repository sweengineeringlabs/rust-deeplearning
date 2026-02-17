# Gemma 3 Tokenizer Bug: Root Cause Analysis & Fix

## Summary

Gemma 3 1B-IT GGUF inference produced grammatically coherent but semantically wrong output. The root cause was the `GgufTokenizer` using greedy longest-match instead of proper SentencePiece BPE encoding, causing the model to receive entirely wrong token IDs.

**Before fix:** "The capital of France is a popular and well-established online platform for the sale of used books..."
**After fix:** "The capital of France is, indeed, Paris."

## Symptom Analysis

| Observation | Implication |
|---|---|
| Output is coherent English | Model weights load correctly, forward pass is numerically sound |
| Output is semantically unrelated to prompt | Model receives wrong input tokens |
| Top-5 predictions are generic words ("a", "the", "been") | Context representation is meaningless to the model |
| Chat mode produces safety refusal for benign question | IT model misinterprets the token sequence |

The key diagnostic insight: **coherent but wrong** output rules out numerical bugs (which produce garbage) and points to an input encoding problem.

## Root Cause: Broken SentencePiece BPE

### The Old Tokenizer

The `GgufTokenizer` implemented a naive greedy longest-match algorithm:

```
for each position in text:
    try matching from longest to shortest substring
    if match found in vocabulary: emit token
    else: emit byte fallback token
```

**Problems:**

1. **No BPE merge rules** - The algorithm ignored `tokenizer.ggml.scores` entirely. SentencePiece BPE requires merge scores to determine which adjacent token pairs should be combined and in what priority order.

2. **Broken space normalization** - The condition `pos == 0 || bytes[pos - 1] == b' '` only checked if the previous byte was a space, not whether the current position is a word boundary in the SentencePiece sense.

3. **Wrong tokenization strategy** - Greedy longest-match produces fundamentally different token sequences than BPE. BPE starts with characters and merges upward; greedy starts long and works down.

### Token ID Comparison

For the prompt "The capital of France is":

| Tokenizer | Token IDs | Count |
|---|---|---|
| **Broken** (greedy) | `[2, 669, 270, 5279, 270, 529, 270, 7001, 270, 563]` | 10 |
| **Fixed** (BPE) | `[2, 818, 5279, 529, 7001, 563]` | 6 |

The broken tokenizer split every space into a separate `<0x20>` byte token (ID 270), producing tokens the model never saw during training:

```
Broken: BOS "The" <0x20> "capital" <0x20> "of" <0x20> "France" <0x20> "is"
Fixed:  BOS "The" "capital" "of" "France" "is"
```

Token 669 ("The" without SentencePiece marker) vs token 818 ("The" - the correct vocabulary entry). These are entirely different embeddings in the model's weight matrix.

## The Fix

### 1. Load BPE Merge Scores from GGUF

The GGUF format stores three tokenizer arrays:
- `tokenizer.ggml.tokens` - vocabulary strings (262,144 entries for Gemma 3)
- `tokenizer.ggml.scores` - float scores per token (merge priority)
- `tokenizer.ggml.token_type` - type flags (normal, byte, control, etc.)

The scores represent log-probabilities. Higher score = should be merged earlier in the BPE algorithm.

### 2. Proper SentencePiece BPE Encoding

The corrected algorithm:

```
1. Normalize text: replace " " with "▁" (U+2581)
   - If add_space_prefix=true: prepend "▁"
   - Gemma 3 GGUF sets add_space_prefix=false

2. Split into initial character tokens:
   - Each character → vocabulary lookup
   - Unknown characters → UTF-8 byte fallback (<0xHH> tokens)

3. BPE merge loop:
   while tokens.len() >= 2:
       for each adjacent pair (tokens[i], tokens[i+1]):
           merged_string = token_str[i] + token_str[i+1]
           if merged_string in vocabulary:
               candidate_score = scores[vocab[merged_string]]
       merge the pair with highest score
       if no merge possible: break
```

### 3. Respect GGUF Tokenizer Settings

- `tokenizer.ggml.add_space_prefix = false` - Gemma 3 does NOT add a leading "▁" to input text
- This is critical: adding an unwanted "▁" would shift the first token to a different vocabulary entry

### 4. Proper Byte Token Decoding

The decode method now converts `<0xHH>` byte tokens back to actual bytes before UTF-8 assembly, instead of emitting them as literal strings.

## Additional Fix: Q4_1 Weight Preservation

While investigating, we found that `ffn_down` weights in the Gemma 3 GGUF are stored as **Q4_1** (not Q4_0). The `get_weight` function only preserved Q4_0/Q8_0/F32, causing Q4_1 weights to be unnecessarily dequantized to F32.

Adding `DType::Q4_1` to the preserve list reduced model build time from **21.6s to 9.4s** (the 26-layer x 1 `ffn_down` weight dequantization was the bottleneck).

## Performance After Fix

| Metric | Before | After |
|---|---|---|
| Model build time | 21.6s | 9.4s |
| Prefill throughput | 3.8 tok/s | 5.5-9.8 tok/s |
| Decode throughput | 1.7-2.3 tok/s | 2.4-2.7 tok/s |
| Output quality | Wrong | Correct |

## Files Changed

| File | Change |
|---|---|
| `rustml/nlp/main/src/core/tokenizer/mod.rs` | Rewrote `GgufTokenizer` with BPE scores, merge loop, space normalization |
| `rustml/nlp/main/src/core/model.rs` | Added `DType::Q4_1` to weight preservation in `get_weight` (3 call sites) |

## Lessons Learned

1. **Tokenizer correctness is paramount** - A broken tokenizer produces coherent but wrong output, which is harder to diagnose than numerical errors (which produce garbage).

2. **SentencePiece is not just a vocabulary** - The vocabulary alone is insufficient; the merge scores define the encoding algorithm. Loading tokens without scores is fundamentally broken.

3. **Test with reference tokenization** - Comparing token IDs against a reference implementation (llama.cpp, HuggingFace) immediately reveals tokenizer bugs. The 10-token vs 6-token discrepancy was the smoking gun.

4. **Check GGUF metadata flags** - Settings like `add_space_prefix` materially affect tokenization. The GGUF format stores these for a reason.
