# GGUF USER_DEFINED Token Bug: Pre-BPE Greedy Matching

## Summary

The `GgufTokenizer` failed to match USER_DEFINED tokens (multi-space, newline runs, HTML tags, tabs) because GGUF stores them with **literal spaces** while SentencePiece normalization converts spaces to `▁` before BPE. The BPE merge `▁▁` could never be found in the vocabulary (which only had `"  "`), so consecutive spaces were never merged.

**Before fix:** `"word   with"` → `word ▁ ▁ ▁with` (10 tokens, wrong)
**After fix:** `"word   with"` → `word ▁▁▁ with` (7 tokens, matches HF reference)

## Root Cause

### The GGUF Vocabulary Layout

GGUF stores three tokenizer arrays plus a type array:
- `tokenizer.ggml.tokens` — vocabulary strings
- `tokenizer.ggml.scores` — float scores per token (merge priority)
- `tokenizer.ggml.token_type` — type flags per token:
  - 1 = NORMAL (standard BPE tokens)
  - 3 = CONTROL (`<bos>`, `<eos>`, `<pad>`, etc.)
  - 4 = USER_DEFINED (added tokens: multi-space, newlines, HTML tags)
  - 6 = BYTE (`<0xHH>` fallback tokens)

### USER_DEFINED Tokens in Gemma 3

Gemma 3 has 163 USER_DEFINED tokens, all with score `-1000.0`:

| ID Range | Content | Count |
|---|---|---|
| 5 | `[multimodal]` | 1 |
| 107-137 | `\n` through `\n×31` | 31 |
| 138-167 | `"  "` through `" ×30"` (literal spaces) | 30 |
| 168-237 | HTML tags (`<table>`, `<h1>`, `</div>`, etc.) | 70 |
| 255968-255998 | `\t` through `\t×31` | 31 |

### The Score Mismatch

The sentinel score `-1000.0` means these tokens cannot participate in score-based BPE merging:

| Token | GGUF String | Score | Type | HF Merge Rank |
|---|---|---|---|---|
| id=138 | `"  "` (2 spaces) | -1000.0 | USER_DEFINED | 1393 |
| id=515 | `"▁w"` | -21.0 | NORMAL | 1418 |

In score-based BPE: `-21.0 > -1000.0`, so `▁w` always merges before `▁▁`.
In HF's merge list: position 1393 < 1418, so `▁ ▁` merges before `▁ w`.

### The Normalization Mismatch

Even if the score weren't an issue, the merge lookup itself fails:

1. Input: `"word   with"`
2. After normalization (space → `▁`): `"word▁▁▁with"`
3. BPE tries to merge `▁` + `▁` → looks up `"▁▁"` in vocab
4. Vocab has `"  "` (0x20 0x20), **NOT** `"▁▁"` (0xE2 0x96 0x81 × 2)
5. Lookup fails → merge never happens

This is the fundamental bug: GGUF stores space-containing USER_DEFINED tokens with literal ASCII spaces, but the tokenizer normalizes spaces to `▁` before BPE, making these tokens unreachable through merging.

## The Fix

### SentencePiece's Approach

In SentencePiece (and llama.cpp), USER_DEFINED tokens are matched **greedily on the original text** before normalization and BPE:

1. Scan raw text left-to-right
2. At each position, try matching the longest USER_DEFINED token
3. Split text into: matched USER_DEFINED tokens + remaining text segments
4. For each remaining segment: normalize (space → `▁`) then run BPE
5. Combine all token IDs

### Implementation

Added to `GgufTokenizer`:

1. **Load `tokenizer.ggml.token_type`** from GGUF metadata
2. **Collect USER_DEFINED tokens** (type=4, length ≥ 2), sorted by length descending
3. **`split_user_defined(text)`** — greedy longest-match on original text, returns segments
4. **Modified `encode()`** — match USER_DEFINED first, then normalize+BPE remaining segments

```
encode("word   with   extra   spaces"):
  1. Match USER_DEFINED: "   " at pos 4, 11, 18
  2. Segments: Text("word"), Token(139), Text("with"), Token(139), Text("extra"), Token(139), Text("spaces")
  3. Normalize+BPE each text segment: "word"→[3017], "with"→[4060], "extra"→[27909], "spaces"→[35220]
  4. Result: [3017, 139, 4060, 139, 27909, 139, 35220]  ← matches HF exactly
```

## Verification

### TokenizerVerifier Results (16/16 PASS)

| Test String | Tokens | Result |
|---|---|---|
| `"The capital of France is"` | 5 | PASS |
| `"Hello, world!"` | 4 | PASS |
| `"café résumé naïve"` | 4 | PASS |
| `"word   with   extra   spaces"` | 7 | PASS (was FAIL) |
| `"fn main() { println!(\"hello\"); }"` | 9 | PASS |
| `"line1\nline2\ttab"` | 7 | PASS |
| ... and 10 more | | PASS |

### Inference Output (unchanged)

```
Prompt: "The capital of France is"
Token IDs: [2, 818, 5279, 529, 7001, 563]
Output: "The capital of France is, indeed, Paris."
```

## Files Changed

| File | Change |
|---|---|
| `rustml/nlp/main/src/core/tokenizer/mod.rs` | Load token_type, collect USER_DEFINED tokens, greedy pre-match before BPE |

## Relationship to Previous Tokenizer Fix

This is the second tokenizer fix for Gemma 3. The first fix (see `gemma3-tokenizer-fix.md`) replaced the broken greedy longest-match algorithm with proper SentencePiece BPE. This fix addresses a subtlety within the BPE implementation: USER_DEFINED tokens with literal spaces in the GGUF vocab are unreachable through score-based merging after space→`▁` normalization.

## Lessons Learned

1. **GGUF token types matter** — The `token_type` array isn't just metadata; it determines how tokens participate in encoding. USER_DEFINED tokens (type=4) must be matched greedily before BPE, not through the merge loop.

2. **Score -1000.0 is a sentinel** — All USER_DEFINED tokens have score `-1000.0`, making them effectively unmergeable through score-based BPE. This is by design: they're meant to be matched as whole units.

3. **Normalization creates a namespace split** — GGUF stores tokens in "raw" form (literal spaces), but encoding operates in "normalized" form (`▁`). Any token containing characters that are transformed during normalization becomes unreachable through BPE and must be handled separately.

4. **Build a verifier** — The `TokenizerVerifier` tool comparing GGUF against HF (reference) on 16 test strings immediately revealed this bug. Without it, multi-space inputs would silently produce wrong token IDs.
