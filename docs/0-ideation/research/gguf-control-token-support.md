# GGUF CONTROL Token Bug: Special Tokens Split Into Characters

## Summary

The `GgufTokenizer` only matched USER_DEFINED tokens (type 4) greedily before BPE. CONTROL tokens (type 2) like `<start_of_turn>`, `<end_of_turn>`, and `<bos>` were not recognized and got split into individual characters, producing wrong token IDs.

**Before fix:** `"<bos><start_of_turn>user\nHi<end_of_turn>"` → 35 tokens (each `<` `b` `o` `s` `>` as separate characters)
**After fix:** `"<bos><start_of_turn>user\nHi<end_of_turn>"` → 16 tokens (special tokens matched as single IDs)

## Root Cause

### GGUF Token Type 2: CONTROL

The GGUF `tokenizer.ggml.token_type` array classifies tokens into types:

| Type | Name | Examples |
|---|---|---|
| 1 | NORMAL | Standard BPE tokens |
| 2 | CONTROL | `<bos>` (id=2), `<eos>` (id=1), `<start_of_turn>` (id=106), `<end_of_turn>` (id=107) |
| 4 | USER_DEFINED | Multi-space, newline runs, HTML tags |
| 6 | BYTE | `<0xHH>` fallback tokens |

The previous fix (see `gguf-user-defined-tokens.md`) added greedy pre-BPE matching for USER_DEFINED tokens. However, the type check only matched type 4:

```rust
if ttype == TOKEN_TYPE_USER_DEFINED && i < id_to_token.len() {
```

CONTROL tokens (type 2) share the same fundamental property as USER_DEFINED tokens: they are multi-character strings (e.g. `<start_of_turn>` is 16 chars) that cannot be reached through BPE merging from individual characters and must be matched as whole units.

### The Failure Mode

Without greedy pre-matching, CONTROL tokens go through normalization and character-level BPE:

```
Input: "<start_of_turn>"
Normalization: "▁<start_of_turn>"
Initial tokens: ▁ < s t a r t _ o f _ t u r n >
BPE merges: ▁< star t _of _turn >   (partial merges, wrong tokens)
```

The BPE merge loop may combine some adjacent characters into subword tokens, but it can never reconstruct the original CONTROL token because `<start_of_turn>` is stored with a sentinel score of `-1000.0` and exists in the vocabulary as a single unit, not as a BPE merge product.

### Impact on Chat Template Tokenization

Gemma 3 chat templates use CONTROL tokens extensively:

```
<bos><start_of_turn>user
What is 2+2?<end_of_turn>
<start_of_turn>model
```

The logit verifier test showed a token count mismatch: **ours=35, ref=16**. The 19 extra tokens came from CONTROL tokens being shattered into characters.

## The Fix

### 1. Recognize CONTROL Tokens (tokenizer)

Added `TOKEN_TYPE_CONTROL` constant and expanded the type check in `from_gguf()`:

```rust
const TOKEN_TYPE_CONTROL: i32 = 2;
const TOKEN_TYPE_USER_DEFINED: i32 = 4;

// In from_gguf():
if (ttype == TOKEN_TYPE_USER_DEFINED || ttype == TOKEN_TYPE_CONTROL) && i < id_to_token.len() {
```

CONTROL tokens are now collected into the same `user_defined` list, sorted by length descending for greedy longest-match. The existing `split_user_defined()` and `encode()` logic handles the rest with no changes.

### 2. Fix BOS Double-Prepend (test)

The logit verifier test manually prepends `bos_id = 2` before encoding. Now that `<bos>` is recognized as a CONTROL token, `encode("<bos>...")` produces token ID 2 as its first output. The test was updated to detect this and skip the manual prepend:

```rust
let encoded = tokenizer.encode(prompt_text).expect("Tokenization failed");
let our_token_ids = if encoded.first() == Some(&bos_id) {
    encoded
} else {
    let mut ids = vec![bos_id];
    ids.extend(encoded);
    ids
};
```

## Verification

After the fix, the logit verifier shows token ID match (16 tokens, no WARN) on the chat_template prompt, and the tokenizer verifier still passes (CONTROL tokens don't appear in those test strings).

## Files Changed

| File | Change |
|---|---|
| `rustml/nlp/main/src/core/tokenizer/mod.rs` | Add `TOKEN_TYPE_CONTROL`, expand type check to include CONTROL tokens |
| `rustml/nlp/tests/logit_verifier_test.rs` | Detect BOS in encoded output to avoid double-prepend |

## Relationship to Previous Tokenizer Fixes

This is the third tokenizer fix for Gemma 3:

1. **`gemma3-tokenizer-fix.md`** — Replaced broken greedy longest-match with proper SentencePiece BPE
2. **`gguf-user-defined-tokens.md`** — Added greedy pre-BPE matching for USER_DEFINED tokens (type 4)
3. **This fix** — Extended pre-BPE matching to CONTROL tokens (type 2)

## Lessons Learned

1. **GGUF type 2 and type 4 share the same encoding requirement** — Both CONTROL and USER_DEFINED tokens are multi-character strings with sentinel scores that cannot be reached through BPE merging. They must be matched greedily on the raw text before normalization.

2. **Chat templates expose special token bugs** — Simple prompts like "The capital of France is" don't contain CONTROL tokens, so the bug was invisible until testing with chat-formatted prompts that include `<start_of_turn>` and `<end_of_turn>`.

3. **Token count is a fast diagnostic** — A 35 vs 16 token count mismatch immediately pointed to special tokens being shattered, even before examining the individual IDs.
