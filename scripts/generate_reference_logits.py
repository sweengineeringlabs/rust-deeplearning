#!/usr/bin/env python3
"""Generate reference logits from llama-cpp-python for verifying Rust forward pass.

Usage:
    python3 scripts/generate_reference_logits.py --model /path/to/model.gguf

Outputs:
    rustml/nlp/tests/fixtures/gemma3_reference_logits.json
"""

import argparse
import json
import math
import os
import sys

import numpy as np
from llama_cpp import Llama


TEST_PROMPTS = [
    {
        "name": "plain_text",
        "text": "The capital of France is",
        "description": "Plain text completion - should predict 'Paris'",
    },
    {
        "name": "chat_template",
        "text": "<bos><start_of_turn>user\nWhat is 1+1?<end_of_turn>\n<start_of_turn>model\n",
        "description": "Chat template format for IT model",
    },
    {
        "name": "short_input",
        "text": "Hi",
        "description": "Short edge case - minimal input",
    },
]

TOP_K = 20


def extract_logit_info(logits: np.ndarray, top_k: int = TOP_K) -> dict:
    """Extract top-k logits and statistics from a logit vector."""
    # Top-k by value
    top_indices = np.argsort(logits)[::-1][:top_k]
    top_logits = [
        {"token_id": int(idx), "logit": float(logits[idx])}
        for idx in top_indices
    ]

    return {
        "top_logits": top_logits,
        "stats": {
            "mean": float(np.mean(logits)),
            "std": float(np.std(logits)),
            "min": float(np.min(logits)),
            "max": float(np.max(logits)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate reference logits for Rust verification")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument(
        "--output",
        default="rustml/nlp/tests/fixtures/gemma3_reference_logits.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {args.model}")
    llm = Llama(
        model_path=args.model,
        n_ctx=256,
        logits_all=True,
        verbose=False,
    )
    print(f"  Vocab size: {llm.n_vocab()}")

    results = []

    for prompt_info in TEST_PROMPTS:
        name = prompt_info["name"]
        text = prompt_info["text"]
        print(f"\nProcessing: {name}")
        print(f"  Text: {text!r}")

        # Tokenize - use special=True to handle <bos>, <start_of_turn>, etc.
        # If the prompt already starts with <bos>, don't add another one.
        has_bos = text.startswith("<bos>")
        token_ids = llm.tokenize(text.encode("utf-8"), add_bos=not has_bos, special=True)
        print(f"  Token IDs ({len(token_ids)}): {token_ids}")

        # Run forward pass (eval) to get logits for all positions
        llm.reset()
        llm.eval(token_ids)

        # Get logits for the last token position
        # llama-cpp-python stores logits for each evaluated token
        # After eval, scores[i] has logits after processing token i
        # We want the last position's logits (prediction for next token)
        last_pos = len(token_ids) - 1
        logits = np.array(llm.scores[last_pos], dtype=np.float32)

        logit_info = extract_logit_info(logits)

        # Decode top-1 for sanity check
        top1_id = logit_info["top_logits"][0]["token_id"]
        top1_token = llm.detokenize([top1_id]).decode("utf-8", errors="replace")
        print(f"  Top-1 prediction: id={top1_id} -> {top1_token!r}")
        print(f"  Logit stats: mean={logit_info['stats']['mean']:.4f}, "
              f"std={logit_info['stats']['std']:.4f}, "
              f"min={logit_info['stats']['min']:.4f}, "
              f"max={logit_info['stats']['max']:.4f}")

        # Show top-5
        print("  Top-5:")
        for entry in logit_info["top_logits"][:5]:
            tok = llm.detokenize([entry["token_id"]]).decode("utf-8", errors="replace")
            print(f"    id={entry['token_id']:<6} logit={entry['logit']:>8.3f}  {tok!r}")

        results.append({
            "name": name,
            "description": prompt_info["description"],
            "prompt": text,
            "token_ids": token_ids,
            "num_tokens": len(token_ids),
            "last_position_logits": logit_info,
        })

    # Save
    output_data = {
        "model_path": os.path.basename(args.model),
        "generator": "llama-cpp-python",
        "top_k": TOP_K,
        "prompts": results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved reference logits to: {args.output}")
    print(f"  File size: {os.path.getsize(args.output)} bytes")


if __name__ == "__main__":
    main()
