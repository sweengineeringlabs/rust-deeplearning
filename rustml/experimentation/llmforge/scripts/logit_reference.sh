#!/usr/bin/env bash
#
# Reference logit dump using llama.cpp server.
#
# Starts llama-server, sends a completion request with logprobs enabled,
# and extracts top-20 token predictions per step. No Python required.
#
# Prerequisites:
#   - llama-server on PATH (build from https://github.com/ggml-org/llama.cpp
#     or install via package manager)
#   - jq for JSON processing
#   - curl
#
# Usage:
#   # Auto-detect GGUF from HuggingFace cache
#   ./scripts/logit_reference.sh
#
#   # Explicit model path
#   ./scripts/logit_reference.sh /path/to/tinyllama-1.1b-chat-v1.0.Q4_0.gguf
#
#   # Custom prompt
#   PROMPT="Hello world" ./scripts/logit_reference.sh /path/to/model.gguf

set -euo pipefail

PROMPT="${PROMPT:-The capital of France is}"
PORT="${PORT:-8199}"
TOP_LOGPROBS=20
# Prefill produces 1 token, then 3 decode steps = 4 total tokens generated
N_PREDICT=4

# ── Find model ────────────────────────────────────────────────────────
GGUF_PATH="${1:-}"
if [ -z "$GGUF_PATH" ]; then
    # Look in HuggingFace cache (same location hf-hub downloads to)
    HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}/hub"
    GGUF_PATH=$(find "$HF_CACHE" -name "tinyllama-1.1b-chat-v1.0.Q4_0.gguf" -type f 2>/dev/null | head -1)
    if [ -z "$GGUF_PATH" ]; then
        echo >&2 "ERROR: Could not find TinyLlama Q4_0 GGUF in HuggingFace cache."
        echo >&2 "Either run 'cargo run --release --example gguf_inference' first to download it,"
        echo >&2 "or provide the path: $0 /path/to/model.gguf"
        exit 1
    fi
fi
echo >&2 "Model: $GGUF_PATH"
echo >&2 "Prompt: $PROMPT"

# ── Check dependencies ────────────────────────────────────────────────
for cmd in llama-server jq curl; do
    if ! command -v "$cmd" &>/dev/null; then
        echo >&2 "ERROR: '$cmd' not found. Install it first."
        [ "$cmd" = "llama-server" ] && echo >&2 "  Build from: https://github.com/ggml-org/llama.cpp"
        exit 1
    fi
done

# ── Start server ──────────────────────────────────────────────────────
echo >&2 "Starting llama-server on port $PORT..."
llama-server \
    --model "$GGUF_PATH" \
    --port "$PORT" \
    --n-gpu-layers 0 \
    --ctx-size 512 \
    --log-disable \
    &>/dev/null &
SERVER_PID=$!

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server to be ready
echo >&2 "Waiting for server..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:${PORT}/health" | jq -e '.status == "ok"' &>/dev/null; then
        echo >&2 "Server ready."
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo >&2 "ERROR: Server process died."
        exit 1
    fi
    sleep 1
done

# ── Send completion request ───────────────────────────────────────────
# Using the OpenAI-compatible /v1/completions endpoint with logprobs.
# temperature=0 for greedy decoding (matches our Rust side).
echo >&2 "Sending completion request (n_predict=$N_PREDICT, top_logprobs=$TOP_LOGPROBS)..."

RESPONSE=$(curl -s "http://localhost:${PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "$(jq -n \
        --arg prompt "$PROMPT" \
        --argjson max_tokens "$N_PREDICT" \
        --argjson top_logprobs "$TOP_LOGPROBS" \
        '{
            prompt: $prompt,
            max_tokens: $max_tokens,
            temperature: 0,
            logprobs: $top_logprobs,
            echo: false
        }'
    )")

# ── Format output ─────────────────────────────────────────────────────
# Extract logprobs into our comparison format.
# The response has .choices[0].logprobs with:
#   .tokens[]       - generated token strings
#   .token_logprobs[] - log-probability of chosen token
#   .top_logprobs[] - array of {token: logprob} maps for top-N

echo >&2 ""
echo >&2 "--- Results ---"

# Print the generated text for quick visual check
GENERATED=$(echo "$RESPONSE" | jq -r '.choices[0].text')
echo >&2 "Generated: $GENERATED"

# Build structured JSON output matching logit_dump.rs format
echo "$RESPONSE" | jq --arg prompt "$PROMPT" '{
    prompt: $prompt,
    engine: "llama-server",
    generated_text: .choices[0].text,
    tokens: .choices[0].logprobs.tokens,
    token_logprobs: .choices[0].logprobs.token_logprobs,
    steps: [
        range(.choices[0].logprobs.tokens | length) as $i |
        {
            step: $i,
            chosen_token: .choices[0].logprobs.tokens[$i],
            chosen_logprob: .choices[0].logprobs.token_logprobs[$i],
            top_logprobs: .choices[0].logprobs.top_logprobs[$i]
        }
    ]
}'

echo >&2 ""
echo >&2 "Done. Compare top-1 tokens against: cargo run --release --example logit_dump 2>/dev/null"
