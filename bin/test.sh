#!/usr/bin/env bash
source "$(cd "$(dirname "$0")/.." && pwd)/lib/common.sh"

SUITE="${1:-all}"

if [ "$SUITE" = "scripts" ]; then
  exec bash "$REPO_ROOT/bin/tests/runner.sh"
fi

preflight

run_core_tests() {
  echo "==> Testing core..."
  cargo test -p rustml-core
}

run_nn_tests() {
  echo "==> Testing nn..."
  cargo test -p rustml-nn
}

run_nlp_tests() {
  echo "==> Testing nlp..."
  cargo test -p rustml-nlp
}

run_tokenizer_tests() {
  echo "==> Testing tokenizer..."
  cargo test -p rustml-tokenizer
}

run_gguf_tests() {
  echo "==> Testing gguf..."
  cargo test -p rustml-gguf
}

run_quant_tests() {
  echo "==> Testing quant..."
  cargo test -p rustml-quant
}

run_hub_tests() {
  echo "==> Testing hub..."
  cargo test -p rustml-hub
}

case "$SUITE" in
  core)      run_core_tests ;;
  nn)        run_nn_tests ;;
  nlp)       run_nlp_tests ;;
  tokenizer) run_tokenizer_tests ;;
  gguf)      run_gguf_tests ;;
  quant)     run_quant_tests ;;
  hub)       run_hub_tests ;;
  all)
    run_core_tests
    run_nn_tests
    run_nlp_tests
    run_tokenizer_tests
    run_gguf_tests
    run_quant_tests
    run_hub_tests
    ;;
  *)
    echo "Usage: ./rdl test [core|nn|nlp|tokenizer|gguf|quant|hub|scripts|all]"
    exit 1
    ;;
esac

echo "==> Tests complete ($SUITE)"
