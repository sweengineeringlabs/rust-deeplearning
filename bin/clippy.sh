#!/usr/bin/env bash
source "$(cd "$(dirname "$0")/.." && pwd)/lib/common.sh"

preflight

echo "==> Running cargo clippy..."
cargo clippy --workspace -- -D warnings
echo "==> Done"
