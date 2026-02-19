# Feature Spec: Model Architectures

**Version:** 1.0
**Status:** Draft
**Section:** 2.8

## Requirements

| ID | Source | Title | Priority | Verification | Acceptance |
|-----|--------|-------|----------|--------------|------------|
| REQ-001 | FR-800 | **TCN** (Temporal Convolutional Network): causal padding, exponentially increasing dilation, residual connections, global average pooling | Must | Test | — |
| REQ-002 | FR-801 | **TimeSeriesTransformer**: input embedding, sinusoidal positional encoding, N transformer blocks with causal masking, last-timestep projection | Must | Test | — |
| REQ-003 | FR-802 | Multi-head attention with Q/K/V projections, scaled dot-product, causal + padding mask support | Must | Test | — |
| REQ-004 | FR-803 | **N-BEATS**: FC stacks with backcast subtraction (doubly-residual), forecast accumulation, configurable stacks/blocks/layers | Should | Test | — |
| REQ-005 | FR-804 | LSTM-based forecasting model (built from FR-303) | Must | Test | — |

## Acceptance Criteria

- **REQ-001** (FR-800): To be defined
- **REQ-002** (FR-801): To be defined
- **REQ-003** (FR-802): To be defined
- **REQ-004** (FR-803): To be defined
- **REQ-005** (FR-804): To be defined

