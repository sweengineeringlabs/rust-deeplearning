# Feature Spec: Neural Network Layers

**Version:** 1.0
**Status:** Draft
**Section:** 2.3

## Requirements

| ID | Source | Title | Priority | Verification | Acceptance |
|-----|--------|-------|----------|--------------|------------|
| REQ-001 | FR-300 | `Layer` trait: `forward(&mut self, input, tape)`, `parameters()`, `parameters_mut()`, `train()`, `eval()`, `parameter_count()` | Must | Test | — |
| REQ-002 | FR-301 | `Sequential` container chaining arbitrary layers | Must | Test | — |
| REQ-003 | FR-302 | `Linear` layer with Xavier (default) and He initialization | Must | Test | — |
| REQ-004 | FR-303 | `LSTM` layer: multi-layer, configurable hidden size, optional bidirectional, dropout between layers | Must | Test | — |
| REQ-005 | FR-304 | `Conv1d` layer with stride, padding, dilation | Must | Test | — |
| REQ-006 | FR-305 | Activation layers: `ReLU`, `Tanh`, `Sigmoid`, `GELU`, `SiLU` | Must | Test | — |
| REQ-007 | FR-306 | `LayerNorm` with configurable epsilon | Must | Test | — |
| REQ-008 | FR-307 | `BatchNorm1d` with running statistics (updated via `&mut self` in training mode) | Should | Test | — |
| REQ-009 | FR-308 | `Dropout` layer: active in training, identity in eval | Must | Test | — |
| REQ-010 | FR-309 | `model_summary()` prints total/trainable/frozen parameter counts | Should | Test | — |

## Acceptance Criteria

- **REQ-001** (FR-300): To be defined
- **REQ-002** (FR-301): To be defined
- **REQ-003** (FR-302): To be defined
- **REQ-004** (FR-303): To be defined
- **REQ-005** (FR-304): To be defined
- **REQ-006** (FR-305): To be defined
- **REQ-007** (FR-306): To be defined
- **REQ-008** (FR-307): To be defined
- **REQ-009** (FR-308): To be defined
- **REQ-010** (FR-309): To be defined

