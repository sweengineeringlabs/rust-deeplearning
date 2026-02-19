# Feature Spec: Optimizers

**Version:** 1.0
**Status:** Draft
**Section:** 2.4

## Requirements

| ID | Source | Title | Priority | Verification | Acceptance |
|-----|--------|-------|----------|--------------|------------|
| REQ-001 | FR-400 | `Optimizer` trait: `step(params, tape)`, `lr()`, `set_lr()` — reads gradients from tape | Must | Test | — |
| REQ-002 | FR-401 | `SGD` with momentum, dampening, weight decay, Nesterov | Must | Test | — |
| REQ-003 | FR-402 | `Adam` with configurable betas, epsilon, L2 regularization | Must | Test | — |
| REQ-004 | FR-403 | `AdamW` with decoupled weight decay (applied before Adam update) | Must | Test | — |
| REQ-005 | FR-404 | Gradient clipping by global L2 norm (`clip_grad_norm`) | Must | Test | — |
| REQ-006 | FR-405 | Gradient clipping by value (`clip_grad_value`) | Should | Test | — |

## Acceptance Criteria

- **REQ-001** (FR-400): To be defined
- **REQ-002** (FR-401): To be defined
- **REQ-003** (FR-402): To be defined
- **REQ-004** (FR-403): To be defined
- **REQ-005** (FR-404): To be defined
- **REQ-006** (FR-405): To be defined

