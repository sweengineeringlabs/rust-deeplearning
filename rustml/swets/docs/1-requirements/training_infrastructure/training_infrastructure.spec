# Feature Spec: Training Infrastructure

**Version:** 1.0
**Status:** Draft
**Section:** 2.9

## Requirements

| ID | Source | Title | Priority | Verification | Acceptance |
|-----|--------|-------|----------|--------------|------------|
| REQ-001 | FR-900 | `Trainer` struct: owns model, optimizer, loss function | Must | Test | — |
| REQ-002 | FR-901 | `train_epoch()`: forward pass, backward pass, gradient clipping, optimizer step — fresh tape per batch | Must | Test | — |
| REQ-003 | FR-902 | `validate()`: forward pass with no tape (no gradient computation) | Must | Test | — |
| REQ-004 | FR-903 | `fit()`: epoch loop with train/validate, logging, scheduler step, early stopping check | Must | Test | — |
| REQ-005 | FR-904 | `predict()`: switches model to eval mode, runs forward with no tape | Must | Test | — |
| REQ-006 | FR-905 | Early stopping with configurable patience | Must | Test | — |
| REQ-007 | FR-906 | Best-model checkpointing on validation loss improvement | Must | Test | — |
| REQ-008 | FR-907 | Builder pattern: `with_scheduler()`, `with_early_stopping()`, `with_grad_clip()`, `with_checkpoint_dir()` | Should | Test | — |

## Acceptance Criteria

- **REQ-001** (FR-900): To be defined
- **REQ-002** (FR-901): To be defined
- **REQ-003** (FR-902): To be defined
- **REQ-004** (FR-903): To be defined
- **REQ-005** (FR-904): To be defined
- **REQ-006** (FR-905): To be defined
- **REQ-007** (FR-906): To be defined
- **REQ-008** (FR-907): To be defined

