# Feature Spec: Loss Functions

**Version:** 1.0
**Status:** Draft
**Section:** 2.6

## Requirements

| ID | Source | Title | Priority | Verification | Acceptance |
|-----|--------|-------|----------|--------------|------------|
| REQ-001 | FR-600 | `Loss` trait: `forward(predictions, targets, tape)` | Must | Test | — |
| REQ-002 | FR-601 | `MSELoss` — mean squared error | Must | Test | — |
| REQ-003 | FR-602 | `MAELoss` — mean absolute error | Must | Test | — |
| REQ-004 | FR-603 | `HuberLoss` — quadratic near zero, linear far from zero, configurable delta | Must | Test | — |
| REQ-005 | FR-604 | `CrossEntropyLoss` — numerically stable log-softmax + NLL | Should | Test | — |
| REQ-006 | FR-605 | `QuantileLoss` — asymmetric loss for quantile regression | Should | Test | — |

## Acceptance Criteria

- **REQ-001** (FR-600): To be defined
- **REQ-002** (FR-601): To be defined
- **REQ-003** (FR-602): To be defined
- **REQ-004** (FR-603): To be defined
- **REQ-005** (FR-604): To be defined
- **REQ-006** (FR-605): To be defined

