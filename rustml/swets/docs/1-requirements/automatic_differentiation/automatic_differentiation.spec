# Feature Spec: Automatic Differentiation

**Version:** 1.0
**Status:** Draft
**Section:** 2.2

## Requirements

| ID | Source | Title | Priority | Verification | Acceptance |
|-----|--------|-------|----------|--------------|------------|
| REQ-001 | FR-200 | `GradientTape` records forward operations as `TapeEntry` list | Must | Test | — |
| REQ-002 | FR-201 | `TapeEntry` stores: `BackwardOp`, output ID, input IDs, saved tensors | Must | Test | — |
| REQ-003 | FR-202 | `BackwardOp` trait with `backward(grad_output, saved) -> Vec<input_grads>` | Must | Test | — |
| REQ-004 | FR-203 | `GradientTape::backward(loss_id)` replays tape in reverse, accumulates gradients in `HashMap<TensorId, Tensor>` | Must | Test | — |
| REQ-005 | FR-204 | `GradientTape::grad(id)` retrieves gradient for a tensor | Must | Test | — |
| REQ-006 | FR-205 | `GradientTape::clear()` resets ops and gradients between training steps | Must | Test | — |
| REQ-007 | FR-206 | `tape.enabled` flag to disable recording (inference mode) | Must | Test | — |
| REQ-008 | FR-207 | `Layer::forward` accepts `Option<&mut GradientTape>` — `None` means no recording | Must | Test | — |
| REQ-009 | FR-208 | Scoped `no_grad` helper that temporarily disables tape | Should | Test | — |
| REQ-010 | FR-209 | Non-tracking `_raw` variants of ops (e.g. `matmul_raw`) for use inside backward implementations | Must | Test | — |
| REQ-011 | FR-210 | MatMul | Must | Test | `grad_a = grad @ b^T`, `grad_b = a^T @ grad` |
| REQ-012 | FR-211 | Add | Must | Test | `unbroadcast(grad)` per input |
| REQ-013 | FR-212 | Mul (element-wise) | Must | Test | `grad * other_input` |
| REQ-014 | FR-213 | Sigmoid | Must | Test | `grad * sigmoid(x) * (1 - sigmoid(x))` |
| REQ-015 | FR-214 | Tanh | Must | Test | `grad * (1 - tanh(x)^2)` |
| REQ-016 | FR-215 | ReLU | Must | Test | `grad * (input > 0)` |
| REQ-017 | FR-216 | Softmax | Must | Test | `s * (g - sum(g * s, dim))` |
| REQ-018 | FR-217 | Conv1d | Must | Test | Transposed convolution for input grad, cross-correlation for weight grad |

## Acceptance Criteria

- **REQ-001** (FR-200): To be defined
- **REQ-002** (FR-201): To be defined
- **REQ-003** (FR-202): To be defined
- **REQ-004** (FR-203): To be defined
- **REQ-005** (FR-204): To be defined
- **REQ-006** (FR-205): To be defined
- **REQ-007** (FR-206): To be defined
- **REQ-008** (FR-207): To be defined
- **REQ-009** (FR-208): To be defined
- **REQ-010** (FR-209): To be defined
- **REQ-011** (FR-210): `grad_a = grad @ b^T`, `grad_b = a^T @ grad`
- **REQ-012** (FR-211): `unbroadcast(grad)` per input
- **REQ-013** (FR-212): `grad * other_input`
- **REQ-014** (FR-213): `grad * sigmoid(x) * (1 - sigmoid(x))`
- **REQ-015** (FR-214): `grad * (1 - tanh(x)^2)`
- **REQ-016** (FR-215): `grad * (input > 0)`
- **REQ-017** (FR-216): `s * (g - sum(g * s, dim))`
- **REQ-018** (FR-217): Transposed convolution for input grad, cross-correlation for weight grad

