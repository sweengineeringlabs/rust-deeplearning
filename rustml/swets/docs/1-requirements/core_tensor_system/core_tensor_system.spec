# Feature Spec: Core Tensor System

**Version:** 1.0
**Status:** Draft
**Section:** 2.1

## Requirements

| ID | Source | Title | Priority | Verification | Acceptance |
|-----|--------|-------|----------|--------------|------------|
| REQ-001 | FR-100 | Tensor struct with unique `TensorId`, `Arc<Storage>` data, shape, strides, dtype, device | Must | Test | — |
| REQ-002 | FR-101 | `TensorId` generated via atomic counter for tape tracking | Must | Test | — |
| REQ-003 | FR-102 | `Storage` enum: `Owned(Vec<u8>)`, `View { parent, offset, len }`, `MMap` | Must | Test | — |
| REQ-004 | FR-103 | `DType` enum matching rustml-core: F32, F16, BF16, I8, U8, Q8_0, Q4_0, Q4_1 | Must | Test | — |
| REQ-005 | FR-104 | Thread-local `TensorPool` for buffer recycling during backward pass | Should | Test | — |
| REQ-006 | FR-105 | Creation ops: `zeros`, `ones`, `randn`, `uniform`, `from_vec`, `arange` | Must | Test | — |
| REQ-007 | FR-106 | Shape ops: `reshape`, `transpose`, `permute`, `squeeze`, `unsqueeze`, `flatten`, `view` | Must | Test | — |
| REQ-008 | FR-107 | Indexing ops: `slice`, `index_select`, `gather`, `masked_select` | Must | Test | — |
| REQ-009 | FR-108 | Element-wise math: add, sub, mul, div, pow, sqrt, exp, log, abs, neg | Must | Test | — |
| REQ-010 | FR-109 | Reductions: sum, mean, max, min, std, var | Must | Test | — |
| REQ-011 | FR-110 | `matmul` and `dot` with broadcasting | Must | Test | — |
| REQ-012 | FR-111 | `conv1d` with stride, padding, and dilation parameters | Must | Test | — |
| REQ-013 | FR-112 | `concat`, `stack`, `split` tensor joining/splitting | Must | Test | — |
| REQ-014 | FR-113 | Row-major strides computation and flat offset indexing | Must | Test | — |

## Acceptance Criteria

- **REQ-001** (FR-100): To be defined
- **REQ-002** (FR-101): To be defined
- **REQ-003** (FR-102): To be defined
- **REQ-004** (FR-103): To be defined
- **REQ-005** (FR-104): To be defined
- **REQ-006** (FR-105): To be defined
- **REQ-007** (FR-106): To be defined
- **REQ-008** (FR-107): To be defined
- **REQ-009** (FR-108): To be defined
- **REQ-010** (FR-109): To be defined
- **REQ-011** (FR-110): To be defined
- **REQ-012** (FR-111): To be defined
- **REQ-013** (FR-112): To be defined
- **REQ-014** (FR-113): To be defined

