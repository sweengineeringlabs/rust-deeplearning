# Feature Spec: Rust Safety

**Version:** 1.0
**Status:** Draft
**Section:** 3.3

## Requirements

| ID | Source | Title | Priority | Verification | Acceptance |
|-----|--------|-------|----------|--------------|------------|
| REQ-001 | NFR-300 | No `unsafe` code outside SIMD kernels | Must | Analysis | — |
| REQ-002 | NFR-301 | `Storage::View` holds `Arc<Storage>` (not `Arc<Tensor>`) to prevent reference cycles | Must | Analysis | — |
| REQ-003 | NFR-302 | `GradientTape` borrow semantics must compile — design must resolve `Option<&mut GradientTape>` threading through nested layer calls (see Section 5.1) | Must | Analysis | — |
| REQ-004 | NFR-303 | All `BackwardOp` implementations must be `Send + Sync` | Must | Analysis | — |

## Acceptance Criteria

- **REQ-001** (NFR-300): To be defined
- **REQ-002** (NFR-301): To be defined
- **REQ-003** (NFR-302): To be defined
- **REQ-004** (NFR-303): To be defined

