# Roofline Model for Performance Analysis

A framework for understanding whether code is limited by memory bandwidth or compute.

---

## Overview

The Roofline Model (Berkeley, 2009) visualizes performance limits:

```
                        ┌─────────────── Compute ceiling (peak FLOP/s)
                        │
Performance             │         ╱
(FLOP/s)               │       ╱
                        │     ╱  ← Roofline
                        │   ╱
                        │ ╱
                        ╱─────────────── Memory ceiling (bandwidth × AI)
                      ╱ │
                    ╱   │
                  ╱     │
                ╱       │
              ╱         │
            ──────────────────────────
                    Arithmetic Intensity (FLOP/byte)
```

---

## Core Formula

```
Arithmetic Intensity (AI) = FLOPs performed / Bytes transferred
```

| AI | Bound by | Optimization target |
|----|----------|---------------------|
| Low (< ridge point) | Memory bandwidth | Improve data locality, prefetching |
| High (> ridge point) | Compute (FLOP/s) | Better SIMD, parallelism |

---

## Applying to Q8_0 Matmul

### What happens per weight block

```
Benchmark:     Read → Write                              (memory only)
Matmul:        Read → Dequantize → Multiply → Add → Write
```

Steps:
1. Read Q8_0 block (34 bytes)
2. Dequantize 32 values to F32
3. Multiply 32 values × input
4. Accumulate sums
5. Write output

### Arithmetic Intensity

```
Q8_0 matmul:  ~32 FLOPs / 34 bytes ≈ 0.94 FLOP/byte
```

This is very low → **memory-bound operation**

---

## Bandwidth Utilization Analysis

Since matmul is memory-bound, we measure performance by comparing achieved bandwidth to peak bandwidth.

### Benchmarking Peak Bandwidth

Run parallel memory benchmark to establish system limits:

```rust
// Parallel copy measures read+write bandwidth
// Parallel read-sum measures read-only bandwidth
```

Example results (DDR5-5500, 8 threads):

| Benchmark | Bandwidth |
|-----------|-----------|
| Parallel Read | ~20-22 GB/s |
| Parallel Copy (Read+Write) | ~14 GB/s |

### Interpreting Utilization

| Utilization | Meaning |
|-------------|---------|
| 90-100% | Near-optimal, memory-bound as expected |
| 70-90% | Good balance of compute and memory overhead |
| 50-70% | Room for improvement (prefetching, cache) |
| <50% | Inefficient memory access or compute-heavy |

**Important:** 100% is impossible for matmul because:
- Benchmark only moves memory
- Matmul also computes (dequantize, multiply, accumulate)
- During compute, memory bandwidth is unused

---

## Example Analysis

From rustml profiling (2026-02-19):

| Operation | Measured | Peak (benchmark) | Utilization |
|-----------|----------|------------------|-------------|
| lm_head | 18 GB/s | ~20 GB/s (read-heavy) | ~85% |
| Gate+Up | 13.9 GB/s | ~14 GB/s (copy) | ~99% |
| Down_proj | 11.1 GB/s | ~14 GB/s (copy) | ~79% |

**Interpretation:**
- Gate+Up at 99% = optimal, compute is nearly free
- Down_proj at 79% = 21% overhead from compute/sync, room for improvement

---

## Optimization Strategies by Bound Type

### Memory-Bound (low AI)

- Prefetch next data while computing current
- Improve cache locality
- Reduce thread synchronization
- Use streaming stores (bypass cache for write-only)

### Compute-Bound (high AI)

- Better SIMD utilization (AVX2/AVX-512)
- Loop unrolling
- Reduce branch mispredictions
- More parallelism

---

## References

- Williams, S., Waterman, A., & Patterson, D. (2009). "Roofline: An Insightful Visual Performance Model for Multicore Architectures"
- [Berkeley Roofline Model](https://crd.lbl.gov/divisions/amcr/computer-science-amcr/par/research/roofline/)

---

## Related Files

- `docs/7-operations/audit/perf_round6_trace_level_analysis.md` — Bandwidth measurements
- `rustml/quant/main/src/core/quantize.rs` — Q8_0 matmul with bandwidth logging
