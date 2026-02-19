# Performance Optimization Status Summary

**Date:** 2026-02-19
**Branch:** `sweengineeringlabs`
**Platform:** Windows 11, DDR5-5500 (16GB), 8 threads

---

## Current State

### GPT-2 (163M params)

```
Decode step: ~50ms (20 tokens/sec)

Layers ████████████████████████████████████░░ 90%
  ├─ Attention (F32) ██████████░░░░░░░░░░░░░ 50%
  └─ FFN (Q8_0)      ██████████░░░░░░░░░░░░░ 50%
Projection ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  6%
Other ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  4%
```

### Gemma 3 1B (1.3B params)

```
Decode step: ~200ms (5 tokens/sec)

Layers ████████████████████████████████████░░ 87%
  ├─ Attention (Q8_0) ███░░░░░░░░░░░░░░░░░░░ 15%
  └─ FFN (Q8_0)       █████████████████████░ 85%
Projection ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 11%
Other ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  2%
```

---

## Target State

### GPT-2 Target

```
Decode step: ~40ms (25 tokens/sec) [+25%]

Task #2: Quantize attention F32 → Q8_0
         Potential: 10-20% layer speedup

Attention (F32) ██████████  →  Attention (Q8_0) ████
           50%                              ~30%
```

### Gemma 3 Target

```
Decode step: ~170ms (6 tokens/sec) [+20%]

Task #6: Optimize FFN down_proj (79% → 90% BW util)
         Potential: ~15% FFN speedup

FFN ████████████████████  →  FFN ████████████████░░
    85% @ 2.09ms/layer        85% @ 1.7ms/layer
```

---

## Bandwidth Analysis

### System Memory Bandwidth

| Level | Bandwidth | Notes |
|-------|-----------|-------|
| DDR5-5500 Theoretical | 88 GB/s | Spec sheet |
| Parallel Read (measured) | 20-22 GB/s | 8-thread benchmark |
| Parallel Copy (measured) | 14 GB/s | Read + Write |

### Code Bandwidth Utilization

| Operation | Measured | Peak | Utilization |
|-----------|----------|------|-------------|
| lm_head | 18 GB/s | ~20 GB/s | **85%** |
| Gate+Up | 13.9 GB/s | ~14 GB/s | **99%** |
| Down_proj | 11.1 GB/s | ~14 GB/s | **79%** |

---

## The Hard Truth

```
                Theoretical          Your System        Current Code
                ───────────          ───────────        ────────────
DDR5-5500:         88 GB/s     →       14-22 GB/s   →    11-18 GB/s
                                       (measured)        (achieved)

                ▼ 75% lost to          ▼ 15-20% headroom
                system overhead         (diminishing returns)
```

**Conclusion:** Code is already at **79-99%** of achievable bandwidth. Major speedups require algorithmic changes (flash attention, better fusion), not tuning.

---

## Backlog Status

| Task | Description | Status | Expected Gain |
|------|-------------|--------|---------------|
| #1 | Trace-level profiling | ✓ Done | Foundation |
| #2 | Q8_0 for GPT-2 attention | Ready | 10-20% |
| #3 | Reduce timing jitter | Pending | Consistency |
| #4 | Attention optimization | Pending | 5-10% |
| #5 | lm_head variance | ✓ Resolved | N/A (artifact) |
| #6 | FFN optimization | Ready | 15-20% |
| #7 | Prefill optimization | Low | TTFT |

---

## Next Steps

1. **Task #6** — Optimize down_proj (79% → 90% utilization)
2. **Task #2** — Enable Q8_0 for GPT-2 attention
3. **Algorithmic** — Consider flash attention for larger context

---

## Related Documents

- `docs/7-operations/audit/perf_round6_bottleneck_analysis_report.md`
- `docs/7-operations/audit/perf_round6_trace_level_analysis.md`
- `docs/3-design/guides/roofline_model.md`
- `rustml/BACKLOG.md`
