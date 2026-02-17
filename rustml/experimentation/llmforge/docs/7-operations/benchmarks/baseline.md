# Performance Baseline

> **TLDR:** Reference inference throughput on known hardware. All future optimizations are measured against these numbers.

**Date**: 2026-02-13
**Commit**: Post-Phase 6 (GGUF tokenizer auto-generation)

---

## Hardware

| Component | Value |
|-----------|-------|
| CPU | AMD Ryzen 5 7520U with Radeon Graphics |
| Threads | 8 |
| RAM | 6.6 GB |
| OS | Linux 6.6.87.2-microsoft-standard-WSL2 |
| Platform | WSL2 on Windows |

## Inference Throughput

### Pre-Phase 7 (parameter count bug)

| Model | Format | Params (reported) | Quant | Build | tok/s |
|-------|--------|-------------------|-------|-------|-------|
| GPT-2 124M | SafeTensors | 78.0M* | F32 | release | **12.0** |
| TinyLlama 1.1B | GGUF | 131.1M* | Q4_0 | release | **1.8** |
| TinyLlama 1.1B | GGUF | 131.1M* | Q4_0 | debug | **0.03** |

*\* Parameter count was wrong — `LlmModel::parameter_count()` skipped transformer layers entirely. Fixed in Phase 7.*

### Post-Phase 7 (parameter count fix + SIMD/thread logging)

| Model | Format | Params (reported) | Quant | Build | tok/s | SIMD | Threads |
|-------|--------|-------------------|-------|-------|-------|------|---------|
| GPT-2 124M | SafeTensors | 163.0M | F32 | release | **13.0** | AVX2 | 8 |
| TinyLlama 1.1B | GGUF | 1,100.1M | Q4_0 | release | **1.6** | AVX2 | 8 |

**Test command:**
```bash
./target/release/llmforge run --model <model> --prompt "Once upon a time" --max-tokens 64 --temperature 0
```

## Notes

- Release vs debug: ~40x speedup (1.8 vs 0.03 tok/s)
- KV cache is wired in via `Generator::generate_stream()` using `make_cache()` + `forward_with_cache()`
- SIMD dispatch confirmed: AVX2 active for Q4_0 dot products
- Rayon parallelism confirmed: 8 threads via `par_chunks_mut()` in quantized matmul
- Tokenizer: GPT-2 uses HFTokenizer (from repo); TinyLlama uses auto-generated tokenizer.json from GGUF metadata
- Generation quality: Phase 7 fixed critical dequantization bugs (Q6_K, Q4_0, Q4_K, Q5_K, Q3_K, Q2_K element ordering), BOS token wiring, and SentencePiece decode. GPT-2 and TinyLlama now produce coherent topical output. Known remaining gap: TinyLlama open-ended generation is lower quality than llama.cpp due to dequant-to-F32 matmul accumulating different numerical errors than native Q4_0×Q8_0 integer dot products over 22 layers. Future fix: native quantized matmul.
