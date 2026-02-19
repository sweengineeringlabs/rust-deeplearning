use std::sync::atomic::{AtomicUsize, Ordering};

/// Global threshold for switching softmax from sequential to parallel (rayon).
pub(crate) static SOFTMAX_PAR_THRESHOLD: AtomicUsize = AtomicUsize::new(4096);

/// Global threshold for switching batched_matmul from sequential to parallel (rayon).
pub(crate) static BATCHED_MATMUL_PAR_THRESHOLD: AtomicUsize = AtomicUsize::new(4096);

/// Global threshold for switching F32 gemv (M=1) to custom parallel path.
/// When N >= this threshold, use parallel gemv instead of faer.
pub(crate) static GEMV_PAR_THRESHOLD: AtomicUsize = AtomicUsize::new(4096);

/// Runtime configuration for parallelism and thread management.
/// Must be applied (via `apply()`) before any computation to take effect.
pub struct RuntimeConfig {
    /// Number of threads for faer and rayon parallelism.
    /// 0 means auto-detect (use all available cores).
    pub num_threads: usize,
    /// Element count below which softmax uses a sequential path (default 4096).
    pub softmax_par_threshold: usize,
    /// Element count below which batched_matmul uses a sequential path (default 4096).
    pub batched_matmul_par_threshold: usize,
    /// Minimum N (out_features) for parallel F32 gemv when M=1 (default 4096).
    pub gemv_par_threshold: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            num_threads: 0,
            softmax_par_threshold: 4096,
            batched_matmul_par_threshold: 4096,
            gemv_par_threshold: 4096,
        }
    }
}

impl RuntimeConfig {
    /// Apply this runtime configuration globally.
    ///
    /// Sets faer's global parallelism and optionally configures
    /// rayon's global thread pool. Writes optimization thresholds
    /// to global atomics. Must be called before any computation
    /// (matmul, attention, etc.) for settings to take effect.
    pub fn apply(&self) -> Result<(), crate::api::error::TensorError> {
        use faer::{Parallelism, set_global_parallelism};

        if self.num_threads == 0 {
            set_global_parallelism(Parallelism::Rayon(0));
        } else {
            set_global_parallelism(Parallelism::Rayon(self.num_threads));
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.num_threads)
                .build_global()
                .map_err(|e| crate::api::error::TensorError::InvalidOperation(
                    format!("Failed to set rayon thread pool: {}", e)
                ))?;
        }

        // Write optimization thresholds to global atomics
        SOFTMAX_PAR_THRESHOLD.store(self.softmax_par_threshold, Ordering::Relaxed);
        BATCHED_MATMUL_PAR_THRESHOLD.store(self.batched_matmul_par_threshold, Ordering::Relaxed);
        GEMV_PAR_THRESHOLD.store(self.gemv_par_threshold, Ordering::Relaxed);

        // Log SIMD capabilities
        let simd = Self::detect_simd();
        eprintln!("[runtime] SIMD: {}", simd);

        // Log thread count
        let threads = rayon::current_num_threads();
        eprintln!("[runtime] Rayon threads: {}", threads);

        Ok(())
    }

    /// Detect available SIMD instruction sets.
    pub fn detect_simd() -> &'static str {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return "AVX2";
            }
            if is_x86_feature_detected!("sse2") {
                return "SSE2";
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on aarch64
            return "NEON";
        }
        "scalar"
    }
}

/// Optimization profiles for A/B benchmarking.
///
/// Controls rayon thresholds and whether in-place/buffered optimizations are used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptProfile {
    /// All optimizations enabled (default behavior).
    Optimized,
    /// All optimizations disabled: allocating paths, thresholds set to MAX.
    Baseline,
    /// Lower thresholds (1024) for aggressive parallelism.
    Aggressive,
}

impl OptProfile {
    /// Build a `RuntimeConfig` matching this profile.
    pub fn runtime_config(&self) -> RuntimeConfig {
        match self {
            OptProfile::Optimized => RuntimeConfig::default(),
            OptProfile::Baseline => RuntimeConfig {
                softmax_par_threshold: usize::MAX,
                batched_matmul_par_threshold: usize::MAX,
                gemv_par_threshold: usize::MAX,
                ..RuntimeConfig::default()
            },
            OptProfile::Aggressive => RuntimeConfig {
                softmax_par_threshold: 1024,
                batched_matmul_par_threshold: 1024,
                gemv_par_threshold: 1024,
                ..RuntimeConfig::default()
            },
        }
    }

    /// Whether in-place ops should be used (attention scaling, residual adds).
    pub fn use_inplace_ops(&self) -> bool {
        *self != OptProfile::Baseline
    }

    /// Whether buffered sampling should be used (pre-allocated logits/sort buffers).
    pub fn use_buffered_sampling(&self) -> bool {
        *self != OptProfile::Baseline
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let config = RuntimeConfig::default();
        assert_eq!(config.num_threads, 0);
        assert_eq!(config.softmax_par_threshold, 4096);
        assert_eq!(config.batched_matmul_par_threshold, 4096);
        assert_eq!(config.gemv_par_threshold, 4096);
    }

    #[test]
    fn test_detect_simd() {
        let simd = RuntimeConfig::detect_simd();
        assert!(!simd.is_empty());
    }

    #[test]
    fn test_opt_profile_optimized() {
        let p = OptProfile::Optimized;
        let cfg = p.runtime_config();
        assert_eq!(cfg.softmax_par_threshold, 4096);
        assert_eq!(cfg.batched_matmul_par_threshold, 4096);
        assert_eq!(cfg.gemv_par_threshold, 4096);
        assert!(p.use_inplace_ops());
        assert!(p.use_buffered_sampling());
    }

    #[test]
    fn test_opt_profile_baseline() {
        let p = OptProfile::Baseline;
        let cfg = p.runtime_config();
        assert_eq!(cfg.softmax_par_threshold, usize::MAX);
        assert_eq!(cfg.batched_matmul_par_threshold, usize::MAX);
        assert_eq!(cfg.gemv_par_threshold, usize::MAX);
        assert!(!p.use_inplace_ops());
        assert!(!p.use_buffered_sampling());
    }

    #[test]
    fn test_opt_profile_aggressive() {
        let p = OptProfile::Aggressive;
        let cfg = p.runtime_config();
        assert_eq!(cfg.softmax_par_threshold, 1024);
        assert_eq!(cfg.batched_matmul_par_threshold, 1024);
        assert_eq!(cfg.gemv_par_threshold, 1024);
        assert!(p.use_inplace_ops());
        assert!(p.use_buffered_sampling());
    }
}
