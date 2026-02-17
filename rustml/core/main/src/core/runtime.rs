/// Runtime configuration for parallelism and thread management.
/// Must be applied (via `apply()`) before any computation to take effect.
pub struct RuntimeConfig {
    /// Number of threads for faer and rayon parallelism.
    /// 0 means auto-detect (use all available cores).
    pub num_threads: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self { num_threads: 0 }
    }
}

impl RuntimeConfig {
    /// Apply this runtime configuration globally.
    ///
    /// Sets faer's global parallelism and optionally configures
    /// rayon's global thread pool. Must be called before any
    /// computation (matmul, attention, etc.) for settings to take effect.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let config = RuntimeConfig::default();
        assert_eq!(config.num_threads, 0);
    }

    #[test]
    fn test_detect_simd() {
        let simd = RuntimeConfig::detect_simd();
        assert!(!simd.is_empty());
    }
}
