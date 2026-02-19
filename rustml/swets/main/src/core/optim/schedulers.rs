use std::f32::consts::PI;

use crate::api::optim::{LRScheduler, Optimizer};

// ---------------------------------------------------------------------------
// FR-501: StepLR – Decay by gamma every step_size steps
// ---------------------------------------------------------------------------

pub struct StepLR {
    initial_lr: f32,
    step_size: usize,
    gamma: f32,
    current_step: usize,
}

impl StepLR {
    pub fn new(initial_lr: f32, step_size: usize, gamma: f32) -> Self {
        Self {
            initial_lr,
            step_size,
            gamma,
            current_step: 0,
        }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        optimizer.set_lr(self.get_lr());
    }

    fn get_lr(&self) -> f32 {
        let exponent = (self.current_step / self.step_size) as u32;
        self.initial_lr * self.gamma.powi(exponent as i32)
    }
}

// ---------------------------------------------------------------------------
// FR-502: CosineAnnealingLR – Cosine decay to eta_min
// ---------------------------------------------------------------------------

pub struct CosineAnnealingLR {
    initial_lr: f32,
    t_max: usize,
    eta_min: f32,
    current_step: usize,
}

impl CosineAnnealingLR {
    pub fn new(initial_lr: f32, t_max: usize, eta_min: f32) -> Self {
        Self {
            initial_lr,
            t_max,
            eta_min,
            current_step: 0,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        optimizer.set_lr(self.get_lr());
    }

    fn get_lr(&self) -> f32 {
        let cos_value = (PI * self.current_step as f32 / self.t_max as f32).cos();
        self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (1.0 + cos_value)
    }
}

// ---------------------------------------------------------------------------
// FR-503: WarmupCosineScheduler – Linear warmup then cosine decay
// ---------------------------------------------------------------------------

pub struct WarmupCosineScheduler {
    initial_lr: f32,
    warmup_steps: usize,
    total_steps: usize,
    eta_min: f32,
    current_step: usize,
}

impl WarmupCosineScheduler {
    pub fn new(
        initial_lr: f32,
        warmup_steps: usize,
        total_steps: usize,
        eta_min: f32,
    ) -> Self {
        Self {
            initial_lr,
            warmup_steps,
            total_steps,
            eta_min,
            current_step: 0,
        }
    }
}

impl LRScheduler for WarmupCosineScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.current_step += 1;
        optimizer.set_lr(self.get_lr());
    }

    fn get_lr(&self) -> f32 {
        if self.current_step < self.warmup_steps {
            // Linear warmup: scale from 0 to initial_lr
            self.initial_lr * self.current_step as f32 / self.warmup_steps as f32
        } else {
            // Cosine decay from initial_lr to eta_min over the remaining steps
            let decay_steps = self.total_steps - self.warmup_steps;
            let progress = (self.current_step - self.warmup_steps) as f32 / decay_steps as f32;
            let cos_value = (PI * progress).cos();
            self.eta_min + 0.5 * (self.initial_lr - self.eta_min) * (1.0 + cos_value)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::error::SwetsResult;
    use crate::api::tensor::Tensor;

    /// Minimal mock optimizer for testing scheduler behavior.
    struct MockOptimizer {
        lr: f32,
    }

    impl MockOptimizer {
        fn new(lr: f32) -> Self {
            Self { lr }
        }
    }

    impl Optimizer for MockOptimizer {
        fn step(&mut self, _params: &mut [&mut Tensor]) -> SwetsResult<()> {
            Ok(())
        }
        fn lr(&self) -> f32 {
            self.lr
        }
        fn set_lr(&mut self, lr: f32) {
            self.lr = lr;
        }
    }

    // -- StepLR tests -------------------------------------------------------

    #[test]
    fn step_lr_initial_lr_unchanged_before_step_size() {
        let scheduler = StepLR::new(0.1, 10, 0.5);
        // At step 0, no decay has occurred yet
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn step_lr_decays_at_step_size_boundary() {
        let mut scheduler = StepLR::new(0.1, 5, 0.5);
        let mut opt = MockOptimizer::new(0.1);

        for _ in 0..5 {
            scheduler.step(&mut opt);
        }
        // After 5 steps with step_size=5: gamma^1 = 0.5 => lr = 0.05
        assert!((scheduler.get_lr() - 0.05).abs() < 1e-6);
        assert!((opt.lr() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn step_lr_multiple_decays() {
        let mut scheduler = StepLR::new(0.1, 3, 0.5);
        let mut opt = MockOptimizer::new(0.1);

        for _ in 0..9 {
            scheduler.step(&mut opt);
        }
        // After 9 steps with step_size=3: gamma^3 = 0.125 => lr = 0.0125
        assert!((scheduler.get_lr() - 0.0125).abs() < 1e-6);
    }

    // -- CosineAnnealingLR tests --------------------------------------------

    #[test]
    fn cosine_annealing_starts_at_initial_lr() {
        let scheduler = CosineAnnealingLR::new(0.1, 100, 0.0);
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn cosine_annealing_reaches_eta_min_at_t_max() {
        let mut scheduler = CosineAnnealingLR::new(0.1, 50, 0.001);
        let mut opt = MockOptimizer::new(0.1);

        for _ in 0..50 {
            scheduler.step(&mut opt);
        }
        // At t_max, cos(pi) = -1, so lr = eta_min
        assert!((scheduler.get_lr() - 0.001).abs() < 1e-6);
    }

    #[test]
    fn cosine_annealing_midpoint() {
        let mut scheduler = CosineAnnealingLR::new(0.1, 100, 0.0);
        let mut opt = MockOptimizer::new(0.1);

        for _ in 0..50 {
            scheduler.step(&mut opt);
        }
        // At midpoint, cos(pi/2) = 0, so lr = 0.5 * initial_lr = 0.05
        assert!((scheduler.get_lr() - 0.05).abs() < 1e-4);
    }

    // -- WarmupCosineScheduler tests ----------------------------------------

    #[test]
    fn warmup_cosine_starts_at_zero() {
        let scheduler = WarmupCosineScheduler::new(0.1, 10, 100, 0.0);
        // At step 0, warmup fraction is 0 => lr = 0
        assert!((scheduler.get_lr()).abs() < 1e-6);
    }

    #[test]
    fn warmup_cosine_linear_phase() {
        let mut scheduler = WarmupCosineScheduler::new(0.1, 10, 100, 0.0);
        let mut opt = MockOptimizer::new(0.0);

        for _ in 0..5 {
            scheduler.step(&mut opt);
        }
        // At step 5 of 10 warmup steps: lr = 0.1 * 5/10 = 0.05
        assert!((scheduler.get_lr() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn warmup_cosine_reaches_peak_at_warmup_end() {
        let mut scheduler = WarmupCosineScheduler::new(0.1, 10, 100, 0.0);
        let mut opt = MockOptimizer::new(0.0);

        for _ in 0..10 {
            scheduler.step(&mut opt);
        }
        // At step 10 (start of decay phase), progress=0, cos(0)=1 => lr = initial_lr
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn warmup_cosine_decays_to_eta_min() {
        let mut scheduler = WarmupCosineScheduler::new(0.1, 10, 100, 0.001);
        let mut opt = MockOptimizer::new(0.0);

        for _ in 0..100 {
            scheduler.step(&mut opt);
        }
        // At total_steps, cos(pi)=-1 => lr = eta_min
        assert!((scheduler.get_lr() - 0.001).abs() < 1e-6);
    }
}
