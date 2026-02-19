pub struct Metrics {
    sum_squared_error: f64,
    sum_absolute_error: f64,
    count: usize,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            sum_squared_error: 0.0,
            sum_absolute_error: 0.0,
            count: 0,
        }
    }

    pub fn update(&mut self, predictions: &[f32], targets: &[f32]) {
        assert_eq!(predictions.len(), targets.len());
        for (p, t) in predictions.iter().zip(targets.iter()) {
            let diff = (*p - *t) as f64;
            self.sum_squared_error += diff * diff;
            self.sum_absolute_error += diff.abs();
            self.count += 1;
        }
    }

    pub fn mse(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum_squared_error / self.count as f64
    }

    pub fn mae(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum_absolute_error / self.count as f64
    }

    pub fn rmse(&self) -> f64 {
        self.mse().sqrt()
    }

    pub fn reset(&mut self) {
        self.sum_squared_error = 0.0;
        self.sum_absolute_error = 0.0;
        self.count = 0;
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}
