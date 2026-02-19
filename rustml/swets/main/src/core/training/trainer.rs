use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::loss::Loss;
use crate::api::optim::{LRScheduler, Optimizer};
use crate::api::tape;
use crate::api::tensor::Tensor;
use crate::core::serde::save_checkpoint;

pub struct Trainer<M, O, L> {
    pub model: M,
    pub optimizer: O,
    pub loss_fn: L,
    grad_clip_norm: Option<f32>,
    patience: Option<usize>,
    best_val_loss: f32,
    epochs_without_improvement: usize,
    scheduler: Option<Box<dyn LRScheduler>>,
    checkpoint_dir: Option<String>,
}

impl<M: Layer, O: Optimizer, L: Loss> Trainer<M, O, L> {
    pub fn new(model: M, optimizer: O, loss_fn: L) -> Self {
        Self {
            model,
            optimizer,
            loss_fn,
            grad_clip_norm: None,
            patience: None,
            best_val_loss: f32::INFINITY,
            epochs_without_improvement: 0,
            scheduler: None,
            checkpoint_dir: None,
        }
    }

    /// Configure gradient clipping by max norm (FR-907).
    pub fn with_grad_clip(mut self, max_norm: f32) -> Self {
        self.grad_clip_norm = Some(max_norm);
        self
    }

    /// Configure early stopping with the given patience (FR-907).
    /// Training will stop if validation loss does not improve for `patience`
    /// consecutive epochs.
    pub fn with_early_stopping(mut self, patience: usize) -> Self {
        self.patience = Some(patience);
        self
    }

    /// Attach an LR scheduler that gets stepped each epoch (FR-907).
    pub fn with_scheduler(mut self, scheduler: Box<dyn LRScheduler>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Set the directory for best-model checkpoints (FR-907).
    ///
    /// When a checkpoint directory is configured the trainer will automatically
    /// save the model whenever validation loss improves (FR-906).
    pub fn with_checkpoint_dir(mut self, path: impl Into<String>) -> Self {
        self.checkpoint_dir = Some(path.into());
        self
    }

    /// Train one epoch over the provided batches.
    /// Each batch is (input, target).
    /// Returns the average loss over the epoch.
    pub fn train_epoch(&mut self, batches: &[(Tensor, Tensor)]) -> SwetsResult<f32> {
        let mut total_loss = 0.0;

        for (input, target) in batches {
            tape::clear_tape();

            let output = self.model.forward(input)?;
            let loss = self.loss_fn.forward(&output, target)?;
            total_loss += loss.to_vec()[0];

            tape::backward(&loss);

            // Gradient clipping (inline implementation)
            if let Some(max_norm) = self.grad_clip_norm {
                self.clip_gradients(max_norm);
            }

            let mut params: Vec<&mut Tensor> = self.model.parameters_mut();
            let mut param_refs: Vec<&mut Tensor> =
                params.iter_mut().map(|p| &mut **p).collect();
            self.optimizer.step(&mut param_refs)?;
        }

        Ok(total_loss / batches.len() as f32)
    }

    /// Validate on the provided batches (no gradient computation).
    /// Returns the average loss.
    pub fn validate(&mut self, batches: &[(Tensor, Tensor)]) -> SwetsResult<f32> {
        let mut total_loss = 0.0;

        tape::no_grad(|| {
            for (input, target) in batches {
                let output = self.model.forward(input).expect("validate forward");
                let loss = self.loss_fn.forward(&output, target).expect("validate loss");
                total_loss += loss.to_vec()[0];
            }
        });

        Ok(total_loss / batches.len() as f32)
    }

    /// Run a full training loop over multiple epochs (FR-903).
    ///
    /// Returns a `Vec<(train_loss, val_loss)>` for each completed epoch.
    /// If early stopping is configured and triggered, training ends early and
    /// the returned vector will contain fewer than `epochs` entries.
    ///
    /// After each epoch the following steps occur in order:
    /// 1. The LR scheduler (if configured) is stepped (FR-907).
    /// 2. If validation loss improved, a best-model checkpoint is saved to
    ///    `checkpoint_dir` (FR-906).
    /// 3. Early stopping is evaluated (FR-905). Because checkpointing happens
    ///    before the early-stopping check, the best model is guaranteed to have
    ///    been persisted when training halts early.
    pub fn fit(
        &mut self,
        train_batches: &[(Tensor, Tensor)],
        val_batches: &[(Tensor, Tensor)],
        epochs: usize,
    ) -> SwetsResult<Vec<(f32, f32)>> {
        let mut history: Vec<(f32, f32)> = Vec::with_capacity(epochs);

        for epoch in 1..=epochs {
            let train_loss = self.train_epoch(train_batches)?;
            let val_loss = self.validate(val_batches)?;

            log::info!(
                "Epoch {}/{}: train_loss={:.6}, val_loss={:.6}",
                epoch,
                epochs,
                train_loss,
                val_loss,
            );

            history.push((train_loss, val_loss));

            // FR-907: Step the LR scheduler after each epoch.
            if let Some(ref mut scheduler) = self.scheduler {
                scheduler.step(&mut self.optimizer);
                log::debug!(
                    "Epoch {}: LR updated to {:.6}",
                    epoch,
                    scheduler.get_lr(),
                );
            }

            // Track whether this epoch improved on the best validation loss.
            let is_improvement = val_loss < self.best_val_loss;

            // FR-906: Save best-model checkpoint when validation loss improves.
            if is_improvement {
                self.best_val_loss = val_loss;

                if let Some(ref dir) = self.checkpoint_dir {
                    std::fs::create_dir_all(dir).map_err(|e| {
                        crate::api::error::SwetsError::TrainingError(format!(
                            "create checkpoint dir '{}': {}",
                            dir, e
                        ))
                    })?;
                    let path = format!("{}/best_model.bin", dir);
                    save_checkpoint(&self.model, &path, epoch, val_loss)?;
                    log::info!(
                        "Epoch {}: saved best checkpoint (val_loss={:.6}) to {}",
                        epoch,
                        val_loss,
                        path,
                    );
                }
            }

            // FR-905: Early stopping check (must happen after checkpointing so
            // the best model has already been saved).
            if self.should_stop(is_improvement) {
                log::info!(
                    "Early stopping triggered at epoch {} (patience={}, best_val_loss={:.6})",
                    epoch,
                    self.patience.unwrap_or(0),
                    self.best_val_loss,
                );
                break;
            }
        }

        Ok(history)
    }

    /// Run a forward pass in no-grad mode and return the prediction (FR-904).
    pub fn predict(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let output = tape::no_grad(|| self.model.forward(input));
        output
    }

    /// Clip parameter gradients by global norm.
    ///
    /// Computes the total L2 norm across all parameter gradients. If the total
    /// norm exceeds `max_norm`, each gradient is scaled by `max_norm / total_norm`.
    fn clip_gradients(&self, max_norm: f32) {
        let params = self.model.parameters();
        // Compute the total squared norm across all parameter gradients.
        let mut total_norm_sq: f32 = 0.0;
        for param in &params {
            if let Some(grad) = tape::grad(param) {
                let grad_data = grad.to_vec();
                let sq_sum: f32 = grad_data.iter().map(|v| v * v).sum();
                total_norm_sq += sq_sum;
            }
        }

        let total_norm = total_norm_sq.sqrt();
        if total_norm > max_norm {
            let scale = max_norm / total_norm;
            for param in &params {
                if let Some(grad) = tape::grad(param) {
                    let clipped = grad.mul_scalar_raw(scale);
                    tape::set_grad(param, clipped);
                }
            }
        }
    }

    /// Check whether early stopping should be triggered.
    ///
    /// `improved` indicates whether the current epoch's validation loss was
    /// a new best (the caller is responsible for updating `best_val_loss`).
    /// Returns `true` if the patience threshold has been reached.
    fn should_stop(&mut self, improved: bool) -> bool {
        let patience = match self.patience {
            Some(p) => p,
            None => return false,
        };

        if improved {
            self.epochs_without_improvement = 0;
        } else {
            self.epochs_without_improvement += 1;
        }

        self.epochs_without_improvement >= patience
    }
}
