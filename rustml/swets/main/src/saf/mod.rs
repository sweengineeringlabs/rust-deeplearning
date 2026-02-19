// SAF (Simple API Facade) — re-exports for convenient access

// API traits and types
pub use crate::api::error::{SwetsError, SwetsResult};
pub use crate::api::layer::Layer;
pub use crate::api::loss::Loss;
pub use crate::api::optim::{LRScheduler, Optimizer};
pub use crate::api::pool;
pub use crate::api::tape;
pub use crate::api::tensor::{Tensor, TensorId};

// Loss functions
pub use crate::core::loss::cross_entropy::CrossEntropyLoss;
pub use crate::core::loss::huber::HuberLoss;
pub use crate::core::loss::mae::MAELoss;
pub use crate::core::loss::mse::MSELoss;
pub use crate::core::loss::quantile::QuantileLoss;

// Neural network layers
pub use crate::core::nn::activations::{GELU, ReLU, SiLU, Sigmoid, Tanh};
pub use crate::core::nn::batch_norm::BatchNorm1d;
pub use crate::core::nn::conv1d::Conv1d;
pub use crate::core::nn::dropout::Dropout;
pub use crate::core::nn::layer_norm::LayerNorm;
pub use crate::core::nn::linear::Linear;
pub use crate::core::nn::lstm::LSTM;
pub use crate::core::nn::sequential::Sequential;

// Models
pub use crate::core::models::lstm_forecast::LSTMForecast;
pub use crate::core::models::nbeats::NBeats;
pub use crate::core::models::tcn::TCN;
pub use crate::core::models::transformer::TimeSeriesTransformer;

// Optimizers
pub use crate::core::optim::adam::Adam;
pub use crate::core::optim::adamw::AdamW;
pub use crate::core::optim::grad_clip::{clip_grad_norm, clip_grad_value};
pub use crate::core::optim::sgd::SGD;

// LR Schedulers
pub use crate::core::optim::schedulers::{CosineAnnealingLR, StepLR, WarmupCosineScheduler};

// Training
pub use crate::core::training::metrics::Metrics;
pub use crate::core::training::summary::model_summary;
pub use crate::core::training::trainer::Trainer;

// Serialization
pub use crate::core::serde::{save_checkpoint, load_checkpoint, Checkpoint};
