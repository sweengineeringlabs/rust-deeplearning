// SAF (Simple API Facade) — re-exports for convenient access
pub use crate::api::error::{SwetsError, SwetsResult};
pub use crate::api::layer::Layer;
pub use crate::api::loss::Loss;
pub use crate::api::optim::Optimizer;
pub use crate::api::tape;
pub use crate::api::tensor::{Tensor, TensorId};
pub use crate::core::loss::mse::MSELoss;
pub use crate::core::nn::activations::ReLU;
pub use crate::core::nn::linear::Linear;
pub use crate::core::optim::sgd::SGD;
pub use crate::core::training::metrics::Metrics;
pub use crate::core::training::trainer::Trainer;
