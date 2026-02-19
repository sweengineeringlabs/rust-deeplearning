use crate::api::tape;
use crate::api::tensor::Tensor;

/// Clip gradients by global L2 norm (FR-404).
///
/// Computes the total L2 norm across all parameter gradients currently stored
/// on the tape.  If the total norm exceeds `max_norm`, every gradient is
/// scaled down by the factor `max_norm / total_norm` and written back to the
/// tape via `tape::set_grad`.
///
/// Returns the *original* (unclipped) total norm so callers can log it.
///
/// # Usage
///
/// ```ignore
/// // After backward, before optimizer.step:
/// let total_norm = clip_grad_norm(&param_refs, 1.0);
/// optimizer.step(&mut params)?;
/// ```
pub fn clip_grad_norm(params: &[&Tensor], max_norm: f32) -> f32 {
    // 1. Collect all gradients and compute total squared norm.
    let mut total_norm_sq: f64 = 0.0;
    let mut grads: Vec<Option<Tensor>> = Vec::with_capacity(params.len());

    for param in params {
        let g = tape::grad(param);
        if let Some(ref grad) = g {
            let grad_data = grad.to_vec();
            let sq_sum: f64 = grad_data.iter().map(|&x| (x as f64) * (x as f64)).sum();
            total_norm_sq += sq_sum;
        }
        grads.push(g);
    }

    let total_norm = (total_norm_sq as f32).sqrt();

    // 2. If total_norm > max_norm, scale all grads by max_norm / total_norm.
    if total_norm > max_norm {
        let clip_coef = max_norm / total_norm;
        for (param, grad_opt) in params.iter().zip(grads.into_iter()) {
            if let Some(grad) = grad_opt {
                let clipped = grad.mul_scalar_raw(clip_coef);
                tape::set_grad(param, clipped);
            }
        }
    }

    total_norm
}

/// Clip gradients by value (FR-405).
///
/// Clamps every element of every parameter gradient to the range
/// `[-clip_value, clip_value]` and writes the result back to the tape.
///
/// # Usage
///
/// ```ignore
/// // After backward, before optimizer.step:
/// clip_grad_value(&param_refs, 0.5);
/// optimizer.step(&mut params)?;
/// ```
pub fn clip_grad_value(params: &[&Tensor], clip_value: f32) {
    let clip_value = clip_value.abs(); // ensure positive

    for param in params {
        if let Some(grad) = tape::grad(param) {
            let data = grad.to_vec();
            let clamped: Vec<f32> = data
                .into_iter()
                .map(|x| x.clamp(-clip_value, clip_value))
                .collect();
            let shape = grad.shape().to_vec();
            let clipped =
                Tensor::from_vec(clamped, shape).expect("clip_grad_value: shape mismatch");
            tape::set_grad(param, clipped);
        }
    }
}
