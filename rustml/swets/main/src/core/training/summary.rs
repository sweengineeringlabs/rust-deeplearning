use crate::api::layer::Layer;

/// Produce a human-readable summary of a model's parameter budget (FR-309).
///
/// The output includes:
/// - Total parameter count
/// - Number of trainable parameters (`requires_grad == true`)
/// - Estimated memory usage (`param_count * 4` bytes for f32)
pub fn model_summary(model: &dyn Layer) -> String {
    let params = model.parameters();

    let total_params: usize = params.iter().map(|p| p.numel()).sum();
    let trainable_params: usize = params
        .iter()
        .filter(|p| p.requires_grad())
        .map(|p| p.numel())
        .sum();
    let memory_bytes = total_params * std::mem::size_of::<f32>();

    let mut out = String::new();
    out.push_str("===== Model Summary =====\n");
    out.push_str(&format!("Total parameters:     {total_params}\n"));
    out.push_str(&format!("Trainable parameters: {trainable_params}\n"));
    out.push_str(&format!("Memory estimate:      {} bytes", memory_bytes));

    if memory_bytes >= 1024 * 1024 {
        let mb = memory_bytes as f64 / (1024.0 * 1024.0);
        out.push_str(&format!(" ({mb:.2} MB)"));
    } else if memory_bytes >= 1024 {
        let kb = memory_bytes as f64 / 1024.0;
        out.push_str(&format!(" ({kb:.2} KB)"));
    }

    out.push('\n');
    out.push_str("=========================");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::nn::linear::Linear;
    use crate::core::nn::sequential::Sequential;

    #[test]
    fn summary_single_linear() {
        let layer = Linear::new(4, 3);
        let summary = model_summary(&layer);

        // Linear(4, 3): weight [3,4] = 12 params + bias [3] = 3 params => 15 total
        assert!(summary.contains("Total parameters:     15"));
        assert!(summary.contains("Trainable parameters: 15"));
        // 15 * 4 = 60 bytes
        assert!(summary.contains("60 bytes"));
    }

    #[test]
    fn summary_sequential() {
        let model = Sequential::new(vec![
            Box::new(Linear::new(10, 5)),
            Box::new(Linear::new(5, 2)),
        ]);
        let summary = model_summary(&model);

        // Linear(10,5): 50 + 5 = 55 params
        // Linear(5,2):  10 + 2 = 12 params
        // Total: 67 params, 67*4 = 268 bytes
        assert!(summary.contains("Total parameters:     67"));
        assert!(summary.contains("Trainable parameters: 67"));
        assert!(summary.contains("268 bytes"));
    }
}
