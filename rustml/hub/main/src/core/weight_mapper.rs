//! Weight mapping from HuggingFace to RustML model architectures

use crate::api::error::{HubError, HubResult};
use rustml_core::Tensor;
use std::collections::HashMap;

/// Trait for mapping weights from HuggingFace format to RustML models
pub trait WeightMapper {
    /// Map weights from HuggingFace naming to RustML naming
    fn map_weights(&self, weights: HashMap<String, Tensor>) -> HubResult<HashMap<String, Tensor>>;

    /// Get the expected weight names for validation
    fn expected_weights(&self) -> Vec<&'static str>;
}

/// Weight mapper for GPT-2 models
///
/// Maps HuggingFace GPT-2 weight names to RustML names:
///
/// | HuggingFace Name                        | RustML Name                    |
/// |-----------------------------------------|--------------------------------|
/// | transformer.wte.weight                  | wte.weight                     |
/// | transformer.wpe.weight                  | wpe.weight                     |
/// | transformer.h.{i}.ln_1.weight           | blocks.{i}.ln_1.weight         |
/// | transformer.h.{i}.ln_1.bias             | blocks.{i}.ln_1.bias           |
/// | transformer.h.{i}.attn.c_attn.weight    | blocks.{i}.attn.c_attn.weight  |
/// | transformer.h.{i}.attn.c_attn.bias      | blocks.{i}.attn.c_attn.bias    |
/// | transformer.h.{i}.attn.c_proj.weight    | blocks.{i}.attn.c_proj.weight  |
/// | transformer.h.{i}.attn.c_proj.bias      | blocks.{i}.attn.c_proj.bias    |
/// | transformer.h.{i}.ln_2.weight           | blocks.{i}.ln_2.weight         |
/// | transformer.h.{i}.ln_2.bias             | blocks.{i}.ln_2.bias           |
/// | transformer.h.{i}.mlp.c_fc.weight       | blocks.{i}.mlp.c_fc.weight     |
/// | transformer.h.{i}.mlp.c_fc.bias         | blocks.{i}.mlp.c_fc.bias       |
/// | transformer.h.{i}.mlp.c_proj.weight     | blocks.{i}.mlp.c_proj.weight   |
/// | transformer.h.{i}.mlp.c_proj.bias       | blocks.{i}.mlp.c_proj.bias     |
/// | transformer.ln_f.weight                 | ln_f.weight                    |
/// | transformer.ln_f.bias                   | ln_f.bias                      |
#[derive(Debug, Default)]
pub struct Gpt2WeightMapper {
    /// Number of transformer layers
    pub n_layer: usize,
}

impl Gpt2WeightMapper {
    /// Create a new GPT-2 weight mapper
    pub fn new(n_layer: usize) -> Self {
        Self { n_layer }
    }

    /// Map a single weight name from HuggingFace to RustML format
    fn map_name(&self, hf_name: &str) -> Option<String> {
        // Remove "transformer." prefix if present
        let name = hf_name.strip_prefix("transformer.").unwrap_or(hf_name);

        // Handle different patterns
        if name.starts_with("h.") {
            // Block weights: h.{i}.* -> blocks.{i}.*
            let rest = name.strip_prefix("h.")?;
            Some(format!("blocks.{}", rest))
        } else {
            // Top-level weights: wte, wpe, ln_f
            Some(name.to_string())
        }
    }

    /// Get the number of layers from weights
    pub fn detect_n_layer(weights: &HashMap<String, Tensor>) -> usize {
        weights
            .keys()
            .filter_map(|k| {
                // Extract layer number from patterns like "transformer.h.11.ln_1.weight"
                let parts: Vec<&str> = k.split('.').collect();
                for (i, part) in parts.iter().enumerate() {
                    if *part == "h" && i + 1 < parts.len() {
                        return parts[i + 1].parse::<usize>().ok();
                    }
                }
                None
            })
            .max()
            .map(|n| n + 1)
            .unwrap_or(12)
    }
}

impl WeightMapper for Gpt2WeightMapper {
    fn map_weights(&self, weights: HashMap<String, Tensor>) -> HubResult<HashMap<String, Tensor>> {
        let mut mapped = HashMap::new();

        for (name, tensor) in weights {
            if let Some(new_name) = self.map_name(&name) {
                // GPT-2 uses transposed linear layers, we may need to transpose
                // HuggingFace GPT-2 stores weights as [in_features, out_features]
                // We expect [out_features, in_features]
                let tensor = if new_name.contains("c_attn.weight")
                    || new_name.contains("c_proj.weight")
                    || new_name.contains("c_fc.weight")
                {
                    // Transpose 2D weight matrices
                    if tensor.ndim() == 2 {
                        tensor.t().unwrap_or(tensor)
                    } else {
                        tensor
                    }
                } else {
                    tensor
                };

                mapped.insert(new_name, tensor);
            } else {
                // Keep original name for unmapped weights
                mapped.insert(name, tensor);
            }
        }

        Ok(mapped)
    }

    fn expected_weights(&self) -> Vec<&'static str> {
        vec![
            "wte.weight",
            "wpe.weight",
            "ln_f.weight",
            "ln_f.bias",
        ]
    }
}

/// Validate that all expected weights are present
pub fn validate_weights(
    weights: &HashMap<String, Tensor>,
    expected: &[&str],
) -> Result<(), HubError> {
    for name in expected {
        if !weights.contains_key(*name) {
            return Err(HubError::WeightLoadError(format!(
                "Missing expected weight: {}",
                name
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt2_name_mapping() {
        let mapper = Gpt2WeightMapper::new(12);

        assert_eq!(
            mapper.map_name("transformer.wte.weight"),
            Some("wte.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("transformer.h.0.ln_1.weight"),
            Some("blocks.0.ln_1.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("transformer.h.11.attn.c_attn.weight"),
            Some("blocks.11.attn.c_attn.weight".to_string())
        );
        assert_eq!(
            mapper.map_name("transformer.ln_f.weight"),
            Some("ln_f.weight".to_string())
        );
    }

    #[test]
    fn test_detect_n_layer() {
        let mut weights = HashMap::new();
        weights.insert(
            "transformer.h.11.ln_1.weight".to_string(),
            Tensor::zeros(vec![768]),
        );
        weights.insert(
            "transformer.h.5.ln_1.weight".to_string(),
            Tensor::zeros(vec![768]),
        );

        assert_eq!(Gpt2WeightMapper::detect_n_layer(&weights), 12);
    }
}
