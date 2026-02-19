use crate::api::error::SwetsResult;
use crate::api::layer::Layer;
use crate::api::tensor::Tensor;

/// Sequential container that chains layers in order (FR-301).
///
/// Forward pass feeds each layer's output as the next layer's input.
/// Parameters are the union of all contained layers' parameters.
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { layers }
    }

    /// Returns the number of layers in the container.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns true if the container has no layers.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Layer for Sequential {
    fn forward(&mut self, input: &Tensor) -> SwetsResult<Tensor> {
        let mut x = input.clone();
        for layer in &mut self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.layers
            .iter_mut()
            .flat_map(|l| l.parameters_mut())
            .collect()
    }
}
