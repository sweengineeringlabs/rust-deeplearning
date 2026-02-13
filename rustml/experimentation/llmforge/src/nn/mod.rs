use crate::core::tensor::{Tensor, DType};
use crate::error::{LLMForgeError, Result};
use rand_distr::{Distribution, Normal};
use rand::thread_rng;

pub trait Layer {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
}

/// Trait for layers that support parameter freezing (for fine-tuning).
pub trait Freezable {
    fn is_frozen(&self) -> bool;
    fn freeze(&mut self);
    fn unfreeze(&mut self);
}

pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub frozen: bool,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Init weights with Xavier/Glorot
        let mut rng = thread_rng();
        let std = (2.0 / (in_features + out_features) as f64).sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        let mut w_data = Vec::with_capacity(out_features * in_features * 4);
        for _ in 0..(out_features * in_features) {
             let val = normal.sample(&mut rng) as f32;
             w_data.extend_from_slice(&val.to_ne_bytes());
        }

        // Shape: [Out, In] for standard Wx + b
        let weight = Tensor::new(w_data, vec![out_features, in_features], DType::F32);

        let bias_tensor = if bias {
            Some(Tensor::zeros(&[out_features]))
        } else {
            None
        };

        Self {
            weight,
            bias: bias_tensor,
            frozen: false,
        }
    }

    /// Construct a Linear layer from pre-loaded weight and optional bias tensors.
    pub fn from_weights(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias, frozen: false }
    }

    /// Quantize this layer's weight to Q8_0 format in-place.
    pub fn quantize(&mut self) -> Result<()> {
        self.weight = crate::quantization::quantize_tensor(&self.weight)?;
        Ok(())
    }

    /// Returns true if the weight is in quantized (Q8_0) format.
    pub fn is_quantized(&self) -> bool {
        self.weight.dtype == DType::Q8_0
    }
}

impl Freezable for Linear {
    fn is_frozen(&self) -> bool { self.frozen }
    fn freeze(&mut self) { self.frozen = true; }
    fn unfreeze(&mut self) { self.frozen = false; }
}

impl Layer for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Input: [..., In]
        // Weight: [Out, In]
        // Output: [..., Out]
        let output = if self.is_quantized() {
            crate::quantization::quantized_matmul(input, &self.weight)?
        } else {
            let w_t = self.weight.transpose(0, 1)?;
            input.matmul(&w_t)?
        };

        if let Some(b) = &self.bias {
             output.add(b)
        } else {
             Ok(output)
        }
    }
}

pub struct Embedding {
    pub weight: Tensor,
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub frozen: bool,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
         let mut rng = thread_rng();
         let normal = Normal::new(0.0, 1.0).unwrap();

         let mut data = Vec::with_capacity(num_embeddings * embedding_dim * 4);
         for _ in 0..(num_embeddings * embedding_dim) {
             let val = normal.sample(&mut rng) as f32;
             data.extend_from_slice(&val.to_ne_bytes());
         }

         let weight = Tensor::new(data, vec![num_embeddings, embedding_dim], DType::F32);

         Self {
             weight,
             num_embeddings,
             embedding_dim,
             frozen: false,
         }
    }

    /// Construct an Embedding layer from a pre-loaded weight tensor.
    /// Derives num_embeddings and embedding_dim from weight.shape() [vocab, dim].
    pub fn from_weights(weight: Tensor) -> Self {
        let num_embeddings = weight.shape()[0];
        let embedding_dim = weight.shape()[1];
        Self { weight, num_embeddings, embedding_dim, frozen: false }
    }

    pub fn forward(&self, indices: &Tensor) -> Result<Tensor> {
        let indices_data = indices.as_slice_f32()?;
        let weights_data = self.weight.as_slice_f32()?;
        let d_model = self.embedding_dim;

        let batch_size = indices.shape()[0];
        let seq_len = indices.shape()[1];

        let mut out_data = Vec::with_capacity(indices_data.len() * d_model);

        for &idx in indices_data {
            let idx = idx as usize;
            if idx >= self.num_embeddings {
                 return Err(LLMForgeError::IndexOutOfBounds {
                    index: idx,
                    dim: 0,
                    size: self.num_embeddings,
                });
            }

            let start = idx * d_model;
            let end = start + d_model;
            out_data.extend_from_slice(&weights_data[start..end]);
        }

        let out_bytes = crate::core::tensor::f32_vec_to_bytes(out_data);

        Ok(Tensor::new(out_bytes, vec![batch_size, seq_len, d_model], DType::F32))
    }
}

impl Freezable for Embedding {
    fn is_frozen(&self) -> bool { self.frozen }
    fn freeze(&mut self) { self.frozen = true; }
    fn unfreeze(&mut self) { self.frozen = false; }
}

pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f32,
    pub normalized_shape: Vec<usize>,
    pub frozen: bool,
}

impl LayerNorm {
    pub fn new(normalized_shape: Vec<usize>, eps: f32) -> Self {
        let size: usize = normalized_shape.iter().product();
        // Gamma (weight) init to 1
        let mut w_data = Vec::with_capacity(size * 4);
        for _ in 0..size {
            w_data.extend_from_slice(&1.0f32.to_ne_bytes());
        }
        let weight = Tensor::new(w_data, normalized_shape.clone(), DType::F32);

        // Beta (bias) init to 0
        let bias = Tensor::zeros(&normalized_shape);

        Self {
            weight,
            bias,
            eps,
            normalized_shape,
            frozen: false,
        }
    }

    /// Construct a LayerNorm from pre-loaded weight and bias tensors.
    pub fn from_weights(weight: Tensor, bias: Tensor, eps: f32) -> Self {
        let normalized_shape = weight.shape().to_vec();
        Self { weight, bias, eps, normalized_shape, frozen: false }
    }

    /// Construct a LayerNorm from weight only (zero bias).
    /// Used for Llama RMSNorm-style layers that have no learned bias.
    pub fn from_weight_only(weight: Tensor, eps: f32) -> Self {
        let normalized_shape = weight.shape().to_vec();
        let bias = Tensor::zeros(&normalized_shape);
        Self { weight, bias, eps, normalized_shape, frozen: false }
    }
}

impl Freezable for LayerNorm {
    fn is_frozen(&self) -> bool { self.frozen }
    fn freeze(&mut self) { self.frozen = true; }
    fn unfreeze(&mut self) { self.frozen = false; }
}

impl Layer for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.layer_norm(&self.weight, &self.bias, self.eps)
    }
}
