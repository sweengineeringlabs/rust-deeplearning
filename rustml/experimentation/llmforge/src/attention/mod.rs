use crate::core::tensor::Tensor;
use crate::error::{LLMForgeError, Result};
use crate::nn::{Linear, Layer};

pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    d_model: usize,

    // Projections
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize, bias: bool) -> Result<Self> {
        if d_model % num_heads != 0 {
            return Err(LLMForgeError::InvalidConfig(
                format!("d_model ({}) must be divisible by num_heads ({})", d_model, num_heads)
            ));
        }
        let head_dim = d_model / num_heads;

        Ok(Self {
            num_heads,
            head_dim,
            d_model,
            q_proj: Linear::new(d_model, d_model, bias),
            k_proj: Linear::new(d_model, d_model, bias),
            v_proj: Linear::new(d_model, d_model, bias),
            out_proj: Linear::new(d_model, d_model, bias),
        })
    }

    /// Construct from pre-loaded projection layers.
    pub fn from_weights(d_model: usize, num_heads: usize, q_proj: Linear, k_proj: Linear, v_proj: Linear, out_proj: Linear) -> Result<Self> {
        if d_model % num_heads != 0 {
            return Err(LLMForgeError::InvalidConfig(
                format!("d_model ({}) must be divisible by num_heads ({})", d_model, num_heads)
            ));
        }
        let head_dim = d_model / num_heads;
        Ok(Self { num_heads, head_dim, d_model, q_proj, k_proj, v_proj, out_proj })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];

        // Project inputs [Batch, Seq, D_model]
        let q = self.q_proj.forward(input)?;
        let k = self.k_proj.forward(input)?;
        let v = self.v_proj.forward(input)?;

        // Split heads and transpose
        // shape: [Batch, Seq, Heads, Head_Dim]
        let q = q.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
                 .transpose(1, 2)?; // [Batch, Heads, Seq, Head_Dim]

        let k = k.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
                 .transpose(1, 2)?;

        let v = v.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?
                 .transpose(1, 2)?;

        // Scaled Dot-Product Attention
        let k_t = k.transpose(2, 3)?;

        let scores = q.batched_matmul(&k_t)?;

        let scale = (self.head_dim as f32).sqrt();
        let scores = scores.div_scalar(scale)?;

        let attn = scores.softmax(-1)?;

        // Output = attn * V
        let context = attn.batched_matmul(&v)?;

        // Reassemble
        let context = context.transpose(1, 2)?
                             .reshape(&[batch_size, seq_len, self.d_model])?;

        self.out_proj.forward(&context)
    }

    pub fn forward_with_cache(&self, input: &Tensor, cache: &mut KVCache, layer_idx: usize) -> Result<Tensor> {
        if cache.head_dim() != self.head_dim {
            return Err(LLMForgeError::InvalidConfig(
                format!("KVCache head_dim ({}) does not match attention head_dim ({})", cache.head_dim(), self.head_dim)
            ));
        }

        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];

        // Projects [Batch, Seq, D_Model] -> [Batch, Seq, D_Model]
        let q = self.q_proj.forward(input)?;
        let k = self.k_proj.forward(input)?;
        let v = self.v_proj.forward(input)?;

        // Reshape & Transpose to [Batch, Heads, Seq, Head_Dim]
        let target_shape = vec![batch_size, seq_len, self.num_heads, self.head_dim];

        let q = q.reshape(&target_shape)?.transpose(1, 2)?;
        let k = k.reshape(&target_shape)?.transpose(1, 2)?;
        let v = v.reshape(&target_shape)?.transpose(1, 2)?;

        // Update Cache
        cache.update(layer_idx, k.clone(), v.clone())?;

        // Get full history: 0 .. current_len + seq_len
        let total_len = cache.current_len + seq_len;
        let (k_full, v_full) = cache.get_view(layer_idx, total_len)?;

        // Attention
        let k_t = k_full.transpose(2, 3)?;
        let scores = q.batched_matmul(&k_t)?;

        let scale = (self.head_dim as f32).sqrt();
        let scores = scores.div_scalar(scale)?;
        let attn = scores.softmax(-1)?;

        let context = attn.batched_matmul(&v_full)?;

        let context = context.transpose(1, 2)?
                             .reshape(&[batch_size, seq_len, self.d_model])?;

        self.out_proj.forward(&context)
    }
}

pub struct KVCache {
    past_keys: Vec<Tensor>,
    past_values: Vec<Tensor>,
    max_seq_len: usize,
    pub current_len: usize,
    head_dim: usize,
    num_kv_heads: usize,
}

impl KVCache {
    pub fn new(num_layers: usize, max_seq_len: usize, head_dim: usize, num_kv_heads: usize) -> Self {
        let key_shape = vec![1, num_kv_heads, max_seq_len, head_dim];
        let val_shape = vec![1, num_kv_heads, max_seq_len, head_dim];

        let past_keys = (0..num_layers)
            .map(|_| Tensor::zeros(&key_shape))
            .collect();

        let past_values = (0..num_layers)
            .map(|_| Tensor::zeros(&val_shape))
            .collect();

        Self {
            past_keys,
            past_values,
            max_seq_len,
            current_len: 0,
            head_dim,
            num_kv_heads,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    // Returns view of `0..len`
    pub fn get_view(&self, layer_idx: usize, len: usize) -> Result<(Tensor, Tensor)> {
        let k = self.past_keys[layer_idx].slice_sequence(0, len)?;
        let v = self.past_values[layer_idx].slice_sequence(0, len)?;
        Ok((k, v))
    }

    pub fn update(&mut self, layer_idx: usize, key: Tensor, value: Tensor) -> Result<()> {
        let seq_len = key.shape()[2];
        if self.current_len + seq_len > self.max_seq_len {
            return Err(LLMForgeError::SequenceLengthExceeded {
                max: self.max_seq_len,
                actual: self.current_len + seq_len,
            });
        }
        self.past_keys[layer_idx].slice_assign_sequence(self.current_len, &key)?;
        self.past_values[layer_idx].slice_assign_sequence(self.current_len, &value)?;
        Ok(())
    }

    pub fn advance(&mut self, step: usize) {
        self.current_len += step;
    }
}
