use rustml_core::{DType, Tensor as CoreTensor, TensorError, TensorResult};
use crate::api::error::{SwetsError, SwetsResult};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub(crate) u64);

impl TensorId {
    fn next() -> Self {
        TensorId(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl std::fmt::Display for TensorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TensorId({})", self.0)
    }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub(crate) id: TensorId,
    pub(crate) inner: CoreTensor,
    pub(crate) requires_grad: bool,
}

impl Tensor {
    // --- Constructors ---

    pub fn new(inner: CoreTensor, requires_grad: bool) -> Self {
        Self {
            id: TensorId::next(),
            inner,
            requires_grad,
        }
    }

    pub fn zeros(shape: impl Into<rustml_core::Shape>) -> Self {
        Self::new(CoreTensor::zeros(shape), false)
    }

    pub fn ones(shape: impl Into<rustml_core::Shape>) -> Self {
        Self::new(CoreTensor::ones(shape), false)
    }

    pub fn randn(shape: impl Into<rustml_core::Shape>) -> Self {
        Self::new(CoreTensor::randn(shape), false)
    }

    pub fn from_vec(data: Vec<f32>, shape: impl Into<rustml_core::Shape>) -> TensorResult<Self> {
        Ok(Self::new(CoreTensor::from_vec(data, shape)?, false))
    }

    pub fn full(shape: impl Into<rustml_core::Shape>, value: f32) -> Self {
        Self::new(CoreTensor::full(shape, value), false)
    }

    // --- Accessors ---

    pub fn id(&self) -> TensorId {
        self.id
    }

    pub fn inner(&self) -> &CoreTensor {
        &self.inner
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    pub fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    pub fn numel(&self) -> usize {
        self.inner.numel()
    }

    pub fn dtype(&self) -> DType {
        self.inner.dtype()
    }

    pub fn data(&self) -> TensorResult<&[f32]> {
        self.inner.data()
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.inner.to_vec()
    }

    // --- Raw ops (no tape recording) ---

    pub fn matmul_raw(&self, other: &Tensor) -> TensorResult<Tensor> {
        let result = self.inner.matmul(&other.inner)?;
        Ok(Tensor::new(result, false))
    }

    pub fn add_raw(&self, other: &Tensor) -> TensorResult<Tensor> {
        let result = self.inner.add(&other.inner)?;
        Ok(Tensor::new(result, false))
    }

    pub fn sub_raw(&self, other: &Tensor) -> TensorResult<Tensor> {
        let result = self.inner.sub(&other.inner)?;
        Ok(Tensor::new(result, false))
    }

    pub fn mul_raw(&self, other: &Tensor) -> TensorResult<Tensor> {
        let result = self.inner.mul(&other.inner)?;
        Ok(Tensor::new(result, false))
    }

    pub fn mul_scalar_raw(&self, scalar: f32) -> Tensor {
        Tensor::new(self.inner.mul_scalar(scalar), false)
    }

    pub fn transpose_raw(&self, dim0: i64, dim1: i64) -> TensorResult<Tensor> {
        let result = self.inner.transpose(dim0, dim1)?;
        Ok(Tensor::new(result, false))
    }

    pub fn relu_raw(&self) -> Tensor {
        Tensor::new(self.inner.relu(), false)
    }

    pub fn neg_raw(&self) -> Tensor {
        Tensor::new(self.inner.neg(), false)
    }

    pub fn mean_all_raw(&self) -> f32 {
        self.inner.mean_all()
    }

    pub fn sum_all_raw(&self) -> f32 {
        self.inner.sum_all()
    }

    pub fn pow_raw(&self, exp: f32) -> Tensor {
        Tensor::new(self.inner.pow(exp), false)
    }

    pub fn div_raw(&self, other: &Tensor) -> TensorResult<Tensor> {
        let result = self.inner.div(&other.inner)?;
        Ok(Tensor::new(result, false))
    }

    pub fn div_scalar_raw(&self, scalar: f32) -> Tensor {
        Tensor::new(self.inner.div_scalar(scalar), false)
    }

    pub fn add_scalar_raw(&self, scalar: f32) -> Tensor {
        Tensor::new(self.inner.add_scalar(scalar), false)
    }

    pub fn sqrt_raw(&self) -> Tensor {
        Tensor::new(self.inner.sqrt(), false)
    }

    pub fn reshape_raw(&self, shape: &[usize]) -> TensorResult<Tensor> {
        let result = self.inner.reshape(shape)?;
        Ok(Tensor::new(result, false))
    }

    pub fn sum_raw(&self, dim: i64) -> TensorResult<Tensor> {
        let result = self.inner.sum(dim)?;
        Ok(Tensor::new(result, false))
    }

    // --- FR-106: Shape ops (no tape recording) ---

    /// Permute dimensions. Delegates to CoreTensor::permute (zero-copy stride reorder).
    pub fn permute_raw(&self, dims: &[usize]) -> SwetsResult<Tensor> {
        let result = self.inner.permute(dims).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// Remove a dimension of size 1. Delegates to CoreTensor::squeeze.
    pub fn squeeze_raw(&self, dim: i64) -> SwetsResult<Tensor> {
        let result = self.inner.squeeze(dim).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// Insert a dimension of size 1. Delegates to CoreTensor::unsqueeze.
    pub fn unsqueeze_raw(&self, dim: i64) -> SwetsResult<Tensor> {
        let result = self.inner.unsqueeze(dim).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// Flatten the tensor into a 1D tensor.
    /// Implemented via reshape to a single dimension equal to numel().
    pub fn flatten_raw(&self) -> SwetsResult<Tensor> {
        let n = self.numel();
        let result = self.inner.reshape(&[n]).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// View the tensor with a new shape (alias for reshape).
    /// Total number of elements must remain the same.
    pub fn view_raw(&self, shape: &[usize]) -> SwetsResult<Tensor> {
        let result = self.inner.reshape(shape).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    // --- FR-107: Indexing ops (no tape recording) ---

    /// Slice the tensor along a dimension from `start` to `end` (exclusive).
    /// Delegates to CoreTensor::slice.
    pub fn slice_raw(&self, dim: i64, start: usize, end: usize) -> SwetsResult<Tensor> {
        let result = self.inner.slice(dim, start, end).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// Select elements along `dim` at the positions specified by `indices`.
    /// Returns a tensor with the same number of dimensions where `dim` has size `indices.len()`.
    ///
    /// CoreTensor does not provide index_select natively, so this is implemented
    /// element-wise via to_vec()/from_vec().
    pub fn index_select_raw(&self, dim: i64, indices: &[usize]) -> SwetsResult<Tensor> {
        let shape = self.shape();
        let ndim = self.ndim();
        let dim_idx = normalize_dim_helper(dim, ndim)?;
        let dim_size = shape[dim_idx];

        // Validate indices
        for &idx in indices {
            if idx >= dim_size {
                return Err(SwetsError::TensorError(TensorError::IndexOutOfBounds {
                    dim: dim_idx,
                    index: idx,
                    size: dim_size,
                }));
            }
        }

        // Compute output shape
        let mut out_shape: Vec<usize> = shape.to_vec();
        out_shape[dim_idx] = indices.len();
        let out_numel: usize = out_shape.iter().product();

        let src_data = self.to_vec();
        let src_strides = compute_strides(&shape);
        let out_strides = compute_strides(&out_shape);

        let mut out_data = vec![0.0f32; out_numel];
        let mut out_indices = vec![0usize; ndim];

        for flat in 0..out_numel {
            // Convert flat index to multi-dimensional index in output
            let mut rem = flat;
            for d in 0..ndim {
                out_indices[d] = rem / out_strides[d];
                rem %= out_strides[d];
            }
            // Map the output index to source index
            let mut src_flat = 0;
            for d in 0..ndim {
                let idx = if d == dim_idx {
                    indices[out_indices[d]]
                } else {
                    out_indices[d]
                };
                src_flat += idx * src_strides[d];
            }
            out_data[flat] = src_data[src_flat];
        }

        let result = CoreTensor::from_vec(out_data, out_shape).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// Gather elements along `dim` using the index tensor.
    /// `index` must have the same number of dimensions as `self`.
    /// The output has the same shape as `index`.
    ///
    /// For a 3D tensor: `output[i][j][k] = self[i][index[i][j][k]][k]` when dim=1.
    ///
    /// CoreTensor does not provide gather natively, so this is implemented
    /// element-wise via to_vec()/from_vec().
    pub fn gather_raw(&self, dim: i64, index: &Tensor) -> SwetsResult<Tensor> {
        let shape = self.shape();
        let ndim = self.ndim();
        let dim_idx = normalize_dim_helper(dim, ndim)?;

        let idx_shape = index.shape();
        if idx_shape.len() != ndim {
            return Err(SwetsError::TensorError(TensorError::InvalidOperation(
                format!(
                    "gather: index must have same ndim as self ({}), got {}",
                    ndim,
                    idx_shape.len()
                ),
            )));
        }

        let src_data = self.to_vec();
        let idx_data = index.to_vec();
        let src_strides = compute_strides(shape);
        let idx_strides = compute_strides(idx_shape);
        let out_numel: usize = idx_shape.iter().product();

        let mut out_data = vec![0.0f32; out_numel];
        let mut multi_idx = vec![0usize; ndim];

        for flat in 0..out_numel {
            // Convert flat index to multi-dimensional index in index tensor
            let mut rem = flat;
            for d in 0..ndim {
                multi_idx[d] = rem / idx_strides[d];
                rem %= idx_strides[d];
            }
            // The gather index along dim
            let gather_idx = idx_data[flat] as usize;
            if gather_idx >= shape[dim_idx] {
                return Err(SwetsError::TensorError(TensorError::IndexOutOfBounds {
                    dim: dim_idx,
                    index: gather_idx,
                    size: shape[dim_idx],
                }));
            }
            // Compute source flat index
            let mut src_flat = 0;
            for d in 0..ndim {
                let idx = if d == dim_idx { gather_idx } else { multi_idx[d] };
                src_flat += idx * src_strides[d];
            }
            out_data[flat] = src_data[src_flat];
        }

        let result =
            CoreTensor::from_vec(out_data, idx_shape.to_vec()).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// Select elements where `mask` is non-zero. Returns a 1D tensor.
    /// `mask` must be broadcastable to `self`'s shape.
    ///
    /// CoreTensor does not provide masked_select natively, so this is implemented
    /// element-wise via to_vec()/from_vec().
    pub fn masked_select_raw(&self, mask: &Tensor) -> SwetsResult<Tensor> {
        let self_data = self.to_vec();

        // Broadcast mask to self's shape if needed
        let mask_data = if self.shape() == mask.shape() {
            mask.to_vec()
        } else {
            let target = rustml_core::Shape::new(self.shape().to_vec());
            let broadcast_mask = mask
                .inner
                .broadcast_to(&target)
                .map_err(SwetsError::TensorError)?;
            broadcast_mask.to_vec()
        };

        let selected: Vec<f32> = self_data
            .iter()
            .zip(mask_data.iter())
            .filter(|(_, m)| **m != 0.0)
            .map(|(v, _)| *v)
            .collect();
        let n = selected.len();
        let result = CoreTensor::from_vec(selected, vec![n]).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    // --- FR-108: Element-wise math ops (no tape recording) ---

    /// Element-wise exponential. Delegates to CoreTensor::exp.
    pub fn exp_raw(&self) -> Tensor {
        Tensor::new(self.inner.exp(), false)
    }

    /// Element-wise natural logarithm. Delegates to CoreTensor::log (which calls ln()).
    pub fn log_raw(&self) -> Tensor {
        Tensor::new(self.inner.log(), false)
    }

    /// Element-wise absolute value. Delegates to CoreTensor::abs.
    pub fn abs_raw(&self) -> Tensor {
        Tensor::new(self.inner.abs(), false)
    }

    // --- FR-109: Reduction ops (no tape recording) ---

    /// Mean along a dimension. Delegates to CoreTensor::mean.
    pub fn mean_raw(&self, dim: i64) -> SwetsResult<Tensor> {
        let result = self.inner.mean(dim).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// Max along a dimension. Returns (values, indices).
    /// Delegates to CoreTensor::max.
    pub fn max_raw(&self, dim: i64) -> SwetsResult<(Tensor, Tensor)> {
        let (values, indices) = self.inner.max(dim).map_err(SwetsError::TensorError)?;
        Ok((Tensor::new(values, false), Tensor::new(indices, false)))
    }

    /// Min along a dimension. Returns (values, indices).
    /// Delegates to CoreTensor::min.
    pub fn min_raw(&self, dim: i64) -> SwetsResult<(Tensor, Tensor)> {
        let (values, indices) = self.inner.min(dim).map_err(SwetsError::TensorError)?;
        Ok((Tensor::new(values, false), Tensor::new(indices, false)))
    }

    /// Variance along a dimension. Delegates to CoreTensor::var.
    pub fn var_raw(&self, dim: i64) -> SwetsResult<Tensor> {
        let result = self.inner.var(dim).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// Standard deviation along a dimension.
    /// Computed as sqrt(var(dim)).
    pub fn std_dev_raw(&self, dim: i64) -> SwetsResult<Tensor> {
        let var = self.inner.var(dim).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(var.sqrt(), false))
    }

    // --- FR-112: Joining ops (no tape recording) ---

    /// Concatenate tensors along a dimension. Delegates to CoreTensor::cat.
    pub fn concat_raw(tensors: &[&Tensor], dim: i64) -> SwetsResult<Tensor> {
        let core_tensors: Vec<&CoreTensor> = tensors.iter().map(|t| &t.inner).collect();
        let result = CoreTensor::cat(&core_tensors, dim).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// Stack tensors along a new dimension.
    /// All tensors must have the same shape. A new dimension of size `tensors.len()`
    /// is inserted at position `dim`.
    ///
    /// CoreTensor does not provide stack natively, so this is implemented by
    /// unsqueezing each tensor and then concatenating.
    pub fn stack_raw(tensors: &[&Tensor], dim: i64) -> SwetsResult<Tensor> {
        if tensors.is_empty() {
            return Err(SwetsError::TensorError(TensorError::EmptyTensor));
        }

        let first_shape = tensors[0].shape();
        for t in tensors.iter().skip(1) {
            if t.shape() != first_shape {
                return Err(SwetsError::TensorError(TensorError::ShapeMismatch {
                    expected: first_shape.to_vec(),
                    got: t.shape().to_vec(),
                }));
            }
        }

        // Unsqueeze each tensor at the target dim, then concatenate
        let unsqueezed: Vec<CoreTensor> = tensors
            .iter()
            .map(|t| t.inner.unsqueeze(dim))
            .collect::<Result<Vec<_>, _>>()
            .map_err(SwetsError::TensorError)?;
        let refs: Vec<&CoreTensor> = unsqueezed.iter().collect();
        let result = CoreTensor::cat(&refs, dim).map_err(SwetsError::TensorError)?;
        Ok(Tensor::new(result, false))
    }

    /// Split a tensor into chunks along a dimension.
    /// `split_size` is the size of each chunk (the last chunk may be smaller).
    ///
    /// CoreTensor does not provide split natively, so this is implemented via
    /// repeated slicing.
    pub fn split_raw(&self, split_size: usize, dim: i64) -> SwetsResult<Vec<Tensor>> {
        let shape = self.shape();
        let ndim = self.ndim();
        let dim_idx = normalize_dim_helper(dim, ndim)?;
        let dim_size = shape[dim_idx];

        if split_size == 0 {
            return Err(SwetsError::TensorError(TensorError::InvalidOperation(
                "split_size must be greater than 0".into(),
            )));
        }

        let mut chunks = Vec::new();
        let mut start = 0;
        while start < dim_size {
            let end = (start + split_size).min(dim_size);
            let chunk = self
                .inner
                .slice(dim, start, end)
                .map_err(SwetsError::TensorError)?;
            chunks.push(Tensor::new(chunk, false));
            start = end;
        }

        Ok(chunks)
    }

    /// Replace inner data while preserving TensorId (for optimizer in-place updates).
    pub fn update_data_from(&mut self, other: &Tensor) {
        self.inner = other.inner.clone();
    }
}

// --- Helper functions ---

/// Normalize a dimension index, converting negative indices to positive.
fn normalize_dim_helper(dim: i64, ndim: usize) -> SwetsResult<usize> {
    let ndim_i64 = ndim as i64;
    let normalized = if dim < 0 { dim + ndim_i64 } else { dim };
    if normalized >= 0 && normalized < ndim_i64 {
        Ok(normalized as usize)
    } else {
        Err(SwetsError::TensorError(TensorError::InvalidDimension {
            dim,
            ndim,
        }))
    }
}

/// Compute row-major strides from a shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
