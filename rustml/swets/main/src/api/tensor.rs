use rustml_core::{DType, Tensor as CoreTensor, TensorResult};
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

    pub fn div_scalar_raw(&self, scalar: f32) -> Tensor {
        Tensor::new(self.inner.div_scalar(scalar), false)
    }

    pub fn reshape_raw(&self, shape: &[usize]) -> TensorResult<Tensor> {
        let result = self.inner.reshape(shape)?;
        Ok(Tensor::new(result, false))
    }

    pub fn sum_raw(&self, dim: i64) -> TensorResult<Tensor> {
        let result = self.inner.sum(dim)?;
        Ok(Tensor::new(result, false))
    }

    /// Replace inner data while preserving TensorId (for optimizer in-place updates).
    pub fn update_data_from(&mut self, other: &Tensor) {
        self.inner = other.inner.clone();
    }
}
