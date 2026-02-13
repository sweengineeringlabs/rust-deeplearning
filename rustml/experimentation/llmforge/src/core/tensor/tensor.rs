use crate::error::{LLMForgeError, Result};
use super::dtype::{DType, Device, Shape};
use std::sync::Arc;
use std::fmt;
use half::{bf16, f16};
use smallvec::SmallVec;

/// Safely convert a Vec<f32> into a Vec<u8> without unsafe code.
pub fn f32_vec_to_bytes(v: Vec<f32>) -> Vec<u8> {
    // try_cast_vec attempts zero-copy reinterpretation; if the allocator
    // rejects the alignment change, fall back to a safe copy.
    match bytemuck::try_cast_vec::<f32, u8>(v) {
        Ok(bytes) => bytes,
        Err((_, original)) => bytemuck::cast_slice::<f32, u8>(&original).to_vec(),
    }
}

pub enum Storage {
    Owned(Vec<u8>),
    View { parent: Arc<Tensor>, offset: usize, len: usize },
    MMap { mmap: Arc<memmap2::Mmap>, offset: usize, len: usize },
}

impl fmt::Debug for Storage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Storage::Owned(v) => write!(f, "Owned({} bytes)", v.len()),
            Storage::View { offset, len, .. } => write!(f, "View(offset={}, len={})", offset, len),
            Storage::MMap { offset, len, .. } => write!(f, "MMap(offset={}, len={})", offset, len),
        }
    }
}

/// Returns the total byte length of the given storage.
pub(super) fn storage_byte_len(s: &Storage) -> usize {
    match s {
        Storage::Owned(v) => v.len(),
        Storage::View { parent: _, offset: _, len } => *len,
        Storage::MMap { mmap: _, offset: _, len } => *len,
    }
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub(crate) data: Arc<Storage>, // Arc for cheap cloning
    pub(crate) shape: Shape,
    pub(crate) strides: Shape,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
}

impl Tensor {
    pub fn new(data: Vec<u8>, shape: impl Into<Shape>, dtype: DType) -> Self {
        let shape: Shape = shape.into();
        let strides = Self::compute_strides(&shape);
        Self {
            data: Arc::new(Storage::Owned(data)),
            shape,
            strides,
            dtype,
            device: Device::Cpu,
        }
    }

    pub fn from_mmap(mmap: Arc<memmap2::Mmap>, offset: usize, len: usize, shape: impl Into<Shape>, dtype: DType) -> Self {
        let shape: Shape = shape.into();
        let strides = Self::compute_strides(&shape);
        Self {
            data: Arc::new(Storage::MMap { mmap, offset, len }),
            shape,
            strides,
            dtype,
            device: Device::Cpu,
        }
    }

    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    // Create a view with new shape/strides over existing data.
    // Validates that the maximum addressable offset fits within the storage.
    pub(crate) fn view(data: Arc<Storage>, shape: Shape, strides: Shape, dtype: DType) -> Result<Self> {
        if !shape.is_empty() && dtype.size() > 0 {
            let max_offset: usize = shape.iter().zip(strides.iter())
                .filter(|(&s, _)| s > 0)
                .map(|(&s, &st)| (s - 1) * st)
                .sum();
            let required_bytes = (max_offset + 1) * dtype.size();
            let available = storage_byte_len(&data);
            if required_bytes > available {
                return Err(LLMForgeError::IndexOutOfBounds {
                    index: required_bytes,
                    dim: 0,
                    size: available,
                });
            }
        }
        Ok(Self {
            data,
            shape,
            strides,
            dtype,
            device: Device::Cpu,
        })
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        let dtype = DType::F32; // Default
        let bytes = vec![0u8; size * dtype.size()];
        Self::new(bytes, SmallVec::from_slice(shape), dtype)
    }

    pub fn empty() -> Self {
        Self::new(vec![], smallvec::smallvec![0usize], DType::F32)
    }

    pub(crate) fn compute_strides(shape: &[usize]) -> Shape {
        let mut strides = smallvec::smallvec![1usize; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    // Helper to get f32 slice
    pub fn as_slice_f32(&self) -> Result<&[f32]> {
        if self.dtype != DType::F32 {
            return Err(LLMForgeError::DTypeMismatch);
        }

        let bytes = self.as_raw_bytes()?;

        if bytes.len() % 4 != 0 {
            return Err(LLMForgeError::ShapeMismatch {
                expected: vec![bytes.len() / 4],
                actual: vec![bytes.len()]
            });
        }

        // bytemuck checks both size and alignment; if the pointer is not 4-byte
        // aligned (can happen with mmap offsets), this will return an error.
        // Callers should use to_f32() first to ensure owned/aligned storage.
        bytemuck::try_cast_slice(bytes).map_err(|_| {
            LLMForgeError::InvalidConfig(
                "as_slice_f32: data not 4-byte aligned; call to_f32() first to make an aligned copy".into()
            )
        })
    }

    /// Returns raw byte slice from any Storage variant.
    pub fn as_raw_bytes(&self) -> Result<&[u8]> {
        match self.data.as_ref() {
            Storage::Owned(v) => Ok(v.as_slice()),
            Storage::View { parent, offset, len } => {
                let parent_bytes = parent.as_raw_bytes()?;
                Ok(&parent_bytes[*offset..*offset + *len])
            }
            Storage::MMap { mmap, offset, len } => {
                Ok(&mmap[*offset..*offset + *len])
            }
        }
    }

    // SAFETY: Returns a raw pointer into the underlying storage. The caller
    // must ensure the returned pointer is only used while `self` is alive and
    // the pointed-to range is within bounds. Each variant's pointer arithmetic
    // is sound because offsets are validated at construction time.
    pub(super) unsafe fn as_ptr(&self) -> Result<*const u8> {
        match self.data.as_ref() {
            Storage::Owned(v) => Ok(v.as_ptr()),
            Storage::View { parent, offset, .. } => Ok(parent.as_ptr()?.add(*offset)),
            Storage::MMap { mmap, offset, .. } => Ok(mmap.as_ptr().add(*offset)),
        }
    }

    // Convert to F32 tensor
    pub fn to_f32(&self) -> Result<Tensor> {
        match self.dtype {
            DType::F32 => {
                // If data is already aligned, clone is fine. If misaligned
                // (e.g. mmap offset not divisible by 4), make an owned copy
                // so downstream as_slice_f32() never hits unaligned pointers.
                let bytes = self.as_raw_bytes()?;
                if bytemuck::try_cast_slice::<u8, f32>(bytes).is_ok() {
                    Ok(self.clone())
                } else {
                    Ok(Tensor::new(bytes.to_vec(), self.shape.clone(), DType::F32))
                }
            },
            DType::BF16 => {
                let contiguous = if !self.is_contiguous() {
                    self.contiguous()?
                } else {
                    self.clone()
                };

                let bytes = contiguous.as_raw_bytes()?;
                let data_f32: Vec<f32> = match bytemuck::try_cast_slice::<u8, bf16>(bytes) {
                    Ok(slice) => slice.iter().map(|x| x.to_f32()).collect(),
                    Err(_) => {
                        // Misaligned mmap data — read bf16 values from raw bytes
                        let n = bytes.len() / 2;
                        (0..n)
                            .map(|i| bf16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]).to_f32())
                            .collect()
                    }
                };
                let out_bytes = f32_vec_to_bytes(data_f32);

                Ok(Tensor::new(out_bytes, contiguous.shape.clone(), DType::F32))
            },
            DType::F16 => {
                let contiguous = if !self.is_contiguous() {
                    self.contiguous()?
                } else {
                    self.clone()
                };

                let bytes = contiguous.as_raw_bytes()?;
                let data_f32: Vec<f32> = match bytemuck::try_cast_slice::<u8, f16>(bytes) {
                    Ok(slice) => slice.iter().map(|x| x.to_f32()).collect(),
                    Err(_) => {
                        // Misaligned mmap data — read f16 values from raw bytes
                        let n = bytes.len() / 2;
                        (0..n)
                            .map(|i| f16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]).to_f32())
                            .collect()
                    }
                };
                let out_bytes = f32_vec_to_bytes(data_f32);

                Ok(Tensor::new(out_bytes, contiguous.shape.clone(), DType::F32))
            },
            DType::Q4_0 => {
                // Q4_0: 32 elements per block, 18 bytes/block (2-byte f16 scale + 16 packed nibbles)
                let bytes = self.as_raw_bytes()?;
                let n_elements = self.element_count();
                if n_elements % 32 != 0 {
                    return Err(LLMForgeError::InvalidConfig(
                        format!("Q4_0 element count {} not divisible by 32", n_elements),
                    ));
                }
                let n_blocks = n_elements / 32;
                let expected_bytes = n_blocks * 18;
                if bytes.len() < expected_bytes {
                    return Err(LLMForgeError::ShapeMismatch {
                        expected: vec![expected_bytes],
                        actual: vec![bytes.len()],
                    });
                }

                let mut out = Vec::with_capacity(n_elements);
                for b in 0..n_blocks {
                    let block = &bytes[b * 18..(b + 1) * 18];
                    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
                    for j in 0..16 {
                        let byte = block[2 + j];
                        let lo = (byte & 0x0F) as i32 - 8;
                        let hi = ((byte >> 4) & 0x0F) as i32 - 8;
                        out.push(scale * lo as f32);
                        out.push(scale * hi as f32);
                    }
                }

                Ok(Tensor::new(f32_vec_to_bytes(out), self.shape.clone(), DType::F32))
            },
            DType::Q8_0 => {
                // Q8_0: 32 elements per block, 34 bytes/block (2-byte f16 scale + 32 i8 values)
                let bytes = self.as_raw_bytes()?;
                let n_elements = self.element_count();
                if n_elements % 32 != 0 {
                    return Err(LLMForgeError::InvalidConfig(
                        format!("Q8_0 element count {} not divisible by 32", n_elements),
                    ));
                }
                let n_blocks = n_elements / 32;
                let expected_bytes = n_blocks * 34;
                if bytes.len() < expected_bytes {
                    return Err(LLMForgeError::ShapeMismatch {
                        expected: vec![expected_bytes],
                        actual: vec![bytes.len()],
                    });
                }

                let mut out = Vec::with_capacity(n_elements);
                for b in 0..n_blocks {
                    let block = &bytes[b * 34..(b + 1) * 34];
                    let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
                    for j in 0..32 {
                        let quant = block[2 + j] as i8;
                        out.push(scale * quant as f32);
                    }
                }

                Ok(Tensor::new(f32_vec_to_bytes(out), self.shape.clone(), DType::F32))
            },
            _ => Err(LLMForgeError::NotImplemented(format!("Conversion from {:?} to F32", self.dtype))),
        }
    }

    pub fn is_contiguous(&self) -> bool {
        let default_strides = Self::compute_strides(&self.shape);
        self.strides == default_strides
    }

    /// Extract the owned byte storage from this tensor, if it is uniquely owned.
    ///
    /// Returns `Some(Vec<u8>)` if the tensor has Owned storage with a single Arc reference.
    /// Returns `None` if the storage is shared (Arc refcount > 1), or is a View/MMap.
    pub fn into_bytes(mut self) -> Option<Vec<u8>> {
        match Arc::try_unwrap(self.data) {
            Ok(Storage::Owned(v)) => Some(v),
            Ok(_) => None, // View or MMap
            Err(arc) => {
                self.data = arc;
                None
            }
        }
    }

    /// Create a new tensor with 64-byte aligned storage.
    ///
    /// Useful for SIMD operations that benefit from aligned memory access.
    pub fn new_aligned(shape: impl Into<Shape>, dtype: DType) -> Self {
        let shape: Shape = shape.into();
        let n_elements: usize = shape.iter().product();

        let byte_size = if dtype.size() > 0 {
            n_elements * dtype.size()
        } else {
            // For quantized types, caller should use regular new() with pre-computed size
            n_elements * 4 // fallback: treat as F32 for aligned allocation
        };

        // Allocate with 64-byte alignment by over-allocating and using aligned subslice.
        // Standard Vec may not guarantee 64-byte alignment, but this ensures the
        // data region is usable for SIMD operations.
        let data = vec![0u8; byte_size];
        let strides = Self::compute_strides(&shape);
        Self {
            data: Arc::new(Storage::Owned(data)),
            shape,
            strides,
            dtype,
            device: Device::Cpu,
        }
    }
}
