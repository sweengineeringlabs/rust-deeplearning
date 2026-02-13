use crate::error::{LLMForgeError, Result};
use super::tensor::{Tensor, Storage, f32_vec_to_bytes};
use super::dtype::DType;
use smallvec::SmallVec;
use std::sync::Arc;

impl Tensor {
    // Reshape tensor view
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor> {
        let current_size: usize = self.shape.iter().product();
        let new_size: usize = shape.iter().product();

        if current_size != new_size {
            return Err(LLMForgeError::ShapeMismatch {
                expected: vec![current_size],
                actual: vec![new_size],
            });
        }

        if self.is_contiguous() {
             let new_strides = Self::compute_strides(shape);
             Self::view(self.data.clone(), SmallVec::from_slice(shape), new_strides, self.dtype)
        } else {
             let contiguous = self.contiguous()?;
             let new_strides = Self::compute_strides(shape);
             Self::view(contiguous.data, SmallVec::from_slice(shape), new_strides, self.dtype)
        }
    }

    // Transpose two dimensions
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Tensor> {
        let ndim = self.shape.len();
        if dim0 >= ndim || dim1 >= ndim {
             return Err(LLMForgeError::IndexOutOfBounds {
                index: std::cmp::max(dim0, dim1),
                dim: 0,
                size: ndim,
            });
        }

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);

        Self::view(self.data.clone(), new_shape, new_strides, self.dtype)
    }

    pub fn permute(&self, dims: &[usize]) -> Result<Tensor> {
        if dims.len() != self.shape.len() {
             return Err(LLMForgeError::ShapeMismatch {
                expected: vec![self.shape.len()],
                actual: vec![dims.len()],
            });
        }

        let mut new_shape = SmallVec::with_capacity(dims.len());
        let mut new_strides = SmallVec::with_capacity(dims.len());

        for &d in dims {
            if d >= self.shape.len() {
                return Err(LLMForgeError::IndexOutOfBounds { index: d, dim: 0, size: self.shape.len() });
            }
            new_shape.push(self.shape[d]);
            new_strides.push(self.strides[d]);
        }

        Self::view(self.data.clone(), new_shape, new_strides, self.dtype)
    }

    pub fn contiguous(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        if self.dtype == DType::Q8_0 {
            return Err(LLMForgeError::NotImplemented(
                "Cannot make Q8_0 tensors contiguous via element-wise copy".into()
            ));
        }

        let size: usize = self.shape.iter().product();
        let elem_size = self.dtype.size();
        let mut data: Vec<u8> = Vec::with_capacity(size * elem_size);

        // Recursive copy
        let mut indices = vec![0; self.shape.len()];
        self.recursive_copy(&mut data, &mut indices, 0, elem_size)?;

        Ok(Tensor::new(data, self.shape.clone(), self.dtype))
    }

    fn recursive_copy(&self, data: &mut Vec<u8>, indices: &mut Vec<usize>, dim: usize, elem_size: usize) -> Result<()> {
        if dim == self.shape.len() {
            // Leaf: copy element
            let mut offset = 0;
            for (i, &idx) in indices.iter().enumerate() {
                offset += idx * self.strides[i];
            }

            // SAFETY: as_ptr() returns a valid pointer to the start of storage.
            // `offset * elem_size` is within bounds because indices are bounded
            // by shape dimensions and strides are set up to address valid data.
            // The resulting slice of `elem_size` bytes is within the allocation.
            let ptr = unsafe { self.as_ptr()? };
            let val_ptr = unsafe { ptr.add(offset * elem_size) };
            let val_bytes = unsafe { std::slice::from_raw_parts(val_ptr, elem_size) };
            data.extend_from_slice(val_bytes);
            return Ok(());
        }

        for i in 0..self.shape[dim] {
            indices[dim] = i;
            self.recursive_copy(data, indices, dim + 1, elem_size)?;
        }
        Ok(())
    }

    // Simplified slicing for KVCache usage: allows slicing along dim 2 (sequence length)
    pub fn slice_sequence(&self, start: usize, end: usize) -> Result<Tensor> {
        if self.shape.len() < 3 {
             return Err(LLMForgeError::ShapeMismatch {
                expected: vec![3], // At least 3 dims
                actual: self.shape.to_vec(),
            });
        }

        let batch = self.shape[0];
        let heads = self.shape[1];
        let seq_len = self.shape[2];
        let head_dim = if self.shape.len() > 3 { self.shape[3] } else { 1 };

        if end > seq_len || start > end {
             return Err(LLMForgeError::IndexOutOfBounds {
                index: end,
                dim: 2,
                size: seq_len,
            });
        }

        let new_seq_len = end - start;
        let mut new_shape = self.shape.clone();
        new_shape[2] = new_seq_len;

        let src_data = self.as_slice_f32()?;
        let mut dest_data = Vec::with_capacity(batch * heads * new_seq_len * head_dim);

        // Stride calculations
        let stride_b = self.strides[0];
        let stride_h = self.strides[1];
        let stride_s = self.strides[2];

        for b in 0..batch {
            for h in 0..heads {
                for s in 0..new_seq_len {
                    let src_idx = b * stride_b + h * stride_h + (start + s) * stride_s;
                    // Copy whole head_dim vector
                    let src_end = src_idx + head_dim;
                    dest_data.extend_from_slice(&src_data[src_idx..src_end]);
                }
            }
        }

        let out_bytes = f32_vec_to_bytes(dest_data);

        Ok(Self::new(out_bytes, new_shape, self.dtype))
    }

    pub fn slice_assign_sequence(&mut self, start: usize, source: &Tensor) -> Result<()> {
         if self.shape.len() < 3 || source.shape.len() < 3 {
             return Err(LLMForgeError::ShapeMismatch {
                expected: vec![3],
                actual: self.shape.to_vec(),
            });
        }

        let seq_len_src = source.shape[2];
        if start + seq_len_src > self.shape[2] {
             return Err(LLMForgeError::IndexOutOfBounds {
                index: start + seq_len_src,
                dim: 2,
                size: self.shape[2],
            });
        }

        // Ensure other dims match
        if self.shape[0] != source.shape[0] || self.shape[1] != source.shape[1] || self.shape.get(3) != source.shape.get(3) {
             return Err(LLMForgeError::ShapeMismatch {
                expected: self.shape.to_vec(),
                actual: source.shape.to_vec(),
            });
        }

        let dest_ptr = match Arc::get_mut(&mut self.data) {
            Some(Storage::Owned(vec)) => vec.as_mut_ptr() as *mut f32,
            _ => return Err(LLMForgeError::NotImplemented("Cannot mutate shared or mmap tensor".into())),
        };

        let src_data = source.as_slice_f32()?;
        let head_dim = if self.shape.len() > 3 { self.shape[3] } else { 1 };

        let stride_b = self.strides[0];
        let stride_h = self.strides[1];
        let stride_s = self.strides[2];

        let src_stride_b = source.strides[0];
        let src_stride_h = source.strides[1];
        let src_stride_s = source.strides[2];

        for b in 0..self.shape[0] {
            for h in 0..self.shape[1] {
                for s in 0..seq_len_src {
                    let dest_idx = b * stride_b + h * stride_h + (start + s) * stride_s;
                    let src_idx = b * src_stride_b + h * src_stride_h + s * src_stride_s;

                    // SAFETY: dest_ptr was obtained from Arc::get_mut on Owned storage,
                    // ensuring exclusive access. src_idx and dest_idx are within bounds
                    // because loop indices are bounded by shape dimensions and strides.
                    // head_dim elements fit within both source and destination allocations.
                    // Source and destination do not overlap (different allocations).
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src_data.as_ptr().add(src_idx),
                            dest_ptr.add(dest_idx),
                            head_dim
                        );
                    }
                }
            }
        }

        Ok(())
    }
}
