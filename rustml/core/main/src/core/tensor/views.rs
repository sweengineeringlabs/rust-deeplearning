//! Shape operations: reshape, transpose, permute, contiguous, slice, select, broadcast.

use crate::api::error::{TensorError, TensorResult};
use crate::api::types::DType;
use crate::core::shape::Shape;
use super::tensor::{Tensor, Storage, f32_vec_to_bytes, TensorShape};
use smallvec::SmallVec;
use std::sync::Arc;

impl Tensor {
    // ==================== Reshape ====================

    /// Reshape the tensor. Zero-copy if contiguous.
    pub fn reshape(&self, shape: &[usize]) -> TensorResult<Tensor> {
        let current_size: usize = self.shape_sv.iter().product();
        let new_size: usize = shape.iter().product();

        if current_size != new_size {
            return Err(TensorError::ShapeMismatch {
                expected: vec![current_size],
                got: vec![new_size],
            });
        }

        if self.is_contiguous() {
            let new_strides = Self::compute_strides_sv(shape);
            Self::view(
                self.data.clone(),
                SmallVec::from_slice(shape),
                new_strides,
                self.dtype,
            )
        } else {
            let contiguous = self.contiguous()?;
            let new_strides = Self::compute_strides_sv(shape);
            Self::view(
                contiguous.data,
                SmallVec::from_slice(shape),
                new_strides,
                self.dtype,
            )
        }
    }

    // ==================== Transpose ====================

    /// Transpose two dimensions. Zero-copy via stride swapping.
    pub fn transpose(&self, dim0: i64, dim1: i64) -> TensorResult<Tensor> {
        let dim0_idx = self.normalize_dim(dim0)?;
        let dim1_idx = self.normalize_dim(dim1)?;

        let mut new_shape = self.shape_sv.clone();
        let mut new_strides = self.strides.clone();

        new_shape.swap(dim0_idx, dim1_idx);
        new_strides.swap(dim0_idx, dim1_idx);

        Self::view(self.data.clone(), new_shape, new_strides, self.dtype)
    }

    /// Transpose last two dimensions (convenience for matrix operations).
    pub fn t(&self) -> TensorResult<Tensor> {
        if self.ndim() < 2 {
            return Err(TensorError::InvalidOperation(
                "Cannot transpose tensor with less than 2 dimensions".into(),
            ));
        }
        self.transpose(-2, -1)
    }

    // ==================== Permute ====================

    /// Permute dimensions. Zero-copy via stride reordering.
    pub fn permute(&self, dims: &[usize]) -> TensorResult<Tensor> {
        if dims.len() != self.shape_sv.len() {
            return Err(TensorError::InvalidOperation(format!(
                "Permutation must have {} dimensions, got {}",
                self.shape_sv.len(),
                dims.len()
            )));
        }

        let mut new_shape = SmallVec::with_capacity(dims.len());
        let mut new_strides = SmallVec::with_capacity(dims.len());

        for &d in dims {
            if d >= self.shape_sv.len() {
                return Err(TensorError::IndexOutOfBounds {
                    dim: 0,
                    index: d,
                    size: self.shape_sv.len(),
                });
            }
            new_shape.push(self.shape_sv[d]);
            new_strides.push(self.strides[d]);
        }

        Self::view(self.data.clone(), new_shape, new_strides, self.dtype)
    }

    // ==================== Contiguous ====================

    /// Make tensor contiguous (copy if necessary).
    pub fn contiguous(&self) -> TensorResult<Tensor> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        if matches!(self.dtype, DType::Q8_0 | DType::Q4_0) {
            return Err(TensorError::NotImplemented(
                "Cannot make quantized tensors contiguous via element-wise copy".into(),
            ));
        }

        let size: usize = self.shape_sv.iter().product();
        let elem_size = self.dtype.size();
        if elem_size == 0 {
            return Err(TensorError::NotImplemented(
                "Cannot make block-quantized tensors contiguous".into(),
            ));
        }
        let mut data: Vec<u8> = Vec::with_capacity(size * elem_size);

        let mut indices = vec![0; self.shape_sv.len()];
        self.recursive_copy(&mut data, &mut indices, 0, elem_size)?;

        Ok(Tensor::new(data, self.shape_sv.clone(), self.dtype))
    }

    fn recursive_copy(
        &self,
        data: &mut Vec<u8>,
        indices: &mut Vec<usize>,
        dim: usize,
        elem_size: usize,
    ) -> TensorResult<()> {
        if dim == self.shape_sv.len() {
            let mut offset = 0;
            for (i, &idx) in indices.iter().enumerate() {
                offset += idx * self.strides[i];
            }

            // SAFETY: as_ptr() returns a valid pointer. offset * elem_size is within bounds
            // because indices are bounded by shape dimensions.
            let ptr = unsafe { self.as_ptr()? };
            let val_ptr = unsafe { ptr.add(offset * elem_size) };
            let val_bytes = unsafe { std::slice::from_raw_parts(val_ptr, elem_size) };
            data.extend_from_slice(val_bytes);
            return Ok(());
        }

        for i in 0..self.shape_sv[dim] {
            indices[dim] = i;
            self.recursive_copy(data, indices, dim + 1, elem_size)?;
        }
        Ok(())
    }

    // ==================== Unsqueeze / Squeeze ====================

    /// Add a dimension of size 1 at the specified position.
    pub fn unsqueeze(&self, dim: i64) -> TensorResult<Tensor> {
        let ndim = self.ndim() as i64 + 1;
        let normalized = if dim < 0 { dim + ndim } else { dim };
        if normalized < 0 || normalized > self.ndim() as i64 {
            return Err(TensorError::InvalidDimension {
                dim,
                ndim: self.ndim(),
            });
        }
        let idx = normalized as usize;
        let mut new_dims = self.shape_sv.to_vec();
        new_dims.insert(idx, 1);
        self.reshape(&new_dims)
    }

    /// Remove a dimension of size 1.
    pub fn squeeze(&self, dim: i64) -> TensorResult<Tensor> {
        let dim_idx = self.normalize_dim(dim)?;
        if self.shape_sv[dim_idx] != 1 {
            return Err(TensorError::InvalidOperation(format!(
                "Cannot squeeze dimension {} with size {}",
                dim, self.shape_sv[dim_idx]
            )));
        }
        let mut new_dims = self.shape_sv.to_vec();
        new_dims.remove(dim_idx);
        self.reshape(&new_dims)
    }

    // ==================== Select ====================

    /// Select a single index along a dimension (reduces dimensionality).
    pub fn select(&self, dim: i64, index: usize) -> TensorResult<Tensor> {
        let dim_idx = self.normalize_dim(dim)?;
        let dim_size = self.shape_sv[dim_idx];

        if index >= dim_size {
            return Err(TensorError::IndexOutOfBounds {
                dim: dim_idx,
                index,
                size: dim_size,
            });
        }

        let mut new_dims: Vec<usize> = self.shape_sv.to_vec();
        new_dims.remove(dim_idx);
        let new_shape = if new_dims.is_empty() {
            Shape::scalar()
        } else {
            Shape::new(new_dims)
        };

        let mut new_data = Vec::with_capacity(new_shape.numel());
        self.collect_select(&mut new_data, dim_idx, index, &[], 0);

        Tensor::from_vec(new_data, new_shape)
    }

    fn collect_select(
        &self,
        result: &mut Vec<f32>,
        select_dim: usize,
        select_idx: usize,
        indices: &[usize],
        depth: usize,
    ) {
        if depth == self.ndim() {
            if let Ok(val) = self.get(indices) {
                result.push(val);
            }
            return;
        }

        if depth == select_dim {
            let mut new_indices = indices.to_vec();
            new_indices.push(select_idx);
            self.collect_select(result, select_dim, select_idx, &new_indices, depth + 1);
        } else {
            for i in 0..self.shape_sv[depth] {
                let mut new_indices = indices.to_vec();
                new_indices.push(i);
                self.collect_select(result, select_dim, select_idx, &new_indices, depth + 1);
            }
        }
    }

    // ==================== Slice ====================

    /// Slice the tensor along a dimension.
    pub fn slice(&self, dim: i64, start: usize, end: usize) -> TensorResult<Tensor> {
        let dim_idx = self.normalize_dim(dim)?;
        let dim_size = self.shape_sv[dim_idx];

        if start > end || end > dim_size {
            return Err(TensorError::InvalidSliceRange {
                start,
                end,
                size: dim_size,
            });
        }

        let new_size = end - start;
        let mut new_dims = self.shape_sv.to_vec();
        new_dims[dim_idx] = new_size;
        let new_shape = Shape::new(new_dims);

        let mut new_data = Vec::with_capacity(new_shape.numel());
        self.collect_slice(&mut new_data, dim_idx, start, end, &[], 0);

        Tensor::from_vec(new_data, new_shape)
    }

    fn collect_slice(
        &self,
        result: &mut Vec<f32>,
        slice_dim: usize,
        start: usize,
        end: usize,
        indices: &[usize],
        depth: usize,
    ) {
        if depth == self.ndim() {
            if let Ok(val) = self.get(indices) {
                result.push(val);
            }
            return;
        }

        let range = if depth == slice_dim {
            start..end
        } else {
            0..self.shape_sv[depth]
        };

        for i in range {
            let mut new_indices = indices.to_vec();
            new_indices.push(i);
            self.collect_slice(result, slice_dim, start, end, &new_indices, depth + 1);
        }
    }

    /// Slice along dim 2 (sequence length) for KVCache usage.
    pub fn slice_sequence(&self, start: usize, end: usize) -> TensorResult<Tensor> {
        if self.shape_sv.len() < 3 {
            return Err(TensorError::InvalidOperation(
                "slice_sequence requires at least 3D tensor".into(),
            ));
        }

        let batch = self.shape_sv[0];
        let heads = self.shape_sv[1];
        let seq_len = self.shape_sv[2];
        let head_dim = if self.shape_sv.len() > 3 {
            self.shape_sv[3]
        } else {
            1
        };

        if end > seq_len || start > end {
            return Err(TensorError::IndexOutOfBounds {
                dim: 2,
                index: end,
                size: seq_len,
            });
        }

        let new_seq_len = end - start;
        let mut new_shape = self.shape_sv.clone();
        new_shape[2] = new_seq_len;

        let src_data = self.as_slice_f32()?;
        let mut dest_data = Vec::with_capacity(batch * heads * new_seq_len * head_dim);

        let stride_b = self.strides[0];
        let stride_h = self.strides[1];
        let stride_s = self.strides[2];

        for b in 0..batch {
            for h in 0..heads {
                for s in 0..new_seq_len {
                    let src_idx = b * stride_b + h * stride_h + (start + s) * stride_s;
                    let src_end = src_idx + head_dim;
                    dest_data.extend_from_slice(&src_data[src_idx..src_end]);
                }
            }
        }

        Ok(Tensor::new(
            f32_vec_to_bytes(dest_data),
            new_shape,
            self.dtype,
        ))
    }

    /// Slice rows from a 2D tensor: returns tensor[start..end, :].
    pub fn slice_rows(&self, start: usize, end: usize) -> TensorResult<Tensor> {
        if self.shape_sv.len() != 2 {
            return Err(TensorError::InvalidOperation(
                "slice_rows requires 2D tensor".into(),
            ));
        }

        let rows = self.shape_sv[0];
        let cols = self.shape_sv[1];

        if start > end || end > rows {
            return Err(TensorError::IndexOutOfBounds {
                dim: 0,
                index: end,
                size: rows,
            });
        }

        let n_rows = end - start;
        let data = self.as_slice_f32()?;
        let mut out = Vec::with_capacity(n_rows * cols);

        for r in start..end {
            let row_start = r * self.strides[0];
            for c in 0..cols {
                out.push(data[row_start + c * self.strides[1]]);
            }
        }

        Ok(Tensor::new(
            f32_vec_to_bytes(out),
            SmallVec::from_slice(&[n_rows, cols]),
            self.dtype,
        ))
    }

    /// In-place assign along dim 2 for KVCache updates.
    pub fn slice_assign_sequence(&mut self, start: usize, source: &Tensor) -> TensorResult<()> {
        if self.shape_sv.len() < 3 || source.shape_sv.len() < 3 {
            return Err(TensorError::InvalidOperation(
                "slice_assign_sequence requires at least 3D tensors".into(),
            ));
        }

        let seq_len_src = source.shape_sv[2];
        if start + seq_len_src > self.shape_sv[2] {
            return Err(TensorError::IndexOutOfBounds {
                dim: 2,
                index: start + seq_len_src,
                size: self.shape_sv[2],
            });
        }

        if self.shape_sv[0] != source.shape_sv[0]
            || self.shape_sv[1] != source.shape_sv[1]
            || self.shape_sv.get(3) != source.shape_sv.get(3)
        {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape_sv.to_vec(),
                got: source.shape_sv.to_vec(),
            });
        }

        let dest_ptr = match Arc::get_mut(&mut self.data) {
            Some(Storage::Owned(vec)) => vec.as_mut_ptr() as *mut f32,
            _ => {
                return Err(TensorError::InvalidOperation(
                    "Cannot mutate shared or mmap tensor".into(),
                ))
            }
        };

        let src_data = source.as_slice_f32()?;
        let head_dim = if self.shape_sv.len() > 3 {
            self.shape_sv[3]
        } else {
            1
        };

        let stride_b = self.strides[0];
        let stride_h = self.strides[1];
        let stride_s = self.strides[2];

        let src_stride_b = source.strides[0];
        let src_stride_h = source.strides[1];
        let src_stride_s = source.strides[2];

        for b in 0..self.shape_sv[0] {
            for h in 0..self.shape_sv[1] {
                for s in 0..seq_len_src {
                    let dest_idx = b * stride_b + h * stride_h + (start + s) * stride_s;
                    let src_idx = b * src_stride_b + h * src_stride_h + s * src_stride_s;

                    // SAFETY: dest_ptr is from Arc::get_mut on Owned storage (exclusive access).
                    // Indices are within bounds due to shape checks above.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src_data.as_ptr().add(src_idx),
                            dest_ptr.add(dest_idx),
                            head_dim,
                        );
                    }
                }
            }
        }

        Ok(())
    }

    // ==================== Broadcast ====================

    /// Broadcast to a Shape.
    pub fn broadcast_to(&self, shape: &Shape) -> TensorResult<Tensor> {
        self.broadcast_to_shape(&SmallVec::from_slice(shape.dims()))
    }

    /// Broadcast to a TensorShape (internal).
    pub(crate) fn broadcast_to_shape(&self, target: &TensorShape) -> TensorResult<Tensor> {
        if self.shape_sv.as_slice() == target.as_slice() {
            return Ok(self.clone());
        }

        // Validate broadcast is possible
        let target_shape = Shape::new(target.to_vec());
        let self_shape = Shape::new(self.shape_sv.to_vec());
        let broadcast_shape = self_shape.broadcast_with(&target_shape).ok_or_else(|| {
            TensorError::BroadcastError {
                shape1: self.shape_sv.to_vec(),
                shape2: target.to_vec(),
            }
        })?;

        if broadcast_shape.dims() != target.as_slice() {
            return Err(TensorError::BroadcastError {
                shape1: self.shape_sv.to_vec(),
                shape2: target.to_vec(),
            });
        }

        let mut new_data = Vec::with_capacity(target_shape.numel());
        self.collect_broadcast(&mut new_data, &target_shape, &[], 0);

        Tensor::from_vec(new_data, target_shape)
    }

    fn collect_broadcast(
        &self,
        result: &mut Vec<f32>,
        target_shape: &Shape,
        indices: &[usize],
        depth: usize,
    ) {
        if depth == target_shape.ndim() {
            let offset = target_shape.ndim() - self.ndim();
            let mut src_indices = Vec::with_capacity(self.ndim());
            for i in 0..self.ndim() {
                let target_idx = indices[offset + i];
                let src_size = self.shape_sv[i];
                src_indices.push(if src_size == 1 { 0 } else { target_idx });
            }
            if let Ok(val) = self.get(&src_indices) {
                result.push(val);
            }
            return;
        }

        for i in 0..target_shape.dims()[depth] {
            let mut ni = indices.to_vec();
            ni.push(i);
            self.collect_broadcast(result, target_shape, &ni, depth + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reshape() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let r = t.reshape(&[3, 2]).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.get(&[0, 0]).unwrap(), 1.0);
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let tr = t.transpose(-2, -1).unwrap();
        assert_eq!(tr.shape(), &[3, 2]);
        // After transposing [2,3], element [0,1] should be original [1,0] = 4.0
        assert_eq!(tr.get(&[0, 1]).unwrap(), 4.0);
    }

    #[test]
    fn test_select() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let selected = t.select(0, 1).unwrap();
        assert_eq!(selected.shape(), &[3]);
        assert_eq!(selected.to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_slice() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let sliced = t.slice(1, 0, 2).unwrap();
        assert_eq!(sliced.shape(), &[2, 2]);
    }

    #[test]
    fn test_broadcast() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let target = Shape::new(vec![2, 3]);
        let b = a.broadcast_to(&target).unwrap();
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(b.get(&[1, 0]).unwrap(), 1.0);
    }

    #[test]
    fn test_tril() {
        let t = Tensor::tril(3);
        assert_eq!(t.shape(), &[3, 3]);
        assert_eq!(t.get(&[0, 1]).unwrap(), 0.0);
        assert_eq!(t.get(&[1, 0]).unwrap(), 1.0);
        assert_eq!(t.get(&[2, 2]).unwrap(), 1.0);
    }

    #[test]
    fn test_unsqueeze_squeeze() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let u = t.unsqueeze(0).unwrap();
        assert_eq!(u.shape(), &[1, 3]);
        let s = u.squeeze(0).unwrap();
        assert_eq!(s.shape(), &[3]);
    }
}
