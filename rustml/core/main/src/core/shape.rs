//! Shape representation and utilities for tensors

use std::fmt;

/// Represents the shape of a tensor
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Create a new shape from dimensions
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    /// Create a scalar shape (0 dimensions)
    pub fn scalar() -> Self {
        Self { dims: vec![] }
    }

    /// Get the dimensions
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Number of dimensions
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Total number of elements
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Check if shape is scalar
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Get size at dimension (supports negative indexing)
    pub fn size(&self, dim: i64) -> Option<usize> {
        let idx = self.normalize_dim(dim)?;
        Some(self.dims[idx])
    }

    /// Normalize negative dimension index
    pub fn normalize_dim(&self, dim: i64) -> Option<usize> {
        let ndim = self.ndim() as i64;
        let normalized = if dim < 0 { dim + ndim } else { dim };
        if normalized >= 0 && normalized < ndim {
            Some(normalized as usize)
        } else {
            None
        }
    }

    /// Compute broadcast shape with another shape
    pub fn broadcast_with(&self, other: &Shape) -> Option<Shape> {
        let max_dims = self.ndim().max(other.ndim());
        let mut result = Vec::with_capacity(max_dims);

        for i in 0..max_dims {
            let a = if i < self.ndim() {
                self.dims[self.ndim() - 1 - i]
            } else {
                1
            };
            let b = if i < other.ndim() {
                other.dims[other.ndim() - 1 - i]
            } else {
                1
            };

            if a == b {
                result.push(a);
            } else if a == 1 {
                result.push(b);
            } else if b == 1 {
                result.push(a);
            } else {
                return None;
            }
        }

        result.reverse();
        Some(Shape::new(result))
    }

    /// Create shape with an additional dimension
    pub fn with_dim(&self, dim: i64, size: usize) -> Option<Shape> {
        let ndim = self.ndim() as i64 + 1;
        let normalized = if dim < 0 { dim + ndim } else { dim };
        if normalized < 0 || normalized > self.ndim() as i64 {
            return None;
        }
        let idx = normalized as usize;
        let mut dims = self.dims.clone();
        dims.insert(idx, size);
        Some(Shape::new(dims))
    }

    /// Remove a dimension
    pub fn squeeze(&self, dim: i64) -> Option<Shape> {
        let idx = self.normalize_dim(dim)?;
        if self.dims[idx] != 1 {
            return None;
        }
        let mut dims = self.dims.clone();
        dims.remove(idx);
        Some(Shape::new(dims))
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape({:?})", self.dims)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape::new(dims.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Shape::new(dims.to_vec())
    }
}

impl AsRef<[usize]> for Shape {
    fn as_ref(&self) -> &[usize] {
        &self.dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_basic() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.numel(), 24);
        assert_eq!(shape.size(0), Some(2));
        assert_eq!(shape.size(-1), Some(4));
    }

    #[test]
    fn test_broadcast() {
        let a = Shape::new(vec![2, 1, 4]);
        let b = Shape::new(vec![3, 4]);
        let c = a.broadcast_with(&b);
        assert_eq!(c, Some(Shape::new(vec![2, 3, 4])));
    }
}
