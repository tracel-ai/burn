use alloc::vec::Vec;

/// Shape of a tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    /// The dimensions of the tensor.
    pub dims: Vec<usize>,
}

impl Shape {
    /// Returns the total number of elements of a tensor having this shape
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// Returns the number of dimensions.
    pub fn num_dims(&self) -> usize {
        self.dims.len()
    }

    /// Constructs a new `Shape`.
    pub fn new<const D: usize>(dims: [usize; D]) -> Self {
        // For backward compat
        Self {
            dims: dims.to_vec(),
        }
    }

    // For compat with dims: [usize; D]
    /// Returns the dimensions of the tensor as an array.
    pub fn dims<const D: usize>(&self) -> [usize; D] {
        let mut dims = [1; D];
        dims[..D].copy_from_slice(&self.dims[..D]);
        dims
    }
}

impl<const D: usize> From<[usize; D]> for Shape {
    fn from(dims: [usize; D]) -> Self {
        Shape::new(dims)
    }
}

impl From<Vec<i64>> for Shape {
    fn from(shape: Vec<i64>) -> Self {
        Self {
            dims: shape.into_iter().map(|d| d as usize).collect(),
        }
    }
}

impl From<Vec<u64>> for Shape {
    fn from(shape: Vec<u64>) -> Self {
        Self {
            dims: shape.into_iter().map(|d| d as usize).collect(),
        }
    }
}

impl From<Vec<usize>> for Shape {
    fn from(shape: Vec<usize>) -> Self {
        Self { dims: shape }
    }
}

impl From<&Vec<usize>> for Shape {
    fn from(shape: &Vec<usize>) -> Self {
        Self {
            dims: shape.clone(),
        }
    }
}

impl From<Shape> for Vec<usize> {
    fn from(shape: Shape) -> Self {
        shape.dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_elements() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);
        assert_eq!(120, shape.num_elements());
    }
}
