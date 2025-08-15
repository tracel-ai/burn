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

    /// Change the shape to one dimensional with the same number of elements.
    pub fn flatten(&self) -> Self {
        Self {
            dims: [self.dims.iter().product()].into(),
        }
    }
}

impl<const D: usize> From<[usize; D]> for Shape {
    fn from(dims: [usize; D]) -> Self {
        Shape::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape { dims: dims.into() }
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
    use crate::baselib::indexing::RangesArg;
    use crate::s;

    #[test]
    fn num_elements() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);
        assert_eq!(120, shape.num_elements());
    }

    #[test]
    fn slice_range_single_dim_leading() {
        let shape = Shape::new([8, 4]);

        // Half-open range
        assert_eq!([0..5], (0..5).into_ranges(shape.clone()));
        assert_eq!([0..5], [0..5].into_ranges(shape.clone()));
        assert_eq!([5..7], [-3..-1].into_ranges(shape.clone()));

        // Inclusive range
        assert_eq!([0..5], (0..=4).into_ranges(shape.clone()));
        assert_eq!([0..5], [0..=4].into_ranges(shape.clone()));
        assert_eq!([6..8], [-2..=-1].into_ranges(shape.clone()));

        // Unbounded start
        assert_eq!([0..3], (..3).into_ranges(shape.clone()));
        assert_eq!([0..3], [..3].into_ranges(shape.clone()));
        assert_eq!([0..3], [..-5].into_ranges(shape.clone()));

        // Unbounded end
        assert_eq!([5..8], (5..).into_ranges(shape.clone()));
        assert_eq!([5..8], [5..].into_ranges(shape.clone()));
        assert_eq!([5..8], [-3..].into_ranges(shape.clone()));

        // Full range
        assert_eq!([0..8], [..].into_ranges(shape));
    }

    #[test]
    fn slice_range_multi_dim() {
        let shape = Shape::new([8, 4]);

        // Multiple ways to provide ranges
        assert_eq!([0..5, 0..4], [0..5, 0..4].into_ranges(shape.clone()));
        assert_eq!([0..8, 0..4], [0.., 0..].into_ranges(shape.clone()));
        assert_eq!([0..8, 0..4], [0..=7, 0..=3].into_ranges(shape.clone()));

        assert_eq!([0..5, 0..3], [0..5, 0..3].into_ranges(shape.clone()));

        assert_eq!([0..8, 0..4], [0.., 0..].into_ranges(shape));
    }

    #[test]
    fn slice_range_multi_dim_index() {
        let shape = Shape::new([8, 4]);

        // Indices (single integer) should also convert to correct range
        assert_eq!([0..1, 2..3], [0, 2].into_ranges(shape.clone()));
        assert_eq!([7..8, 3..4], [-1, -1].into_ranges(shape.clone()));
        assert_eq!([7..8], (-1).into_ranges(shape.clone()));
        assert_eq!([7..8], 7.into_ranges(shape));
    }

    #[test]
    fn slice_range_multi_dim_heterogeneous() {
        // Slice macro `s![]` can be used to provide different range types
        let shape = Shape::new([8, 4, 2]);
        let slice = s![0..5, .., -1];
        assert_eq!([0..5, 0..4, 1..2], slice.into_ranges(shape));

        let shape = Shape::new([8, 4, 2, 3]);
        let slice = s![..=4, 0..=3, .., -2..];
        assert_eq!([0..5, 0..4, 0..2, 1..3], slice.into_ranges(shape));

        let shape = Shape::new([3, 4]);
        let slice = s![1..-1, ..];
        assert_eq!([1..2, 0..4], slice.into_ranges(shape));
    }
}
