use crate::RangesArg;
use alloc::vec::Vec;
use core::ops::Range;

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

    /// Convert to covering ranges for each dimension in the shape.
    pub fn into_ranges(self) -> Vec<Range<usize>> {
        self.into_iter().map(|d| 0..d).collect()
    }

    /// Applies a slice to the shape.
    ///
    /// For more complex indexing with different slice ranges, see also the slice
    /// macro [`s!`](crate::s).
    ///
    /// # Arguments
    ///
    /// * `ranges` - A type implementing the `RangesArg` trait, which can be:
    ///   - A single range (slice the first dimension)
    ///   - A single index (slice the first dimension)
    ///   - An array of ranges
    ///
    /// # Behavior
    ///
    /// - Supports partial and full slicing in any number of dimensions.
    /// - Missing ranges are treated as full slices if D > D2.
    /// - Handles negative indices by wrapping around from the end of the dimension.
    /// - Clamps ranges to the shape's dimensions if they exceed the bounds.
    ///
    /// # Panics
    ///
    /// - If the number of ranges provided exceeds the shape's dimensions.
    /// - If a range is descending (e.g., 2..1) or empty (e.g., 1..1).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape, s};
    ///
    /// fn example<B: Backend>() {
    ///     // 1D slicing
    ///     assert_eq!(Shape::new([4]).slice(s![1..4]), [1..3]);
    ///
    ///     // 2D slicing
    ///     assert_eq!(Shape::new([3, 4]).slice(s![1..4, 0..2]), [1..3, 0..2]);
    ///
    ///     // Using negative indices
    ///     assert_eq!(Shape::new([3]).slice(s![..-2]), [0..1]);
    ///
    ///     // Using the slice macro to select different ranges
    ///     assert_eq!(
    ///         Shape::new([2, 3, 4]).slice(s![.., 1..-1]),
    ///         [0..2, 1..2]);
    /// }
    /// ```
    ///
    /// # Note
    ///
    /// This function uses the `RangesArg` trait for flexible range specification. The trait
    /// handles the conversion of various range formats and applies clamping and negative
    /// index handling internally.
    pub fn slice<const D: usize, R: RangesArg<D>>(self, ranges: R) -> [Range<usize>; D] {
        ranges.into_ranges(self)
    }

    /// Construct a vector of the dims.
    pub fn to_vec(&self) -> Vec<usize> {
        self.dims.clone()
    }
}

impl IntoIterator for Shape {
    type Item = usize;
    type IntoIter = alloc::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.dims.into_iter()
    }
}

impl<const D: usize> From<[usize; D]> for Shape {
    fn from(dims: [usize; D]) -> Self {
        Shape::new(dims)
    }
}

impl<const D: usize> From<[i64; D]> for Shape {
    fn from(dims: [i64; D]) -> Self {
        Shape {
            dims: dims.into_iter().map(|d| d as usize).collect(),
        }
    }
}

impl<const D: usize> From<[i32; D]> for Shape {
    fn from(dims: [i32; D]) -> Self {
        Shape {
            dims: dims.into_iter().map(|d| d as usize).collect(),
        }
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
    use crate::s;
    use alloc::vec;

    #[test]
    fn num_elements() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);
        assert_eq!(120, shape.num_elements());
    }

    #[test]
    fn test_iter() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);

        assert_eq!(shape.into_iter().sum::<usize>(), 14);
    }

    #[test]
    fn test_into_ranges() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);
        assert_eq!(shape.into_ranges(), vec![0..2, 0..3, 0..4, 0..5]);
    }

    #[test]
    fn test_to_vec() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);
        assert_eq!(shape.to_vec(), vec![2, 3, 4, 5]);
    }

    #[allow(clippy::single_range_in_vec_init)]
    #[test]
    fn test_slice() {
        assert_eq!(Shape::new([3]).slice(s![1..4]), [1..3]);

        assert_eq!(Shape::new([3, 4]).slice(s![1..4, 0..2]), [1..3, 0..2]);

        assert_eq!(Shape::new([3]).slice(s![..-2]), [0..1]);

        assert_eq!(Shape::new([2, 3, 4]).slice(s![.., 1..-1]), [0..2, 1..2]);

        assert_eq!(Shape::new([2, 3, 4]).slice(s![..20, 2]), [0..2, 2..3]);
    }
}
