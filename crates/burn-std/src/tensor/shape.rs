//! Tensor shape definition.

use super::{Slice, SliceArg};
use alloc::vec::Vec;
use core::ops::Range;

pub use crate::errors::ExpressionError;

pub use cubecl_zspace::{MetadataError, Shape, calculate_matmul_output, shape};

/// Slice-relatedo ops on [`Shape`]
pub trait SliceOps: Sized {
    /// Convert shape dimensions to full covering ranges (0..dim) for each dimension.
    fn into_ranges(self) -> Vec<Range<usize>>;
    /// Converts slice arguments into an array of slice specifications for the shape.
    ///
    /// This method returns an array of `Slice` objects that can be used for slicing operations.
    /// The slices are clamped to the shape's dimensions. Similar to `into_ranges()`, but
    /// allows custom slice specifications instead of full ranges.
    /// For creating complex slice specifications, use the [`s!`] macro.
    ///
    /// # Arguments
    ///
    /// * `slices` - An array of slice specifications, where each element can be:
    ///   - A range (e.g., `2..5`)
    ///   - An index
    ///   - A `Slice` object
    ///   - The output of the [`s!`] macro for advanced slicing
    ///
    /// # Behavior
    ///
    /// - Supports partial and full slicing in any number of dimensions.
    /// - Missing ranges are treated as full slices if D > D2.
    /// - Handles negative indices by wrapping around from the end of the dimension.
    /// - Clamps ranges to the shape's dimensions if they exceed the bounds.
    ///
    /// # Returns
    ///
    /// An array of `Slice` objects corresponding to the provided slice specifications,
    /// clamped to the shape's actual dimensions.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use burn_std::{Shape, Slice, s};
    ///
    /// fn example() {
    ///     // 1D slicing
    ///     let slices = Shape::new([4]).into_slices(1..4);
    ///     assert_eq!(slices[0].to_range(4), 1..3);
    ///
    ///     // 2D slicing
    ///     let slices = Shape::new([3, 4]).into_slices(s![1..4, 0..2]);
    ///     assert_eq!(slices[0].to_range(3), 1..3);
    ///     assert_eq!(slices[1].to_range(4), 0..2);
    ///
    ///     // Using negative indices
    ///     let slices = Shape::new([3]).into_slices(..-2);
    ///     assert_eq!(slices[0].to_range(3), 0..1);
    ///
    ///     // Using the slice macro to select different ranges
    ///     let slices = Shape::new([2, 3, 4]).into_slices(s![.., 1..-1]);
    ///     assert_eq!(slices[0].to_range(2), 0..2);
    ///     assert_eq!(slices[1].to_range(3), 1..2);
    /// }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`s!`] - The recommended macro for creating slice specifications
    /// - [`Shape::into_ranges`] - Convert to full covering ranges
    ///
    /// [`s!`]: crate::s!
    fn into_slices<S>(self, slices: S) -> Vec<Slice>
    where
        S: SliceArg;
    /// Compute the output shape from the given slices.
    fn slice(self, slices: &[Slice]) -> Result<Self, MetadataError>;
}

impl SliceOps for Shape {
    fn into_ranges(self) -> Vec<Range<usize>> {
        self.iter().map(|&d| 0..d).collect()
    }

    fn into_slices<S>(self, slices: S) -> Vec<Slice>
    where
        S: SliceArg,
    {
        slices.into_slices(&self)
    }

    fn slice(mut self, slices: &[Slice]) -> Result<Self, MetadataError> {
        if slices.len() > self.rank() {
            return Err(MetadataError::RankMismatch {
                left: self.rank(),
                right: slices.len(),
            });
        }

        slices
            .iter()
            .zip(self.iter_mut())
            .for_each(|(slice, dim_size)| *dim_size = slice.output_size(*dim_size));

        Ok(self)
    }
}

#[cfg(test)]
#[allow(clippy::identity_op, reason = "useful for clarity")]
mod tests {
    use super::*;
    use crate::s;
    use alloc::vec;

    #[test]
    fn test_into_ranges() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);
        assert_eq!(shape.into_ranges(), vec![0..2, 0..3, 0..4, 0..5]);
    }

    #[allow(clippy::single_range_in_vec_init)]
    #[test]
    fn test_into_slices() {
        let slices = Shape::new([3]).into_slices(1..4);
        assert_eq!(slices[0].to_range(3), 1..3);

        let slices = Shape::new([3, 4]).into_slices(s![1..4, 0..2]);
        assert_eq!(slices[0].to_range(3), 1..3);
        assert_eq!(slices[1].to_range(4), 0..2);

        let slices = Shape::new([3]).into_slices(..-2);
        assert_eq!(slices[0].to_range(3), 0..1);

        let slices = Shape::new([2, 3, 4]).into_slices(s![.., 1..-1]);
        assert_eq!(slices[0].to_range(2), 0..2);
        assert_eq!(slices[1].to_range(3), 1..2);

        let slices = Shape::new([2, 3, 4]).into_slices(s![..20, 2]);
        assert_eq!(slices[0].to_range(2), 0..2);
        assert_eq!(slices[1].to_range(3), 2..3);
    }

    #[test]
    fn test_shape_as_slice() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);

        assert_eq!(shape.as_slice(), dims.as_slice());

        // Deref coercion
        let shape_slice: &[usize] = &shape;
        assert_eq!(shape_slice, *&[2, 3, 4, 5]);
    }

    #[test]
    fn test_shape_as_mut_slice() {
        let mut dims = [2, 3, 4, 5];
        let mut shape = Shape::new(dims);

        let shape_mut = shape.as_mut_slice();
        assert_eq!(shape_mut, dims.as_mut_slice());
        shape_mut[1] = 6;

        assert_eq!(shape_mut, &[2, 6, 4, 5]);

        let mut shape = Shape::new(dims);
        let shape = &mut shape[..];
        shape[1] = 6;

        assert_eq!(shape, shape_mut)
    }

    #[test]
    fn test_shape_slice_output_shape_basic() {
        // Test basic slicing with step=1
        let slices = [
            Slice::new(0, Some(5), 1), // 5 elements
            Slice::new(2, Some(8), 1), // 6 elements
        ];
        let original_shape = Shape::new([10, 10, 10]);
        let result = original_shape.slice(&slices).unwrap();
        assert_eq!(result, Shape::new([5, 6, 10]));
    }

    #[test]
    fn test_shape_slice_output_shape_with_positive_steps() {
        // Test slicing with various positive steps
        let slices = [
            Slice::new(0, Some(10), 2), // [0,2,4,6,8] -> 5 elements
            Slice::new(1, Some(9), 3),  // [1,4,7] -> 3 elements
            Slice::new(0, Some(7), 4),  // [0,4] -> 2 elements
        ];
        let original_shape = Shape::new([20, 20, 20, 30]);
        let result = original_shape.slice(&slices).unwrap();
        assert_eq!(result, Shape::new([5, 3, 2, 30]));
    }

    #[test]
    fn test_shape_slice_output_shape_with_negative_steps() {
        // Test slicing with negative steps (backward iteration)
        let slices = [
            Slice::new(0, Some(10), -1), // 10 elements traversed backward
            Slice::new(2, Some(8), -2),  // [7,5,3] -> 3 elements
        ];
        let original_shape = Shape::new([20, 20, 20]);
        let result = original_shape.slice(&slices).unwrap();
        assert_eq!(result, Shape::new([10, 3, 20]));
    }

    #[test]
    fn test_shape_slice_output_shape_mixed_steps() {
        // Test with a mix of positive, negative, and unit steps
        let slices = [
            Slice::from_range_stepped(1..6, 1),   // 5 elements
            Slice::from_range_stepped(0..10, -3), // [9,6,3,0] -> 4 elements
            Slice::from_range_stepped(2..14, 4),  // [2,6,10] -> 3 elements
        ];
        let original_shape = Shape::new([20, 20, 20]);
        let result = original_shape.slice(&slices).unwrap();
        assert_eq!(result, Shape::new([5, 4, 3]));
    }

    #[test]
    fn test_shape_slice_output_shape_partial_dims() {
        // Test when slices has fewer dimensions than original shape
        let slices = [
            Slice::from_range_stepped(2..7, 2), // [2,4,6] -> 3 elements
        ];
        let original_shape = Shape::new([10, 20, 30, 40]);
        let result = original_shape.slice(&slices).unwrap();
        assert_eq!(result, Shape::new([3, 20, 30, 40]));
    }

    #[test]
    fn test_shape_slice_output_shape_edge_cases() {
        // Test edge cases with small ranges and large steps
        let slices = [
            Slice::from_range_stepped(0..1, 1),    // Single element
            Slice::from_range_stepped(0..10, 100), // Step larger than range -> 1 element
            Slice::from_range_stepped(5..5, 1),    // Empty range -> 0 elements
        ];
        let original_shape = Shape::new([10, 20, 30]);
        let result = original_shape.slice(&slices).unwrap();
        assert_eq!(result, Shape::new([1, 1, 0]));
    }

    #[test]
    fn test_shape_slice_output_shape_empty() {
        // Test with no slice infos (should return original shape)
        let slices = [];
        let original_shape = Shape::new([10, 20, 30]);
        let result = original_shape.slice(&slices).unwrap();
        assert_eq!(result, Shape::new([10, 20, 30]));
    }

    #[test]
    fn test_shape_slice_output_shape_uneven_division() {
        // Test cases where range size doesn't divide evenly by step
        let slices = [
            Slice::from_range_stepped(0..7, 3), // ceil(7/3) = 3 elements: [0,3,6]
            Slice::from_range_stepped(0..11, 4), // ceil(11/4) = 3 elements: [0,4,8]
            Slice::from_range_stepped(1..10, 5), // ceil(9/5) = 2 elements: [1,6]
        ];
        let original_shape = Shape::new([20, 20, 20]);
        let result = original_shape.slice(&slices).unwrap();
        assert_eq!(result, Shape::new([3, 3, 2]));
    }
}
