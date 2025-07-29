use crate::baselib::indexing::shape::Shape;
use crate::baselib::indexing::slice::Slice;
use std::ops::Range;

/// Trait used for slice dim arguments.
pub trait RangeArg {
    /// Converts into a range for the `tensor.slice_dim()` function
    fn into_range(self, shape_dim: usize) -> Range<usize>;
}

impl<T: Into<Slice>> RangeArg for T {
    fn into_range(self, shape_dim: usize) -> Range<usize> {
        self.into().into_range(shape_dim)
    }
}

/// Trait used for slice arguments
pub trait RangesArg<const D2: usize> {
    /// Converts into a set of ranges to `[Range<usize>; D2]` for the `tensor.slice()` function
    fn into_ranges(self, shape: Shape) -> [Range<usize>; D2];
}

impl<const D2: usize, T: Into<Slice>> RangesArg<D2> for [T; D2] {
    fn into_ranges(self, shape: Shape) -> [Range<usize>; D2] {
        // clamp the ranges to the shape dimensions
        let ranges = self
            .into_iter()
            .enumerate()
            .map(|(i, range)| range.into().into_range(shape.dims[i]))
            .collect::<Vec<_>>();
        ranges.try_into().unwrap()
    }
}

impl<T: Into<Slice>> RangesArg<1> for T {
    fn into_ranges(self, shape: Shape) -> [Range<usize>; 1] {
        [self.into().into_range(shape.dims[0])]
    }
}
