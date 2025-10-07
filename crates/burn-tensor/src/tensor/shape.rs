use crate::{Slice, SliceArg};
use alloc::vec::Vec;
use core::{
    ops::{Deref, DerefMut, Index, IndexMut, Range},
    slice::{Iter, IterMut, SliceIndex},
};
use serde::{Deserialize, Serialize};

/// Shape of a tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape {
    /// The dimensions of the tensor.
    pub dims: Vec<usize>,
}

#[allow(missing_docs)]
#[derive(Debug, Clone, PartialEq, Eq)]
/// Error that can occur when attempting to modify shapes.
pub enum ShapeError {
    /// The operands have different ranks.
    RankMismatch { left: usize, right: usize },
    /// A pair of dimensions are incompatible for broadcasting.
    IncompatibleDims {
        left: usize,
        right: usize,
        dim: usize,
    },
    /// Invalid dimension specified for the rank.
    OutOfBounds { dim: usize, rank: usize },
    /// A pair of shapes are incompatible for the operation.
    IncompatibleShapes { left: Shape, right: Shape },
    /// Invalid empty shape.
    Empty,
}

impl Shape {
    /// Constructs a new `Shape`.
    pub fn new<const D: usize>(dims: [usize; D]) -> Self {
        // For backward compat
        Self {
            dims: dims.to_vec(),
        }
    }

    /// Returns the total number of elements of a tensor having this shape
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// Returns the number of dimensions.
    ///
    /// Alias for `Shape::rank()`.
    pub fn num_dims(&self) -> usize {
        self.dims.len()
    }

    /// Returns the rank (the number of dimensions).
    ///
    /// Alias for `Shape::num_dims()`.
    pub fn rank(&self) -> usize {
        self.num_dims()
    }

    // For compat with dims: [usize; D]
    /// Returns the dimensions of the tensor as an array.
    pub fn dims<const D: usize>(&self) -> [usize; D] {
        let mut dims = [1; D];
        dims[..D].copy_from_slice(&self.dims[..D]);
        dims
    }

    /// Change the shape to one dimensional with the same number of elements.
    pub fn flatten(mut self) -> Self {
        self.dims = [self.num_elements()].into();
        self
    }

    /// Convert shape dimensions to full covering ranges (0..dim) for each dimension.
    pub fn into_ranges(self) -> Vec<Range<usize>> {
        self.into_iter().map(|d| 0..d).collect()
    }

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
    /// use burn_tensor::backend::Backend;
    /// use burn_tensor::{Tensor, Shape, Slice, s};
    ///
    /// fn example<B: Backend>() {
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
    /// - [`Tensor::slice`] - Apply slicing to a tensor
    /// - [`Shape::into_ranges`] - Convert to full covering ranges
    ///
    /// [`s!`]: crate::s!
    /// [`Tensor::slice`]: crate::Tensor::slice
    pub fn into_slices<const D: usize, S>(self, slices: S) -> [Slice; D]
    where
        S: SliceArg<D>,
    {
        slices.into_slices(self)
    }

    /// Construct a vector of the dims.
    pub fn to_vec(&self) -> Vec<usize> {
        self.dims.clone()
    }

    /// Returns an iterator over the shape dimensions.
    pub fn iter(&self) -> Iter<'_, usize> {
        self.dims.iter()
    }

    /// Mutable iterator over the dimensions.
    pub fn iter_mut(&mut self) -> IterMut<'_, usize> {
        self.dims.iter_mut()
    }

    /// Borrow the underlying dimensions slice.
    pub fn as_slice(&self) -> &[usize] {
        &self.dims
    }

    /// Borrow the underlying dimensions slice mutably.
    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        &mut self.dims
    }

    /// Insert a dimension of `size` at position `index`.
    pub fn insert(&mut self, index: usize, size: usize) {
        self.dims.insert(index, size);
    }

    /// Remove and return the dimension at position `index` from the shape.
    pub fn remove(&mut self, index: usize) -> usize {
        self.dims.remove(index)
    }

    /// Swap two dimensions in the shape.
    pub fn swap(mut self, dim1: usize, dim2: usize) -> Result<Self, ShapeError> {
        if dim1 > self.rank() {
            return Err(ShapeError::OutOfBounds {
                dim: dim1,
                rank: self.rank(),
            });
        }
        if dim2 > self.rank() {
            return Err(ShapeError::OutOfBounds {
                dim: dim2,
                rank: self.rank(),
            });
        }
        self.dims.swap(dim1, dim2);
        Ok(self)
    }

    /// Reorder the shape dimensions according to the permutation of `axes`.
    pub fn permute(mut self, axes: &[usize]) -> Result<Self, ShapeError> {
        if axes.len() != self.rank() {
            return Err(ShapeError::RankMismatch {
                left: self.rank(),
                right: axes.len(),
            });
        }
        debug_assert!(axes.iter().all(|i| i < &self.rank()));

        self.dims = axes.iter().map(|&i| self.dims[i]).collect();
        Ok(self)
    }

    /// Repeated the specified `dim` a number of `times`.
    pub fn repeat(mut self, dim: usize, times: usize) -> Self {
        self.dims[dim] *= times;
        self
    }

    /// Concatenates all shapes into a new one along the given dimension.
    pub fn cat<'a, I>(shapes: I, dim: usize) -> Result<Self, ShapeError>
    where
        I: IntoIterator<Item = &'a Shape>,
    {
        let mut iter = shapes.into_iter();

        let first = iter.next().ok_or(ShapeError::Empty)?;

        if dim >= first.rank() {
            return Err(ShapeError::OutOfBounds {
                dim,
                rank: first.rank(),
            });
        }

        let mut shape = first.clone();

        for s in iter {
            if s.rank() != shape.rank() {
                return Err(ShapeError::RankMismatch {
                    left: shape.rank(),
                    right: s.rank(),
                });
            }

            if s[..dim] != shape[..dim] || s[dim + 1..] != shape[dim + 1..] {
                return Err(ShapeError::IncompatibleShapes {
                    left: shape.clone(),
                    right: s.clone(),
                });
            }

            shape[dim] += s[dim];
        }

        Ok(shape)
    }

    /// Compute the output shape for binary operations with broadcasting support.
    ///
    /// - Shapes must be of the same rank (missing dimensions are not handled automatically).
    /// - Two dimensions are compatible if they are equal, or one of them is 1.
    ///
    /// For example, a shape `[1, 1, 2, 4]` can be broadcast into `[7, 6, 2, 4]`
    /// because its axes are either equal or 1. On the other hand, a shape `[2, 2]`
    /// can *not* be broadcast into `[2, 4]`.
    pub fn broadcast(&self, other: &Self) -> Result<Self, ShapeError> {
        if self.rank() != other.rank() {
            return Err(ShapeError::RankMismatch {
                left: self.rank(),
                right: other.rank(),
            });
        }

        let mut broadcasted = self.clone();
        for (dim, (lhs, &rhs)) in broadcasted.iter_mut().zip(other.iter()).enumerate() {
            if *lhs != rhs {
                if *lhs == 1 {
                    *lhs = rhs;
                } else if rhs != 1 {
                    return Err(ShapeError::IncompatibleDims {
                        left: *lhs,
                        right: rhs,
                        dim,
                    });
                }
            }
        }

        Ok(broadcasted)
    }
}

impl IntoIterator for Shape {
    type Item = usize;
    type IntoIter = alloc::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.dims.into_iter()
    }
}

impl<Idx> Index<Idx> for Shape
where
    Idx: SliceIndex<[usize]>,
{
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.dims[index]
    }
}

impl<Idx> IndexMut<Idx> for Shape
where
    Idx: SliceIndex<[usize]>,
{
    fn index_mut(&mut self, index: Idx) -> &mut Self::Output {
        &mut self.dims[index]
    }
}

// Allow `&shape` to behave like a slice `&[usize]` directly
impl Deref for Shape {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.dims
    }
}

// Allow `&shape` to behave like a mut slice `&mut [usize]` directly
impl DerefMut for Shape {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.dims
    }
}

// Conversion sugar
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
    fn num_dims_and_rank() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);
        assert_eq!(4, shape.num_dims());
        assert_eq!(4, shape.rank());
    }

    #[test]
    fn num_elements() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);
        assert_eq!(120, shape.num_elements());
    }

    #[test]
    fn test_shape_into_iter() {
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
    fn test_shape_index() {
        let shape = Shape::new([2, 3, 4, 5]);

        assert_eq!(shape[0], 2);
        assert_eq!(shape[1], 3);
        assert_eq!(shape[2], 4);
        assert_eq!(shape[3], 5);

        // Works with ranges
        assert_eq!(shape[1..3], *&[3, 4]);
        assert_eq!(shape[1..=2], *&[3, 4]);
        assert_eq!(shape[..], *&[2, 3, 4, 5]);
    }

    #[test]
    fn test_shape_slice_methods() {
        let shape = Shape::new([2, 3, 4, 5]);

        let dim = shape.first();
        assert_eq!(dim, Some(&2));
        let dim = shape.last();
        assert_eq!(dim, Some(&5));

        assert!(!shape.is_empty());
        let shape = Shape::new([]);
        assert!(shape.is_empty());
    }

    #[test]
    fn test_shape_iter() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);

        for (d, sd) in dims.iter().zip(shape.iter()) {
            assert_eq!(d, sd);
        }
    }

    #[test]
    fn test_shape_iter_mut() {
        let mut shape = Shape::new([2, 3, 4, 5]);

        for d in shape.iter_mut() {
            *d += 1;
        }

        assert_eq!(&shape.dims, &[3, 4, 5, 6]);
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
    fn test_shape_flatten() {
        let shape = Shape::new([2, 3, 4, 5]);
        assert_eq!(shape.num_elements(), 120);

        let shape = shape.flatten();
        assert_eq!(shape.num_elements(), 120);
        assert_eq!(&shape.dims, &[120]);
    }

    #[test]
    fn test_shape_insert_remove() {
        let dims = [2, 3, 4, 5];
        let mut shape = Shape::new(dims);
        let size = 6;
        shape.insert(1, size);

        assert_eq!(&shape.dims, &[2, 6, 3, 4, 5]);

        let removed = shape.remove(1);
        assert_eq!(removed, size);
        assert_eq!(&shape.dims, &dims);
    }

    #[test]
    fn test_shape_swap_permute() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);
        let shape = shape.swap(1, 2).unwrap();

        assert_eq!(&shape.dims, &[2, 4, 3, 5]);

        let shape = shape.permute(&[0, 2, 1, 3]).unwrap();
        assert_eq!(&shape.dims, &dims);
    }

    #[test]
    #[should_panic]
    fn test_shape_swap_out_of_bounds() {
        let shape = Shape::new([2, 3, 4, 5]);

        shape.swap(0, 4).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_shape_permute_incomplete() {
        let shape = Shape::new([2, 3, 4, 5]);

        shape.permute(&[0, 2, 1]).unwrap();
    }

    #[test]
    fn test_shape_repeat() {
        let shape = Shape::new([2, 3, 4, 5]);

        let shape = shape.repeat(2, 3);
        assert_eq!(&shape.dims, &[2, 3, 12, 5]);
    }

    #[test]
    fn test_shape_broadcast_elemwise() {
        let lhs = Shape::new([1, 1, 2, 4]);
        let rhs = Shape::new([7, 6, 2, 1]);

        let out = lhs.broadcast(&rhs).unwrap();
        assert_eq!(&out.dims, &[7, 6, 2, 4]);
    }

    #[test]
    fn test_shape_broadcast_rank_mismatch() {
        let lhs = Shape::new([1, 2, 4]);
        let rhs = Shape::new([7, 6, 2, 4]);

        let out = lhs.broadcast(&rhs);
        assert_eq!(out, Err(ShapeError::RankMismatch { left: 3, right: 4 }));
    }

    #[test]
    fn test_shape_broadcast_incompatible_dims() {
        let lhs = Shape::new([1, 2, 2, 4]);
        let rhs = Shape::new([7, 6, 2, 1]);

        let out = lhs.broadcast(&rhs);
        assert_eq!(
            out,
            Err(ShapeError::IncompatibleDims {
                left: 2,
                right: 6,
                dim: 1
            })
        );
    }

    #[test]
    fn test_shape_cat() {
        let s1 = Shape::new([2, 3, 4, 5]);
        let s2 = Shape::new([1, 3, 4, 5]);
        let s3 = Shape::new([4, 3, 4, 5]);

        let out = Shape::cat(&[s1, s2, s3], 0).unwrap();
        assert_eq!(out, Shape::new([7, 3, 4, 5]));

        let s1 = Shape::new([2, 3, 4, 5]);
        let s2 = Shape::new([2, 3, 2, 5]);
        let s3 = Shape::new([2, 3, 1, 5]);

        let out = Shape::cat(&[s1, s2, s3], 2).unwrap();
        assert_eq!(out, Shape::new([2, 3, 7, 5]));
    }

    #[test]
    fn test_shape_cat_empty() {
        let out = Shape::cat(&[], 0);
        assert_eq!(out, Err(ShapeError::Empty));
    }

    #[test]
    fn test_shape_cat_dim_out_of_bounds() {
        let s1 = Shape::new([2, 3, 4, 5]);
        let s2 = Shape::new([2, 3, 4, 5]);
        let out = Shape::cat(&[s1, s2], 4);
        assert_eq!(out, Err(ShapeError::OutOfBounds { dim: 4, rank: 4 }));
    }

    #[test]
    fn test_shape_cat_rank_mismatch() {
        let s1 = Shape::new([2, 3, 4, 5]);
        let s2 = Shape::new([2, 3, 4, 5, 6]);
        let out = Shape::cat(&[s1, s2], 0);
        assert_eq!(out, Err(ShapeError::RankMismatch { left: 4, right: 5 }));
    }

    #[test]
    fn test_shape_cat_incompatible_shapes() {
        let s1 = Shape::new([2, 3, 4, 5]);
        let s2 = Shape::new([1, 3, 4, 5]);
        let out = Shape::cat(&[s1.clone(), s2.clone()], 1);

        assert_eq!(
            out,
            Err(ShapeError::IncompatibleShapes {
                left: s1,
                right: s2
            })
        );
    }
}
