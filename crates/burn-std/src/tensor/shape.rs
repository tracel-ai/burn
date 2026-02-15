//! Tensor shape definition.

use super::indexing::ravel_index;
use super::{AsIndex, Slice, SliceArg};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::{Debug, Display, Formatter};
use core::str::FromStr;
use core::{
    ops::{Deref, DerefMut, Index, IndexMut, Range},
    slice::{Iter, IterMut, SliceIndex},
};
use serde::{Deserialize, Serialize};
use smallvec::{SmallVec, smallvec};

pub use crate::errors::ExpressionError;
pub use crate::tensor::index_conversion::AsSize;

/// Shape of a tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape {
    /// The dimensions of the tensor.
    dims: SmallVec<[usize; 5]>,
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
    /// Invalid shape.
    Invalid { reason: String },
}

impl ShapeError {
    fn empty() -> Self {
        Self::Invalid {
            reason: "Shape is empty.".into(),
        }
    }
}

impl Shape {
    /// Constructs a new `Shape`.
    pub fn new<const D: usize>(dims: [usize; D]) -> Self {
        // For backward compat
        Self {
            dims: SmallVec::from_slice(&dims),
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
        self.dims = SmallVec::from_slice(&[self.num_elements()]);
        self
    }

    /// Flatten the shape along a given range of dimensions.
    ///
    /// This function collapses the specified range of dimensions into a single dimension,
    /// effectively flattening the tensor in that range.
    ///
    /// # Arguments
    ///
    /// - `start_dim`: The starting dimension of the range to be flattened,
    ///   supports negative indexing.
    /// - `end_dim`: The ending dimension of the range to be flattened (inclusive),
    ///   supports negative indexing.
    ///
    /// # Returns
    ///
    /// A new `Shape` instance with the specified range of dimensions flattened.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn_std::Shape;
    ///
    /// fn example() {
    ///     let shape = Shape::new([2, 3, 4]);
    ///
    ///     let flattened = shape.flatten_dims(1, 2);
    ///     println!("{flattened}");
    ///     // [2, 12]
    /// }
    /// ```
    pub fn flatten_dims(self, start_dim: impl AsIndex, end_dim: impl AsIndex) -> Self {
        let rank = self.rank();
        let start = start_dim.expect_dim_index(rank);
        let end = end_dim.expect_dim_index(rank);

        assert!(
            start <= end,
            "start_dim ({start}) must be <= than end_dim ({end})"
        );

        let existing = self.dims;

        let flattened_size = existing[start..=end].iter().product();

        let new_rank = rank - (end - start);
        let mut dims = smallvec![0; new_rank];
        dims[..start].copy_from_slice(&existing[..start]);
        dims[start] = flattened_size;
        dims[start + 1..].copy_from_slice(&existing[end + 1..]);

        Self { dims }
    }

    /// Compute the ravel index for the given coordinates.
    ///
    /// This returns the row-major order raveling:
    /// * `strides[-1] = 1`
    /// * `strides[i] = strides[i+1] * dims[i+1]`
    /// * `dim_strides = coords * strides`
    /// * `ravel = sum(dim_strides)`
    ///
    /// # Arguments
    /// - `indices`: the index for each dimension; must be the same length as `shape`.
    ///
    /// # Returns
    /// - the ravel offset index.
    pub fn ravel_index<I: AsIndex>(&self, indices: &[I]) -> usize {
        ravel_index(indices, &self.dims)
    }

    /// Convert shape dimensions to full covering ranges (0..dim) for each dimension.
    pub fn into_ranges(self) -> Vec<Range<usize>> {
        self.iter().map(|&d| 0..d).collect()
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
    pub fn into_slices<S>(self, slices: S) -> Vec<Slice>
    where
        S: SliceArg,
    {
        slices.into_slices(&self)
    }

    /// Construct a vector of the dims.
    pub fn to_vec(&self) -> Vec<usize> {
        self.dims.to_vec()
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

    /// Appends a dimension of `size` to the back of the shape.
    pub fn push(&mut self, size: usize) {
        self.dims.push(size)
    }

    /// Extend the shape with the content of another shape or iterator.
    pub fn extend(&mut self, iter: impl IntoIterator<Item = usize>) {
        self.dims.extend(iter)
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
    pub fn repeat(mut self, dim: usize, times: usize) -> Result<Shape, ShapeError> {
        if dim >= self.rank() {
            return Err(ShapeError::OutOfBounds {
                dim,
                rank: self.rank(),
            });
        }

        self.dims[dim] *= times;
        Ok(self)
    }

    /// Returns a new shape where the specified `dim` is reduced to size 1.
    pub fn reduce(mut self, dim: usize) -> Result<Shape, ShapeError> {
        if dim >= self.rank() {
            return Err(ShapeError::OutOfBounds {
                dim,
                rank: self.rank(),
            });
        }

        self.dims[dim] = 1;
        Ok(self)
    }

    /// Concatenates all shapes into a new one along the given dimension.
    pub fn cat<'a, I>(shapes: I, dim: usize) -> Result<Self, ShapeError>
    where
        I: IntoIterator<Item = &'a Shape>,
    {
        let mut iter = shapes.into_iter();

        let first = iter.next().ok_or(ShapeError::empty())?;

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

    /// Compute the output shape from the given slices.
    pub fn slice(mut self, slices: &[Slice]) -> Result<Self, ShapeError> {
        if slices.len() > self.rank() {
            return Err(ShapeError::RankMismatch {
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

    /// Compute the output shape for binary operations with broadcasting support.
    ///
    /// - Shapes must be of the same rank (missing dimensions are not handled automatically).
    /// - Two dimensions are compatible if they are equal, or one of them is 1.
    ///
    /// For example, a shape `[1, 1, 2, 4]` can be broadcast into `[7, 6, 2, 4]`
    /// because its axes are either equal or 1. On the other hand, a shape `[2, 2]`
    /// can *not* be broadcast into `[2, 4]`.
    pub fn broadcast(&self, other: &Self) -> Result<Self, ShapeError> {
        Self::broadcast_many([self, other])
    }

    /// Compute the broadcasted output shape across multiple input shapes.
    ///
    /// See also [broadcast](Self::broadcast).
    pub fn broadcast_many<'a, I>(shapes: I) -> Result<Self, ShapeError>
    where
        I: IntoIterator<Item = &'a Shape>,
    {
        let mut iter = shapes.into_iter();
        let mut broadcasted = iter.next().ok_or(ShapeError::empty())?.clone();
        let rank = broadcasted.rank();

        for shape in iter {
            if shape.rank() != rank {
                return Err(ShapeError::RankMismatch {
                    left: rank,
                    right: shape.rank(),
                });
            }

            for (dim, (d_lhs, &d_rhs)) in broadcasted.iter_mut().zip(shape.iter()).enumerate() {
                match (*d_lhs, d_rhs) {
                    (a, b) if a == b => {} // same
                    (1, b) => *d_lhs = b,  // broadcast to rhs
                    (_a, 1) => {}          // keep existing dimension
                    _ => {
                        return Err(ShapeError::IncompatibleDims {
                            left: *d_lhs,
                            right: d_rhs,
                            dim,
                        });
                    }
                }
            }
        }

        Ok(broadcasted)
    }

    /// Expand this shape to match the target shape, following broadcasting rules.
    pub fn expand(&self, target: Shape) -> Result<Shape, ShapeError> {
        let target_rank = target.rank();
        if self.rank() > target_rank {
            return Err(ShapeError::RankMismatch {
                left: self.rank(),
                right: target_rank,
            });
        }

        for (i, (dim_target, dim_self)) in target.iter().rev().zip(self.iter().rev()).enumerate() {
            if dim_self != dim_target && *dim_self != 1 {
                return Err(ShapeError::IncompatibleDims {
                    left: *dim_self,
                    right: *dim_target,
                    dim: target_rank - i - 1,
                });
            }
        }

        Ok(target)
    }

    /// Reshape this shape to the target shape.
    pub fn reshape<A, T>(&self, args: A) -> Result<Shape, ShapeError>
    where
        A: AsRef<[T]> + Debug,
        T: AsIndex,
    {
        let args = args.as_ref();
        let mut infer_index = None;
        let mut dims = Vec::new();

        let mut new_size = 1;

        for (idx, &s) in args.iter().enumerate() {
            let s = s.as_index();
            if s > 0 {
                let s = s as usize;
                new_size *= s;
                dims.push(s);
            } else if s == 0 {
                // We need to find the index of the 0 dimensions and
                // replace them with the actual dimension value.
                let s = self.dims[idx];
                new_size *= s;
                dims.push(s);
            } else if s == -1 {
                match infer_index {
                    None => {
                        infer_index = Some(idx);
                        // Used by / Replaced by handling later.
                        dims.push(1);
                    }
                    Some(_) => {
                        return Err(ShapeError::Invalid {
                            reason: "Repeated -1 in reshape".to_string(),
                        });
                    }
                }
            } else {
                return Err(ShapeError::Invalid {
                    reason: "The given shape cannot contain negative dimensions (other than -1)."
                        .to_string(),
                });
            }
        }

        let source_size = self.num_elements();
        match infer_index {
            None => {
                if source_size != new_size {
                    return Err(ShapeError::Invalid {
                        reason: format!(
                            "The given shape doesn't have the same number of elements as the current shape. Current shape: {self}, target shape: {dims:?}.",
                        ),
                    });
                }
            }
            Some(idx) => {
                if !source_size.is_multiple_of(new_size) {
                    return Err(ShapeError::Invalid {
                        reason: format!(
                            "Cannot infer a valid target shape. Current shape: {self}, target dimensions: {args:?}."
                        ),
                    });
                }
                dims[idx] = source_size / new_size;
            }
        }

        Ok(dims.into())
    }

    /// Returns the raw inner storage of the shape. Avoid using this where possible, since the storage
    /// may change at any time.
    pub fn inner_mut(&mut self) -> &mut SmallVec<[usize; 5]> {
        &mut self.dims
    }
}

/// Compute the output shape for matrix multiplication with broadcasting support.
///
/// The last two dimensions are treated as matrices, while preceding dimensions
/// follow broadcast semantics similar to elementwise operations.
pub fn calculate_matmul_output(lhs: &Shape, rhs: &Shape) -> Result<Shape, ShapeError> {
    let rank = lhs.rank();
    if rank != rhs.rank() {
        return Err(ShapeError::RankMismatch {
            left: rank,
            right: rhs.rank(),
        });
    }

    if lhs[rank - 1] != rhs[rank - 2] {
        return Err(ShapeError::IncompatibleShapes {
            left: lhs.clone(),
            right: rhs.clone(),
        });
    }

    let mut shape = if rank > 2 {
        // Broadcast leading dims
        Shape::from(&lhs[..rank - 2]).broadcast(&Shape::from(&rhs[..rank - 2]))?
    } else {
        Shape::new([])
    };
    shape.extend([lhs[rank - 2], rhs[rank - 1]]);

    Ok(shape)
}

impl Display for Shape {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        self.dims.fmt(f)
    }
}

impl FromStr for Shape {
    type Err = crate::ExpressionError;

    fn from_str(source: &str) -> Result<Self, Self::Err> {
        let mut s = source.trim();

        const DELIMS: [(&str, &str); 2] = [("[", "]"), ("(", ")")];

        for (open, close) in DELIMS {
            if let Some(p) = s.strip_prefix(open) {
                if let Some(p) = p.strip_suffix(close) {
                    s = p.trim();
                    break;
                } else {
                    return Err(crate::ExpressionError::ParseError {
                        message: "Unbalanced delimiters".to_string(),
                        source: source.to_string(),
                    });
                }
            }
        }

        if s.is_empty() {
            return Ok(Shape::new([]));
        }

        let dims =
            s.split(',')
                .map(|dim_str| {
                    dim_str.trim().parse::<usize>().map_err(|_| {
                        crate::ExpressionError::ParseError {
                            message: "Unable to parse shape".to_string(),
                            source: source.to_string(),
                        }
                    })
                })
                .collect::<Result<SmallVec<_>, crate::ExpressionError>>()?;

        if dims.is_empty() {
            unreachable!("Split should have returned at least one element");
        }

        Ok(Shape { dims })
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
// Allow `shape.reshape(other_shape)`.
//
// By implementing `AsRef<[usize]>`, `Shape` behaves like a slice of dimensions,
// similar to how `Vec<T>` can be passed to functions expecting a slice.
impl AsRef<[usize]> for Shape {
    fn as_ref(&self) -> &[usize] {
        &self.dims
    }
}

impl From<Shape> for Vec<usize> {
    fn from(shape: Shape) -> Self {
        shape.dims.to_vec()
    }
}

impl<T, I> From<T> for Shape
where
    T: IntoIterator<Item = I>,
    I: AsSize,
{
    fn from(dims: T) -> Self {
        Shape {
            dims: dims.into_iter().map(|d| d.as_size()).collect(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::identity_op, reason = "useful for clarity")]
mod tests {
    use super::*;
    use crate::s;
    use alloc::string::ToString;
    use alloc::vec;

    #[test]
    fn test_shape_to_str() {
        let shape = Shape::new([2, 3, 4, 5]);
        assert_eq!(shape.to_string(), "[2, 3, 4, 5]");
    }

    #[test]
    fn test_shape_from_str() {
        assert_eq!(
            "[2, 3, 4, 5]".parse::<Shape>().unwrap(),
            Shape::new([2, 3, 4, 5])
        );
        assert_eq!(
            "(2, 3, 4, 5)".parse::<Shape>().unwrap(),
            Shape::new([2, 3, 4, 5])
        );
        assert_eq!(
            "2, 3, 4, 5".parse::<Shape>().unwrap(),
            Shape::new([2, 3, 4, 5])
        );

        assert_eq!("[2]".parse::<Shape>().unwrap(), Shape::new([2]));
        assert_eq!("(2)".parse::<Shape>().unwrap(), Shape::new([2]));
        assert_eq!("2".parse::<Shape>().unwrap(), Shape::new([2]));

        assert_eq!("[]".parse::<Shape>().unwrap(), Shape::new([]));
        assert_eq!("".parse::<Shape>().unwrap(), Shape::new([]));

        assert_eq!(
            "[".parse::<Shape>(),
            Err(crate::ExpressionError::ParseError {
                message: "Unbalanced delimiters".to_string(),
                source: "[".to_string()
            })
        );

        assert_eq!(
            "[[1]".parse::<Shape>(),
            Err(crate::ExpressionError::ParseError {
                message: "Unable to parse shape".to_string(),
                source: "[[1]".to_string()
            })
        );
        assert_eq!(
            "[[1]]".parse::<Shape>(),
            Err(crate::ExpressionError::ParseError {
                message: "Unable to parse shape".to_string(),
                source: "[[1]]".to_string()
            })
        );
        assert_eq!(
            "[1)".parse::<Shape>(),
            Err(crate::ExpressionError::ParseError {
                message: "Unbalanced delimiters".to_string(),
                source: "[1)".to_string()
            })
        );

        assert_eq!(
            "]".parse::<Shape>(),
            Err(crate::ExpressionError::ParseError {
                message: "Unable to parse shape".to_string(),
                source: "]".to_string()
            })
        );

        assert_eq!(
            "[a]".parse::<Shape>(),
            Err(crate::ExpressionError::ParseError {
                message: "Unable to parse shape".to_string(),
                source: "[a]".to_string()
            })
        );
    }

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
    #[allow(clippy::into_iter_on_ref)]
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

        assert_eq!(shape.as_slice(), &[3, 4, 5, 6]);
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
        assert_eq!(shape.as_slice(), &[120]);
    }

    #[test]
    fn test_ravel() {
        let shape = Shape::new([2, 3, 4, 5]);

        assert_eq!(shape.ravel_index(&[0, 0, 0, 0]), 0);
        assert_eq!(
            shape.ravel_index(&[1, 2, 3, 4]),
            1 * (3 * 4 * 5) + 2 * (4 * 5) + 3 * 5 + 4
        );
    }

    #[test]
    fn test_shape_insert_remove_push() {
        let dims = [2, 3, 4, 5];
        let mut shape = Shape::new(dims);
        let size = 6;
        shape.insert(1, size);

        assert_eq!(shape, Shape::new([2, 6, 3, 4, 5]));

        let removed = shape.remove(1);
        assert_eq!(removed, size);
        assert_eq!(shape, Shape::new(dims));

        shape.push(6);
        assert_eq!(shape, Shape::new([2, 3, 4, 5, 6]));
    }

    #[test]
    fn test_shape_swap_permute() {
        let dims = [2, 3, 4, 5];
        let shape = Shape::new(dims);
        let shape = shape.swap(1, 2).unwrap();

        assert_eq!(shape.as_slice(), &[2, 4, 3, 5]);

        let shape = shape.permute(&[0, 2, 1, 3]).unwrap();
        assert_eq!(shape, Shape::new(dims));
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

        let out = shape.repeat(2, 3).unwrap();
        assert_eq!(out, Shape::new([2, 3, 12, 5]));
    }

    #[test]
    fn test_shape_repeat_invalid() {
        let shape = Shape::new([2, 3, 4, 5]);

        let out = shape.repeat(5, 3);
        assert_eq!(out, Err(ShapeError::OutOfBounds { dim: 5, rank: 4 }));
    }

    #[test]
    fn test_shape_reduce() {
        let shape = Shape::new([2, 3, 4, 5]);

        let out = shape.reduce(2).unwrap();
        assert_eq!(out, Shape::new([2, 3, 1, 5]));
    }

    #[test]
    fn test_shape_reduce_invalid() {
        let shape = Shape::new([2, 3, 4, 5]);

        let out = shape.reduce(5);
        assert_eq!(out, Err(ShapeError::OutOfBounds { dim: 5, rank: 4 }));
    }

    #[test]
    fn test_shape_broadcast_binary() {
        let lhs = Shape::new([1, 1, 2, 4]);
        let rhs = Shape::new([7, 6, 2, 1]);

        let out = lhs.broadcast(&rhs).unwrap();
        assert_eq!(out, Shape::new([7, 6, 2, 4]));
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
    fn test_shape_broadcast_many() {
        let s1 = Shape::new([1, 1, 2, 4]);
        let s2 = Shape::new([7, 1, 2, 1]);
        let s3 = Shape::new([7, 6, 1, 1]);

        let out = Shape::broadcast_many([&s1, &s2, &s3]).unwrap();
        assert_eq!(out, Shape::new([7, 6, 2, 4]));
    }

    #[test]
    fn test_shape_broadcast_many_rank_mismatch() {
        let s1 = Shape::new([1, 1, 2, 4]);
        let s2 = Shape::new([7, 1, 2, 1]);
        let s3 = Shape::new([1, 6, 1]);

        let out = Shape::broadcast_many([&s1, &s2, &s3]);
        assert_eq!(out, Err(ShapeError::RankMismatch { left: 4, right: 3 }));
    }

    #[test]
    fn test_shape_broadcast_many_incompatible_dims() {
        let s1 = Shape::new([1, 1, 2, 4]);
        let s2 = Shape::new([7, 1, 2, 1]);
        let s3 = Shape::new([4, 6, 1, 1]);

        let out = Shape::broadcast_many([&s1, &s2, &s3]);
        assert_eq!(
            out,
            Err(ShapeError::IncompatibleDims {
                left: 7,
                right: 4,
                dim: 0
            })
        );
    }

    #[test]
    fn test_shape_broadcast_many_empty() {
        let out = Shape::broadcast_many(&[]);
        assert_eq!(out, Err(ShapeError::empty()));
    }

    #[test]
    fn test_shape_matmul_2d() {
        let lhs = Shape::new([2, 4]);
        let rhs = Shape::new([4, 2]);
        let out = calculate_matmul_output(&lhs, &rhs).unwrap();
        assert_eq!(out, Shape::new([2, 2]));
    }

    #[test]
    fn test_shape_matmul_4d_broadcasted() {
        let lhs = Shape::new([1, 3, 2, 4]);
        let rhs = Shape::new([2, 1, 4, 2]);
        let out = calculate_matmul_output(&lhs, &rhs).unwrap();
        assert_eq!(out, Shape::new([2, 3, 2, 2]));
    }

    #[test]
    fn test_shape_matmul_invalid_rank() {
        let lhs = Shape::new([3, 2, 4]);
        let rhs = Shape::new([2, 1, 4, 2]);
        let out = calculate_matmul_output(&lhs, &rhs);
        assert_eq!(out, Err(ShapeError::RankMismatch { left: 3, right: 4 }));
    }

    #[test]
    fn test_shape_matmul_invalid_shape() {
        let lhs = Shape::new([1, 3, 2, 4]);
        let rhs = Shape::new([2, 1, 3, 2]);
        let out = calculate_matmul_output(&lhs, &rhs);
        assert_eq!(
            out,
            Err(ShapeError::IncompatibleShapes {
                left: lhs,
                right: rhs
            })
        );
    }

    #[test]
    fn test_shape_matmul_invalid_broadcast() {
        let lhs = Shape::new([1, 3, 2, 4]);
        let rhs = Shape::new([2, 2, 4, 2]);
        let out = calculate_matmul_output(&lhs, &rhs);
        assert_eq!(
            out,
            Err(ShapeError::IncompatibleDims {
                left: 3,
                right: 2,
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
        assert_eq!(out, Err(ShapeError::empty()));
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

    #[test]
    fn test_shape_expand() {
        let shape = Shape::new([1, 3, 1]);
        let expanded = Shape::new([2, 3, 4]);
        let out = shape.expand(expanded.clone()).unwrap();
        assert_eq!(out, expanded);
    }

    #[test]
    fn test_shape_expand_higher_rank() {
        let shape = Shape::new([1, 4]);
        let expanded = Shape::new([2, 3, 4]);
        let out = shape.expand(expanded.clone()).unwrap();
        assert_eq!(out, expanded);
    }

    #[test]
    fn test_shape_expand_invalid_rank() {
        let shape = Shape::new([1, 3, 1]);
        let expanded = Shape::new([3, 4]);
        let out = shape.expand(expanded);
        assert_eq!(out, Err(ShapeError::RankMismatch { left: 3, right: 2 }));
    }

    #[test]
    fn test_shape_expand_incompatible_dims() {
        let shape = Shape::new([1, 3, 2]);
        let expanded = Shape::new([2, 3, 4]);
        let out = shape.expand(expanded);
        assert_eq!(
            out,
            Err(ShapeError::IncompatibleDims {
                left: 2,
                right: 4,
                dim: 2
            })
        );
    }

    #[test]
    fn test_shape_reshape() {
        let shape = Shape::new([2, 3, 4, 5]);
        let reshaped = Shape::new([1, 2, 12, 5]);
        let out = shape.reshape(reshaped.clone()).unwrap();
        assert_eq!(out, reshaped);
    }

    #[test]
    fn test_shape_reshape_invalid() {
        let shape = Shape::new([2, 3, 4, 5]);
        let reshaped = Shape::new([2, 2, 12, 5]);
        let out = shape.reshape(reshaped.clone());
        assert_eq!(
            out,
            Err(ShapeError::Invalid {
                reason: "The given shape doesn't have the same number of elements as the current shape. Current shape: [2, 3, 4, 5], target shape: [2, 2, 12, 5].".into(),
            })
        );
    }

    #[test]
    fn test_shape_reshape_invalid_inferred() {
        let shape = Shape::new([2, 4]);
        let out = shape.reshape([-1, 3]);
        assert_eq!(
            out,
            Err(ShapeError::Invalid {
                reason: "Cannot infer a valid target shape. Current shape: [2, 4], target dimensions: [-1, 3].".into(),
            })
        );
    }

    #[test]
    fn test_flatten_dims() {
        let shape = Shape::new([2, 3, 4, 5]);
        let flattened = shape.flatten_dims(-2, 3);
        assert_eq!(flattened, Shape::new([2, 3, 20]));
    }
}
