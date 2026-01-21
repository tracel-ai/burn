pub mod dtype;
pub mod index_conversion;
pub mod indexing;
pub mod quantization;
pub mod shape;
pub mod slice;

pub use dtype::*;
pub use index_conversion::*;
pub use indexing::*;
pub use quantization::*;
pub use shape::*;
pub use slice::*;

use alloc::{vec, vec::Vec};

/// Check if the current tensor is contiguous.
///
/// A tensor is considered contiguous if its elements are stored in memory
/// such that the stride at position `k` is equal to the product of the shapes
/// of all dimensions greater than `k`.
///
/// This means that strides increase as you move from the rightmost to the leftmost dimension.
pub fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
    if shape.is_empty() {
        return true;
    }

    for (expected, &stride) in contiguous_strides(shape).into_iter().zip(strides) {
        if expected != stride {
            return false;
        }
    }

    true
}

/// Computes the strides for a contiguous tensor with the given shape.
///
/// In a contiguous row-major tensor, the stride for each dimension
/// equals the product of all dimension sizes to its right.
pub fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = Vec::with_capacity(shape.len());
    let mut current = 1;

    for &dim in shape.iter().rev() {
        strides.push(current);
        current *= dim;
    }

    strides.reverse();
    strides
}

/// The action to take for a reshape operation.
#[derive(Debug)]
pub enum ReshapeAction {
    /// Updating the strides is sufficient to handle the reshape.
    UpdateStrides {
        /// The new strides.
        strides: Vec<usize>,
    },
    /// The strides are not compatible, we should recompute the buffer.
    Recompute,
    /// The strides are already correct.
    NoChange,
}

/// The reshape kind.
#[derive(Debug)]
pub enum ReshapeAnalysis {
    /// Original tensor is contiguous, can update the strides.
    IsContiguous,
    /// Original tensor is highly permutated, can't update the strides.
    HighlyPermuted,
    /// Only batch dimensions are added, can update the strides.
    Broadcasted,
    /// Dimensions are only split, can update the strides.
    Split,
    /// Original tensor is bigger than output shape.
    SmallerRank,
    /// New shape is the same.
    NoChange,
}

impl ReshapeAnalysis {
    /// Returns the proper action to take for the current analysis.
    fn action(self, shape: &[usize], strides: &[usize], shape_new: &[usize]) -> ReshapeAction {
        match self {
            ReshapeAnalysis::IsContiguous => ReshapeAction::UpdateStrides {
                strides: contiguous_strides(shape_new),
            },
            ReshapeAnalysis::NoChange => ReshapeAction::NoChange,
            ReshapeAnalysis::HighlyPermuted | ReshapeAnalysis::SmallerRank => {
                ReshapeAction::Recompute
            }
            ReshapeAnalysis::Broadcasted => {
                let shape_rank = shape.len();
                let shape_new_rank = shape_new.len();
                let n_new_batch = shape_new_rank - shape_rank;
                let num_elems = shape.iter().product::<usize>();
                let strides_new = broadcast_strides(n_new_batch, shape_rank, num_elems, strides);

                ReshapeAction::UpdateStrides {
                    strides: strides_new,
                }
            }
            ReshapeAnalysis::Split => {
                let strides_new = split_strides(shape, strides, shape_new);

                ReshapeAction::UpdateStrides {
                    strides: strides_new,
                }
            }
        }
    }
}

/// Returns the proper action to take when reshaping a tensor.
pub fn reshape_action(shape: &[usize], strides: &[usize], shape_new: &[usize]) -> ReshapeAction {
    reshape_analysis(shape, Some(strides), shape_new).action(shape, strides, shape_new)
}

/// Calculate the new strides given added batch dimensions.
pub fn broadcast_strides(
    n_new_batch: usize,
    rank_prev: usize,
    num_elems: usize,
    strides: &[usize],
) -> Vec<usize> {
    let mut strides_new = vec![num_elems; rank_prev + n_new_batch];

    for (i, s) in strides.iter().enumerate() {
        strides_new[i + n_new_batch] = *s;
    }

    strides_new
}

/// Calculate the new strides given added split dimensions.
pub fn split_strides(shape: &[usize], strides: &[usize], shape_new: &[usize]) -> Vec<usize> {
    let mut strides_new = vec![1; shape_new.len()];

    let mut old_idx = shape.len() - 1;
    let mut current_stride = strides[old_idx];
    let mut dim_prod = 1;

    for (i, dim) in shape_new.iter().enumerate().rev() {
        dim_prod *= *dim;
        strides_new[i] = current_stride;
        if dim_prod == shape[old_idx] {
            old_idx -= 1;
            current_stride = strides[old_idx];
            dim_prod = 1;
        } else {
            current_stride *= *dim;
        }
    }

    strides_new
}

/// Returns the analysis of a reshape operation.
pub fn reshape_analysis(
    shape: &[usize],
    strides: Option<&[usize]>,
    shape_new: &[usize],
) -> ReshapeAnalysis {
    let shape_rank = shape.len();
    let shape_new_rank = shape_new.len();

    let is_contiguous = match strides {
        Some(strides) => is_contiguous(shape, strides),
        None => false,
    };

    if is_contiguous {
        return ReshapeAnalysis::IsContiguous;
    }

    if shape_new_rank < shape_rank {
        return ReshapeAnalysis::SmallerRank;
    }

    let n_new_batch = shape_new_rank - shape_rank;

    match n_new_batch > 0 {
        true => {
            if shape == &shape_new[n_new_batch..shape_new_rank]
                && shape_new[0..n_new_batch] == vec![1; n_new_batch]
            {
                return ReshapeAnalysis::Broadcasted;
            } else {
                let mut dim_prod = 1;
                let mut old_idx = 0;
                for dim in shape_new {
                    dim_prod *= *dim;
                    if dim_prod == shape[old_idx] {
                        dim_prod = 1;
                        old_idx += 1;
                    } else if dim_prod > shape[old_idx] {
                        return ReshapeAnalysis::HighlyPermuted;
                    }
                }
                return ReshapeAnalysis::Split;
            }
        }

        false => {
            if shape == shape_new {
                return ReshapeAnalysis::NoChange;
            }
        }
    };

    ReshapeAnalysis::HighlyPermuted
}
