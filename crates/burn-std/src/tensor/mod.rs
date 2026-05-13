/// Generic container for storing tensors keyed by an id.
pub mod container;
/// Tensor data type definitions.
pub mod dtype;
/// Quantization data representation.
pub mod quantization;
/// Tensor shape utilities.
pub mod shape;
/// Tensor slicing utilities.
pub mod slice;

pub use dtype::*;
pub use quantization::*;
pub use shape::*;
pub use slice::*;

pub use cubecl_zspace::indexing::{self, *};
pub use cubecl_zspace::{Strides, metadata::Metadata, strides};

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

    for (&expected, &stride) in contiguous_strides(shape).iter().zip(strides) {
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
pub fn contiguous_strides(shape: &[usize]) -> Strides {
    let mut strides = strides![0; shape.len()];
    let mut current = 1;

    for (i, &dim) in shape.iter().enumerate().rev() {
        strides[i] = current;
        current *= dim;
    }

    strides
}

/// The action to take for a reshape operation.
#[derive(Debug)]
pub enum ReshapeAction {
    /// Updating the strides is sufficient to handle the reshape.
    UpdateStrides {
        /// The new strides.
        strides: Strides,
    },
    /// The strides are not compatible, we should recompute the buffer.
    Recompute,
    /// The strides are already correct.
    NoChange,
}

/// The reshape kind.
#[derive(Debug, PartialEq)]
pub enum ReshapeAnalysis {
    /// Original tensor is contiguous, can update the strides.
    IsContiguous,
    /// Original tensor is highly permuted, can't update the strides.
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
    pub fn action(&self, shape: &[usize], strides: &[usize], shape_new: &[usize]) -> ReshapeAction {
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
pub fn reshape_action(shape: &Shape, strides: &Strides, shape_new: &Shape) -> ReshapeAction {
    reshape_analysis(shape, Some(strides), shape_new).action(shape, strides, shape_new)
}

/// Calculate the new strides given added batch dimensions.
pub fn broadcast_strides(
    n_new_batch: usize,
    rank_prev: usize,
    num_elems: usize,
    strides: &[usize],
) -> Strides {
    let mut strides_new = strides![num_elems; rank_prev + n_new_batch];

    for (i, s) in strides.iter().enumerate() {
        strides_new[i + n_new_batch] = *s;
    }

    strides_new
}

/// Calculate the new strides given added split dimensions.
pub fn split_strides(shape: &[usize], strides: &[usize], shape_new: &[usize]) -> Strides {
    let mut strides_new = strides![1; shape_new.len()];

    let mut old_idx = shape.len() - 1;
    let mut current_stride = strides[old_idx];
    let mut dim_prod = 1;

    for (i, dim) in shape_new.iter().enumerate().rev() {
        dim_prod *= *dim;
        strides_new[i] = current_stride;
        if *dim == 1 {
            continue;
        } else if dim_prod == shape[old_idx] {
            old_idx = old_idx.saturating_sub(1);
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
    shape: &Shape,
    strides: Option<&Strides>,
    shape_new: &Shape,
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
            if shape.as_ref() == &shape_new[n_new_batch..shape_new_rank]
                && shape_new[0..n_new_batch].iter().all(|it| *it == 1)
            {
                return ReshapeAnalysis::Broadcasted;
            } else {
                let mut dim_prod = 1;
                let mut old_idx = 0;
                for dim in shape_new.iter() {
                    dim_prod *= *dim;

                    // We need to ignore unit dims because they don't affect analysis and break
                    // things because they match the default `dim_prod`. If we don't do this,
                    // reshapes like [2, 3] to [2, 3, 1] will panic from out of bounds access.
                    if *dim == 1 {
                        continue;
                    } else if dim_prod == shape[old_idx] {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reshape_analysis_is_contiguous() {
        let analysis = reshape_analysis(
            &[32, 1, 1, 1].into(),
            Some(&[1, 1, 1, 1].into()),
            &[1, 1, 32, 1, 1, 1].into(),
        );

        assert_eq!(analysis, ReshapeAnalysis::IsContiguous)
    }

    #[test]
    fn test_reshape_analysis_is_contiguous_2() {
        let analysis = reshape_analysis(
            &[32, 1, 1, 8].into(),
            Some(&[8, 8, 8, 1].into()),
            &[1, 1, 32, 1, 1, 8].into(),
        );

        assert_eq!(analysis, ReshapeAnalysis::IsContiguous)
    }

    #[test]
    fn test_reshape_analysis_broadcasted_batch() {
        let analysis = reshape_analysis(
            &[32, 1, 1, 1].into(),
            Some(&[1, 32, 32, 32].into()),
            &[1, 1, 32, 1, 1, 1].into(),
        );

        assert_eq!(analysis, ReshapeAnalysis::Broadcasted)
    }

    #[test]
    fn test_reshape_analysis_unsqueeze_split() {
        // Unsqueeze
        let analysis = reshape_analysis(
            &[32, 1, 1, 1].into(),
            Some(&[1, 32, 32, 32].into()),
            &[32, 1, 1, 1, 1].into(),
        );

        assert_eq!(analysis, ReshapeAnalysis::Split)
    }

    #[test]
    fn test_reshape_analysis_split() {
        let analysis = reshape_analysis(
            &[32, 1, 1, 1].into(),
            Some(&[1, 32, 32, 32].into()),
            &[4, 8, 1, 1, 1].into(),
        );

        assert_eq!(analysis, ReshapeAnalysis::Split)
    }
}
