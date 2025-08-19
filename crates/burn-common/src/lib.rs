#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! # Burn Common Library
//!
//! This library contains common types used by other Burn crates that must be shared.

/// Id module contains types for unique identifiers.
pub mod id;

pub use cubecl_common::*;

#[cfg(feature = "rayon")]
pub use rayon;

extern crate alloc;

/// Network utilities.
#[cfg(feature = "network")]
pub mod network;

/// Parallel utilities.
pub mod parallel;

/// Tensor utilities.
pub mod tensor {
    use alloc::vec::Vec;

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

    /// The result returned by [analyze_reshape()].
    pub enum ReshapeAnalysis {
        /// Updating the strides is sufficient to handle the reshape.
        UpdateStrides {
            /// The new strides.
            strides: Vec<usize>,
        },
        /// The strides are not compatible, we should go through a contiguous layout.
        IntoContiguous,
    }

    /// Returns the proper action to take when reshaping a tensor.
    pub fn analyze_reshape(
        shape: &[usize],
        strides: &[usize],
        shape_new: &[usize],
    ) -> ReshapeAnalysis {
        let shape_rank = shape.len();
        let shape_new_rank = shape_new.len();

        if shape_new_rank < shape_rank {
            return match is_contiguous(&shape, &strides) {
                true => ReshapeAnalysis::UpdateStrides {
                    strides: contiguous_strides(shape_new),
                },
                false => ReshapeAnalysis::IntoContiguous,
            };
        }

        let n_new_batch = shape_new_rank - shape_rank;

        let broadcasted = match n_new_batch > 0 {
            true => {
                shape == &shape_new[n_new_batch..shape_new_rank]
                    && &shape_new[0..n_new_batch] == vec![1; n_new_batch]
            }

            false => {
                return match shape == shape_new || is_contiguous(&shape, &strides) {
                    true => ReshapeAnalysis::UpdateStrides {
                        strides: contiguous_strides(shape_new),
                    },
                    false => ReshapeAnalysis::IntoContiguous,
                };
            }
        };

        if !broadcasted {
            return ReshapeAnalysis::IntoContiguous;
        }

        let num_elems = shape.iter().product::<usize>();
        let mut strides_new = vec![num_elems; shape_new_rank];

        for (i, s) in strides.iter().enumerate() {
            strides_new[i + n_new_batch] = *s;
        }

        ReshapeAnalysis::UpdateStrides {
            strides: strides_new,
        }
    }
}
