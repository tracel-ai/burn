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
    #[must_use]
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
    #[must_use]
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
}
