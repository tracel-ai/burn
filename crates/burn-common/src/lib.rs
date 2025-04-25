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
    /// Check if the current tensor is contiguous.
    ///
    /// A tensor is considered contiguous if its elements are stored in memory
    /// such that the strides are in strictly decreasing order, and the stride at
    /// position `k` is equal to the product of the shapes of all dimensions greater
    /// than `k`. Axes with a shape of 1 are ignored.
    pub fn is_contiguous(shape: &[usize], strides: &[usize]) -> bool {
        if shape.is_empty() {
            return true;
        }

        let mut prev_stride = 1;
        let mut current_num_elems_shape = 1;

        for (i, (&stride, &shape)) in strides.iter().zip(shape).rev().enumerate() {
            if shape == 1 {
                continue;
            }

            if i > 0 {
                if current_num_elems_shape != stride {
                    return false;
                }

                if prev_stride > stride {
                    return false;
                }
            } else if stride != 1 {
                return false;
            }

            current_num_elems_shape *= shape;
            prev_stride = stride;
        }

        true
    }
}
