#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! This library provides the core abstractions required to run tensor operations with Burn.
//! `Tensor`s are generic over the backend to allow users to perform operations using different `Backend` implementations.
//! Burn's tensors also support auto-differentiation thanks to the `AutodiffBackend` trait.

#[macro_use]
extern crate derive_new;

extern crate alloc;

mod tensor;

#[cfg(feature = "export_tests")]
#[allow(missing_docs)]
pub mod tests;

#[cfg(feature = "export_tests")]
// Re-export the might_panic proc macro for easy access
pub use burn_tensor_testgen::might_panic;

pub(crate) use tensor::check::macros::check;
pub use tensor::*;

pub use burn_std::stream_id::StreamId;

// Re-exported types
pub use burn_std::reader::*; // Useful so that backends don't have to add `burn_std` as a dependency.
pub use burn_std::{Bytes, bf16, f16};
