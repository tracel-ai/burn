#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Burn ndarray backend.

#[cfg(any(
    feature = "blas-netlib",
    feature = "blas-openblas",
    feature = "blas-openblas-system",
))]
extern crate blas_src;

mod backend;
mod element;
mod ops;
mod parallel;
mod rand;
mod sharing;
mod storage;
mod tensor;

pub use backend::*;
pub use element::*;
pub(crate) use sharing::*;
pub(crate) use storage::*;
pub use tensor::*;

extern crate alloc;
