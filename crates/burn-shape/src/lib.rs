#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! This library provides shape and indexing utilities for Burn tensors.

extern crate alloc;

mod indexing;
mod shape;
mod slice;

pub use indexing::*;
pub use shape::*;
pub use slice::*;
