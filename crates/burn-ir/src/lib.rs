#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

//! Burn intermediate representation.

extern crate alloc;

mod backend;
mod handle;
mod operation;
mod tensor;

pub use backend::*;
pub use handle::*;
pub use operation::*;
pub use tensor::*;
