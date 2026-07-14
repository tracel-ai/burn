#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(clippy::single_range_in_vec_init)]

//! Burn Tch Backend

mod backend;
mod element;
mod ops;
mod tensor;

pub use backend::*;
pub use element::*;
pub use tensor::*;
