#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(unused)] // TODO remove when backend filled

//! Burn Candle Backend

#[macro_use]
extern crate derive_new;

mod backend;
mod element;
mod ops;
mod tensor;

pub use backend::*;
pub use element::*;
pub use tensor::*;
