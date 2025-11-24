#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! This library provides the core types that define how Burn tensor data is represented, stored, and interpreted.

#[macro_use]
extern crate derive_new;

extern crate alloc;

mod data;
pub use data::*;

pub mod distribution;
pub mod element;

// /// Quantization data representation.
// pub mod quantization;
