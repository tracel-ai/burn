#![allow(clippy::ptr_arg)]
#![allow(clippy::single_match)]
#![allow(clippy::upper_case_acronyms)]

#[macro_use]
extern crate derive_new;

#[cfg(feature = "onnx")]
pub mod onnx;

pub mod burn;

mod formater;
pub use formater::*;
