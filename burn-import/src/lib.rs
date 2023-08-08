#![warn(missing_docs)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::single_match)]
#![allow(clippy::upper_case_acronyms)]

//! `burn-import` is a crate designed to simplify the process of importing models trained in other
//! machine learning frameworks into the Burn framework. This tool generates a Rust source file that
//! aligns the imported model with Burn's model and converts tensor data into a format compatible with
//! Burn.

#[macro_use]
extern crate derive_new;

/// The onnx module.
#[cfg(feature = "onnx")]
pub mod onnx;

/// The module for generating the burn code.
pub mod burn;

mod formatter;
mod logger;
pub use formatter::*;
