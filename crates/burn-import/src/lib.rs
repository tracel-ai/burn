#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(clippy::ptr_arg)]
#![allow(clippy::single_match)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::approx_constant)]

//! `burn-import` is a crate designed to simplify the process of importing models trained in other
//! machine learning frameworks into the Burn framework. This tool generates a Rust source file that
//! aligns the imported model with Burn's model and converts tensor data into a format compatible with
//! Burn.

#[cfg(any(feature = "pytorch", feature = "onnx", feature = "safetensors"))]
#[macro_use]
extern crate derive_new;

// Enabled when the `pytorch` or `onnx` feature is enabled.
#[cfg(any(feature = "pytorch", feature = "onnx"))]
mod logger;

/// The onnx module.
#[cfg(feature = "onnx")]
pub mod onnx;

/// The module for generating the burn code.
#[cfg(feature = "onnx")]
pub mod burn;

/// The PyTorch module for recorder.
#[cfg(feature = "pytorch")]
pub mod pytorch;

/// The Safetensors module for recorder.
#[cfg(feature = "safetensors")]
pub mod safetensors;

// Enabled when the `pytorch` or `safetensors` feature is enabled.
#[cfg(any(feature = "pytorch", feature = "safetensors"))]
mod common;

mod formatter;
pub use formatter::*;
