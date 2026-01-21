#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
// #![allow(clippy::ptr_arg)]
// #![allow(clippy::single_match)]
// #![allow(clippy::upper_case_acronyms)]
// #![allow(clippy::approx_constant)]

//! `burn-import` is a crate designed to simplify the process of importing models trained in other
//! machine learning frameworks into the Burn framework.

#[cfg(any(feature = "pytorch", feature = "safetensors"))]
#[macro_use]
extern crate derive_new;

/// The onnx module.
#[cfg(feature = "onnx")]
#[deprecated(
    since = "0.21.0",
    note = "ONNX import was moved to `burn-onnx`. Use that crate instead."
)]
pub mod onnx {
    #[deprecated(
        since = "0.21.0",
        note = "ONNX import was moved to `burn-onnx`. Use that crate instead."
    )]
    #[allow(missing_docs)]
    pub type ModelGen = burn_onnx::ModelGen;
}

/// The module for generating the burn code.
#[cfg(feature = "onnx")]
#[deprecated(
    since = "0.21.0",
    note = "ONNX import was moved to `burn-onnx`. Use that crate instead."
)]
pub mod burn {
    pub use burn_onnx::burn::*;
}

/// The PyTorch module for recorder.
#[cfg(feature = "pytorch")]
pub mod pytorch;

/// The Safetensors module for recorder.
#[cfg(feature = "safetensors")]
pub mod safetensors;

// Enabled when the `pytorch` or `safetensors` feature is enabled.
#[cfg(any(feature = "pytorch", feature = "safetensors"))]
mod common;
