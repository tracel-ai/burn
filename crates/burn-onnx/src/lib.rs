#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! `burn-onnx` is a crate designed to simplify the process of importing models trained in other
//! machine learning frameworks into the Burn framework via the ONNX format. It generates the Rust
//! source code that aligns the imported model with Burn and converts tensor data into a compatible
//! format.

#[macro_use]
extern crate derive_new;

mod logger;

/// The module for generating the burn code.
pub mod burn;

mod formatter;
mod model_gen;

pub use formatter::*;
pub use model_gen::*;
