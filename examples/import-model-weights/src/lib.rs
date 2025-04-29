//! # Burn Model Weights Import Examples
//!
//! This crate demonstrates various techniques for importing pre-trained model weights
//! from different formats into the Burn deep learning framework. It provides working
//! examples for:
//!
//! - Loading PyTorch `.pt` weights directly
//! - Loading SafeTensors `.safetensors` weights directly
//! - Converting between formats
//! - Using Burn's native named MessagePack format
//!
//! ## Modules
//!
//! - `inference`: Standardized inference functionality for testing models
//! - `model`: MNIST CNN model architecture compatible with imported weights
//!
//! ## Example Binaries
//!
//! - `pytorch`: Demo for importing PyTorch weights
//! - `safetensors`: Demo for importing SafeTensors weights
//! - `convert`: Tool for converting between weight formats
//! - `namedmpk`: Demo for using Burn's native format
//!
//! For detailed usage instructions, see the README.md or the documentation in each binary.

pub mod inference;
pub mod model;

pub use inference::*;
pub use model::*;
