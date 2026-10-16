#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Neural network building blocks for Burn.
//!
//! Every layer is a [`Module`](burn_core::module::Module) built from a `Config`:
//! `LinearConfig::new(784, 128).init(&device)` returns a [`Linear`]. Applications use these through
//! `burn::nn`.
//!
//! - Layers: linear, convolution and transposed convolution (1D to 3D), pooling, normalization
//!   (batch, layer, group, instance, RMS), embeddings, dropout, recurrent layers (LSTM, GRU),
//!   attention and transformers, positional and rotary encodings, interpolation, and more.
//! - [`activation`]: activation functions as modules.
//! - [`loss`]: loss functions, from mean squared error and cross-entropy to CTC.
//! - [`Initializer`]: weight initialization schemes.
//!
//! # Feature flags
//!
//! - `std` (default): standard library support. Without it the crate is `no_std` with `alloc`.
//! - `tracing`: instrument operations with the `tracing` crate.

/// Loss module
pub mod loss;

/// Neural network modules implementations.
pub mod modules;
pub use modules::*;

pub mod activation;
pub use activation::{
    celu::*, elu::*, gelu::*, glu::*, hard_shrink::*, hard_sigmoid::*, leaky_relu::*, prelu::*,
    relu::*, selu::*, shrink::*, sigmoid::*, soft_shrink::*, softplus::*, softsign::*, swiglu::*,
    tanh::*, thresholded_relu::*,
};

mod initializer;
mod padding;

pub use initializer::*;
pub use padding::*;

extern crate alloc;

#[cfg(test)]
fn test_device() -> burn_core::tensor::Device {
    burn_core::tensor::Device::flex()
}
