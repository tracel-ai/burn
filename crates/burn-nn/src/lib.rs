#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![recursion_limit = "256"]

//! Burn neural network module.
//!
//! [`Initializer`] is defined by this crate. The former direct-crate path
//! `burn_core::module::Initializer` has been removed; umbrella users can keep
//! using `burn::nn::Initializer` or `burn::module::Initializer`.

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
