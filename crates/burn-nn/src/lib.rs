#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![recursion_limit = "256"]

//! Burn neural network module.

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

mod padding;
pub use padding::*;

// For backward compat, `burn::nn::Initializer`
pub use burn_core::module::Initializer;

extern crate alloc;
