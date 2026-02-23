//! # Activation Layers
//!
//! Users who desire a selectable activation function should
//! consider [`Activation`], which provides an abstraction over:
//! * [`Relu`] - the default,
//! * ['PRelu']
//! * [`Gelu`]
//! * [`LeakyRelu`]
//! * [`SwiGlu`]
//! * [`Selu`]
//! * [`Sigmoid`]
//! * [`HardSigmoid`]
//! * [`HardSwish`]
//! * [`Softplus`]
//! * [`Softsign`]
//! * [`Tanh`]
//! * [`Elu`]
//! * [`Celu`]
//! * [`ThresholdedRelu`]
//!
//! The activation layer [`GLU`] has shape-changing behaviors
//! not compatible with the common API, and is not included
//! in the abstraction wrappers.

mod activation_wrapper;

// These are pub(crate) for dual-export in `nn` without re-exporting
// all of `nn.activation`, or manually listing each symbol.
pub(crate) mod celu;
pub(crate) mod elu;
pub(crate) mod gelu;
pub(crate) mod glu;
pub(crate) mod hard_sigmoid;
pub(crate) mod hard_swish;
pub(crate) mod hard_shrink;
pub(crate) mod leaky_relu;
pub(crate) mod prelu;
pub(crate) mod relu;
pub(crate) mod selu;
pub(crate) mod sigmoid;
pub(crate) mod softplus;
pub(crate) mod soft_shrink;
pub(crate) mod softsign;
pub(crate) mod swiglu;
pub(crate) mod tanh;
pub(crate) mod thresholded_relu;

pub use activation_wrapper::*;
pub use celu::*;
pub use elu::*;
pub use gelu::*;
pub use glu::*;
pub use hard_sigmoid::*;
pub use hard_swish::*;
pub use hard_shrink::*;
pub use leaky_relu::*;
pub use prelu::*;
pub use relu::*;
pub use selu::*;
pub use sigmoid::*;
pub use softplus::*;
pub use soft_shrink::*;
pub use softsign::*;
pub use swiglu::*;
pub use tanh::*;
pub use thresholded_relu::*;
