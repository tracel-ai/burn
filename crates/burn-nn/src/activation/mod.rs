//! # Activation Layers
//!
//! Users who desire a selectable activation function should
//! consider [`Activation`], which provides an abstraction over:
//! * [`Relu`] - the default,
//! * ['PRelu']
//! * [`Gelu`]
//! * [`LeakyRelu`]
//! * [`SwiGlu`]
//! * [`Sigmoid`]
//! * [`HardSigmoid`]
//! * [`Tanh`]
//!
//! The activation layer [`GLU`] has shape-changing behaviors
//! not compatible with the common API, and is not included
//! in the abstraction wrappers.

mod activation_wrapper;

// These are pub(crate) for dual-export in `nn` without re-exporting
// all of `nn.activation`, or manually listing each symbol.
pub(crate) mod gelu;
pub(crate) mod glu;
pub(crate) mod hard_sigmoid;
pub(crate) mod leaky_relu;
pub(crate) mod prelu;
pub(crate) mod relu;
pub(crate) mod sigmoid;
pub(crate) mod swiglu;
pub(crate) mod tanh;

pub use activation_wrapper::*;
pub use gelu::*;
pub use glu::*;
pub use hard_sigmoid::*;
pub use leaky_relu::*;
pub use prelu::*;
pub use relu::*;
pub use sigmoid::*;
pub use swiglu::*;
pub use tanh::*;
