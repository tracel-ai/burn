//! # Activation Layers
//!
//! Users who desire a selectable activation function should
//! consider [`Activation`] or [`DimSelectActivation`], which
//! provide wrappers for:
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

mod activation_layer;
mod gelu;
mod glu;
mod hard_sigmoid;
mod leaky_relu;
mod prelu;
mod relu;
mod sigmoid;
mod swiglu;
mod tanh;

pub use activation_layer::*;
pub use gelu::*;
pub use glu::*;
pub use hard_sigmoid::*;
pub use leaky_relu::*;
pub use prelu::*;
pub use relu::*;
pub use sigmoid::*;
pub use swiglu::*;
pub use tanh::*;
