//! # Activation Layers

mod activation_layer;

pub use activation_layer::*;

// TODO: move these libs from "nn/" to "nn/activation/"
pub use super::{
    GLU, HardSigmoid, HardSigmoidConfig, LeakyRelu, LeakyReluConfig, Linear, LinearConfig, PRelu,
    PReluConfig, Relu, Sigmoid, SwiGlu, SwiGluConfig, Tanh,
};
