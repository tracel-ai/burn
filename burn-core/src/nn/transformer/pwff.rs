use crate as burn;

use crate::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig, GELU},
    tensor::{backend::Backend, Tensor},
};

/// Configuration to create a [position-wise feed-forward](PositionWiseFeedForward) layer.
#[derive(Config)]
pub struct PositionWiseFeedForwardConfig {
    /// The size of the input and output features.
    pub d_model: usize,
    /// The size of the hidden inner features.
    pub d_ff: usize,
    /// The dropout rate. Default: 0.1
    #[config(default = 0.1)]
    pub dropout: f64,
}

/// Applies the position-wise feed-forward network to the input tensor.
///
/// # Params
///
/// - linear inner: Linear layer with `d_model` input features and `d_ff` output features.
/// - linear outer: Linear layer with `d_ff` input features and `d_model` output features.
#[derive(Module, Debug)]
pub struct PositionWiseFeedForward<B: Backend> {
    linear_inner: Linear<B>,
    linear_outer: Linear<B>,
    dropout: Dropout,
    gelu: GELU,
}

impl PositionWiseFeedForwardConfig {
    /// Initialize a new [position-wise feed-forward](PositionWiseFeedForward) module.
    pub fn init<B: Backend>(&self) -> PositionWiseFeedForward<B> {
        PositionWiseFeedForward {
            linear_inner: LinearConfig::new(self.d_model, self.d_ff).init(),
            linear_outer: LinearConfig::new(self.d_ff, self.d_model).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
            gelu: GELU::new(),
        }
    }
}

impl<B: Backend> PositionWiseFeedForward<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - tensor: `[batch_size, seq_length, d_model]`
    /// - output: `[batch_size, seq_length, d_model]`
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear_inner.forward(input);
        let x = self.gelu.forward(x);
        let x = self.dropout.forward(x);

        self.linear_outer.forward(x)
    }
}
