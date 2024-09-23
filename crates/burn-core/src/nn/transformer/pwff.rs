use crate as burn;

use crate::module::{Content, DisplaySettings, Module, ModuleDisplay};
use crate::nn::Initializer;
use crate::{
    config::Config,
    nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

/// Configuration to create a [position-wise feed-forward](PositionWiseFeedForward) layer using the [init function](PositionWiseFeedForwardConfig::init).
#[derive(Config)]
pub struct PositionWiseFeedForwardConfig {
    /// The size of the input and output features.
    pub d_model: usize,
    /// The size of the hidden inner features.
    pub d_ff: usize,
    /// The dropout rate. Default: 0.1
    #[config(default = 0.1)]
    pub dropout: f64,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Applies the position-wise feed-forward network to the input tensor from the paper [Attention Is All You Need](https://arxiv.org/pdf/1706.03762v7).
///
/// # Params
///
/// - linear inner: Linear layer with `d_model` input features and `d_ff` output features.
/// - linear outer: Linear layer with `d_ff` input features and `d_model` output features.
///
/// `FFN(x) = max(0, xW1 + b1)W2 + b2`
///
/// Should be created using [PositionWiseFeedForwardConfig]
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct PositionWiseFeedForward<B: Backend> {
    /// Linear layer with `d_model` input features and `d_ff` output features.
    pub linear_inner: Linear<B>,
    /// Linear layer with `d_ff` input features and `d_model` output features.
    pub linear_outer: Linear<B>,
    /// Dropout layer.
    pub dropout: Dropout,
    /// GELU activation function.
    pub gelu: Gelu,
}

impl<B: Backend> ModuleDisplay for PositionWiseFeedForward<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_model, dff] = self.linear_inner.weight.shape().dims();

        content
            .add("d_model", &d_model)
            .add("d_ff", &dff)
            .add("prob", &self.dropout.prob)
            .optional()
    }
}

impl PositionWiseFeedForwardConfig {
    /// Initialize a new [position-wise feed-forward](PositionWiseFeedForward) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PositionWiseFeedForward<B> {
        PositionWiseFeedForward {
            linear_inner: LinearConfig::new(self.d_model, self.d_ff)
                .with_initializer(self.initializer.clone())
                .init(device),
            linear_outer: LinearConfig::new(self.d_ff, self.d_model)
                .with_initializer(self.initializer.clone())
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            gelu: Gelu::new(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    #[test]
    fn display() {
        let config = PositionWiseFeedForwardConfig::new(2, 4);
        let pwff = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            alloc::format!("{}", pwff),
            "PositionWiseFeedForward {d_model: 2, d_ff: 4, prob: 0.1, params: 22}"
        );
    }
}
