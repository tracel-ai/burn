use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::tensor::activation::silu;
use crate::tensor::{backend::Backend, Tensor};

use super::{Initializer, Linear, LinearConfig};

/// Configuration to create a [SwiGLU](SwiGLU) activation layer.
#[derive(Config, Debug)]
pub struct SwiGLUConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the output features.
    pub d_output: usize,
    /// If a bias should be applied during the linear transformation. Default behaviour is False
    /// for SwiGLU activation implementations.
    #[config(default = false)]
    pub bias: bool,
    /// The type of function used to initialize the linear layer parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/libm::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Applies the SwiGLU or Swish Gated Linear Unit to the input tensor.
/// The SwiGLU activation function is defined as:
/// `SwiGLU(x) = Swish(W_inner * x + b_inner) * (W_outer * x + b_outer)`
///
/// # Params
///
/// - linear inner: The inner linear layer for Swish activation function
/// with `d_input` input features and `d_output` output features.
/// - linear outer: Outer Linear layer for element wise multiplication
/// with `d_input` input features and `d_output` output features.
#[derive(Module, Debug)]
pub struct SwiGLU<B: Backend> {
    linear_inner: Linear<B>,
    linear_outer: Linear<B>,
}

impl SwiGLUConfig {
    /// Initialize a new [SwiGLU](SwiGLU) activation layer.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SwiGLU<B> {
        SwiGLU {
            linear_inner: LinearConfig::new(self.d_input, self.d_output)
                .with_bias(self.bias)
                .with_initializer(self.initializer.clone())
                .init(device),
            linear_outer: LinearConfig::new(self.d_input, self.d_output)
                .with_bias(self.bias)
                .with_initializer(self.initializer.clone())
                .init(device),
        }
    }
    /// Initialize a new [SwiGLU](SwiGLU) activation layer with a [record](SwiGLU).
    pub fn init_with<B: Backend>(&self, record: SwiGLURecord<B>) -> SwiGLU<B> {
        SwiGLU {
            linear_inner: LinearConfig::new(self.d_input, self.d_output)
                .with_bias(self.bias)
                .init_with(record.linear_inner),
            linear_outer: LinearConfig::new(self.d_input, self.d_output)
                .with_bias(self.bias)
                .init_with(record.linear_outer),
        }
    }
}

impl<B: Backend> SwiGLU<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - tensor: `[batch_size, seq_length, d_input]`
    /// - output: `[batch_size, seq_length, d_output]`
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear_inner.forward(input.clone());
        let x = silu(x);
        let x = x.mul(self.linear_outer.forward(input));
        x
    }
}
