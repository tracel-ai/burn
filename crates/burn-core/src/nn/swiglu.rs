use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::tensor::activation::silu;
use crate::tensor::{backend::Backend, Tensor};

use super::{Initializer, Linear, LinearConfig};

/// Configuration to create a [SwiGlu](SwiGlu) activation layer using the [init function](SwiGluConfig::init).
#[derive(Config, Debug)]
pub struct SwiGluConfig {
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
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Applies the SwiGLU or Swish Gated Linear Unit to the input tensor.
/// The SwiGLU activation function is defined as:
/// `SwiGLU(x) = Swish(W_inner * x + b_inner) * (W_outer * x + b_outer)`
///
/// Should be created with [SwiGluConfig].
#[derive(Module, Debug)]
pub struct SwiGlu<B: Backend> {
    /// The inner linear layer for Swish activation function
    /// with `d_input` input features and `d_output` output features.
    pub linear_inner: Linear<B>,
    /// The outer linear layer for element wise multiplication
    /// with `d_input` input features and `d_output` output features.
    pub linear_outer: Linear<B>,
}

impl SwiGluConfig {
    /// Initialize a new [SwiGLU](SwiGlu) activation layer.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SwiGlu<B> {
        SwiGlu {
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
}

impl<B: Backend> SwiGlu<B> {
    /// Applies the Swish Gated Linear Unit to the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, seq_length, d_input]`  
    /// - output: `[batch_size, seq_length, d_output]`
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear_inner.forward(input.clone());
        let x = silu(x);
        x.mul(self.linear_outer.forward(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;

    #[test]
    fn test_swiglu_forward_no_bias() {
        TestBackend::seed(0);
        let device = Default::default();
        let config = SwiGluConfig::new(3, 3).with_initializer(Initializer::Constant { value: 0.5 });
        let swiglu = config.init(&device);
        let input =
            Tensor::<TestBackend, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
        let output = swiglu.forward(input);
        let expected_output = Tensor::<TestBackend, 2>::from_data(
            [[8.5732, 8.5732, 8.5732], [56.2189, 56.2189, 56.2189]],
            &device,
        );
        output
            .to_data()
            .assert_approx_eq(&expected_output.to_data(), 4);
    }

    #[test]
    fn test_swiglu_forward_with_bias() {
        TestBackend::seed(0);
        let device = Default::default();
        let config = SwiGluConfig::new(3, 3)
            .with_bias(true)
            .with_initializer(Initializer::Constant { value: 0.5 });
        let swiglu = config.init(&device);
        let input =
            Tensor::<TestBackend, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
        let output = swiglu.forward(input);
        let expected_output = Tensor::<TestBackend, 2>::from_data(
            [[11.8909, 11.8909, 11.8909], [63.9785, 63.9785, 63.9785]],
            &device,
        );
        output
            .to_data()
            .assert_approx_eq(&expected_output.to_data(), 4);
    }
}
