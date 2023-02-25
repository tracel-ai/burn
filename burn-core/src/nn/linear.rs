use alloc::{format, vec::Vec};

use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::tensor::backend::Backend;
use crate::tensor::{Distribution, ElementConversion, Tensor};

use libm::sqrt;

/// Configuration to create a [Linear](Linear) layer.
#[derive(Config)]
pub struct LinearConfig {
    /// The size of the input features.
    pub d_input: usize,
    /// The size of the output features.
    pub d_output: usize,
    /// If a bias should be applied during the linear transformation.
    #[config(default = true)]
    pub bias: bool,
}

/// Applies a linear transformation to the input tensor:
///
/// `O = IW + b`
///
/// # Params
///
/// - weight: Matrix of shape `[d_input, d_output]` initialized from a uniform distribution:
///     `U(-k, k)`, where `k = sqrt(1 / d_input)`
///
/// - bias (optional): Vector of size `d_output` initialized from a uniform distribution:
///     `U(-k, k)`, where `k = sqrt(1 / d_input)`
#[derive(Module, Debug)]
pub struct Linear<B: Backend> {
    weight: Param<Tensor<B, 2>>,
    bias: Param<Option<Tensor<B, 1>>>,
}

impl<B: Backend> Linear<B> {
    /// Create the module from the given configuration.
    pub fn new(config: &LinearConfig) -> Self {
        let k = sqrt(1.0 / config.d_input as f64);
        let distribution = Distribution::Uniform((-1.0 * k).to_elem(), k.to_elem());

        let weight = Tensor::random([config.d_input, config.d_output], distribution);
        let bias = match config.bias {
            true => Some(Tensor::random([config.d_output], distribution)),
            false => None,
        };

        Self {
            weight: Param::new(weight),
            bias: Param::new(bias),
        }
    }

    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any, d_input]`
    /// - output: `[..., any, d_output]`
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let output = input.matmul(self.weight.val().unsqueeze());

        match self.bias.val() {
            Some(bias) => output + bias.unsqueeze(),
            None => output,
        }
    }
}
