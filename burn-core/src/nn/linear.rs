use alloc::{format, vec::Vec};

use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::tensor::{backend::Backend, Tensor};

use libm::sqrt;

use super::Initializer;

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
    /// The type of function used to initialize neural network parameters
    #[config(default = "Initializer::UniformDefault")]
    pub initializer: Initializer,
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

        let initializer = if let Initializer::UniformDefault = config.initializer {
            Initializer::Uniform(-k, k)
        } else {
            config.initializer.clone()
        };

        let weight = initializer.init([config.d_input, config.d_output]);

        let bias = if config.bias {
            Some(initializer.init([config.d_output]))
        } else {
            None
        };

        Self {
            weight: Param::from(weight),
            bias: Param::from(bias),
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

#[cfg(test)]
mod tests {
    use super::*;
    pub type TB = burn_ndarray::NdArrayBackend<f32>;

    #[test]
    fn initializer_default() {
        TB::seed(0);
        let config = LinearConfig::new(5, 5);
        let k = sqrt(1.0 / config.d_input as f64);

        assert_eq!(config.initializer, Initializer::UniformDefault);
        let conv: Linear<TB> = Linear::new(&config);
        for item in conv.weight.to_data().value.iter() {
            if *item < -k as f32 || *item > k as f32 {
                panic!("Element ({item}) is not within the range of (-{k},{k})");
            }
        }
    }

    #[test]
    fn initializer_zeros() {
        TB::seed(0);
        let config = LinearConfig::new(5, 5).with_initializer(Initializer::Zeros);
        assert_eq!(config.initializer, Initializer::Zeros);
        let conv: Linear<TB> = Linear::new(&config);
        for item in conv.weight.to_data().value.iter() {
            assert_eq!(*item, 0.0f32);
        }
    }
}
