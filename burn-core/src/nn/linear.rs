use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::tensor::{backend::Backend, Tensor};

use libm::sqrt;

use super::Initializer;

/// Configuration to create a [Linear](Linear) layer.
#[derive(Config, Debug)]
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
    pub(crate) weight: Param<Tensor<B, 2>>,
    pub(crate) bias: Option<Param<Tensor<B, 1>>>,
}

impl LinearConfig {
    /// Initialize a new [linear](Linear) module.
    pub fn init<B: Backend>(&self) -> Linear<B> {
        let k = sqrt(1.0 / self.d_input as f64);

        let initializer = if let Initializer::UniformDefault = self.initializer {
            Initializer::Uniform(-k, k)
        } else {
            self.initializer.clone()
        };

        let weight = initializer.init([self.d_input, self.d_output]);

        let bias = if self.bias {
            Some(initializer.init([self.d_output]))
        } else {
            None
        };

        Linear {
            weight: Param::from(weight),
            bias: bias.map(Param::from),
        }
    }

    /// Initialize a new [linear](Linear) module with a [record](LinearRecord).
    pub fn init_with<B: Backend>(&self, record: LinearRecord<B>) -> Linear<B> {
        Linear {
            weight: record.weight,
            bias: record.bias,
        }
    }
}

impl<B: Backend> Linear<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any, d_input]`
    /// - output: `[..., any, d_output]`
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let output = input.matmul(self.weight.val().unsqueeze());

        match &self.bias {
            Some(bias) => output + bias.val().unsqueeze(),
            None => output,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_tensor::Data;

    #[test]
    fn initializer_default() {
        TestBackend::seed(0);

        let config = LinearConfig::new(5, 5);
        let k = sqrt(1.0 / config.d_input as f64) as f32;
        let linear = config.init::<TestBackend>();

        assert_eq!(config.initializer, Initializer::UniformDefault);
        linear.weight.to_data().assert_in_range(-k, k);
    }

    #[test]
    fn initializer_zeros() {
        TestBackend::seed(0);

        let config = LinearConfig::new(5, 5).with_initializer(Initializer::Zeros);
        let linear = config.init::<TestBackend>();

        assert_eq!(config.initializer, Initializer::Zeros);
        linear
            .weight
            .to_data()
            .assert_approx_eq(&Data::zeros(linear.weight.shape()), 3);
    }
}
