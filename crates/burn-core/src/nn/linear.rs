use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::tensor::{backend::Backend, Tensor};

use super::Initializer;

/// Configuration to create a [Linear](Linear) layer using the [init function](LinearConfig::init).
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
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Applies a linear transformation to the input tensor:
///
/// Should be created with [LinearConfig]
///
/// `O = IW + b`
#[derive(Module, Debug)]
pub struct Linear<B: Backend> {
    /// Matrix of shape `[d_input, d_output]` initialized from a uniform distribution:
    ///     `U(-k, k)`, where `k = sqrt(1 / d_input)`
    pub weight: Param<Tensor<B, 2>>,
    /// Vector of size `d_output` initialized from a uniform distribution:
    ///     `U(-k, k)`, where `k = sqrt(1 / d_input)`
    pub bias: Option<Param<Tensor<B, 1>>>,
}

impl LinearConfig {
    /// Initialize a new [linear](Linear) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Linear<B> {
        let shape = [self.d_input, self.d_output];
        let weight =
            self.initializer
                .init_with(shape, Some(self.d_input), Some(self.d_output), device);
        let bias = if self.bias {
            Some(self.initializer.init_with(
                [self.d_output],
                Some(self.d_input),
                Some(self.d_output),
                device,
            ))
        } else {
            None
        };

        Linear { weight, bias }
    }
}

impl<B: Backend> Linear<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., d_input]`
    /// - output: `[..., d_output]`
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        if D == 1 {
            // Insert and remove an extra batch dimension for the batch matmul to work.
            return Self::forward::<2>(self, input.unsqueeze()).flatten(0, 1);
        }

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
    use crate::tensor::{Data, Shape};
    use crate::TestBackend;

    #[test]
    fn initializer_default() {
        TestBackend::seed(0);

        let config = LinearConfig::new(5, 5);
        let k = (1.0 / config.d_input as f64).sqrt() as f32;
        let device = Default::default();
        let linear = config.init::<TestBackend>(&device);

        assert_eq!(
            config.initializer,
            Initializer::KaimingUniform {
                gain: 1.0 / 3.0f64.sqrt(),
                fan_out_only: false
            }
        );
        linear.weight.to_data().assert_within_range(-k..k);
    }

    #[test]
    fn initializer_zeros() {
        TestBackend::seed(0);

        let config = LinearConfig::new(5, 5).with_initializer(Initializer::Zeros);
        let device = Default::default();
        let linear = config.init::<TestBackend>(&device);

        assert_eq!(config.initializer, Initializer::Zeros);
        linear
            .weight
            .to_data()
            .assert_approx_eq(&Data::zeros(linear.weight.shape()), 3);
    }

    #[test]
    fn test_linear_forward_no_bias() {
        TestBackend::seed(0);

        let value = 2.;
        let config = LinearConfig::new(2, 3)
            .with_initializer(Initializer::Constant { value })
            .with_bias(false);
        let device = Default::default();
        let linear = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::ones(Shape::new([1, 2]), &device);
        let result = linear.forward(input);
        let expected_result = Tensor::<TestBackend, 2>::from_data([[4., 4., 4.]], &device);

        assert_eq!(result.into_data(), expected_result.into_data());
    }

    #[test]
    fn test_linear_forward_with_bias() {
        TestBackend::seed(0);

        let device = Default::default();

        let value = 2.;
        let config = LinearConfig::new(2, 3).with_initializer(Initializer::Constant { value });
        let linear = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 2>::ones(Shape::new([1, 2]), &device);
        let result = linear.forward(input);
        let expected_result = Tensor::<TestBackend, 2>::from_data([[6., 6., 6.]], &device);

        assert_eq!(result.into_data(), expected_result.into_data());
    }

    #[test]
    fn test_linear_1d() {
        TestBackend::seed(0);

        let device = Default::default();

        let value = 2.;
        let config = LinearConfig::new(2, 3).with_initializer(Initializer::Constant { value });
        let linear = config.init::<TestBackend>(&device);

        let input_1d = Tensor::<TestBackend, 1>::ones(Shape::new([2]), &device);
        let input_2d = Tensor::<TestBackend, 2>::ones(Shape::new([1, 2]), &device);

        let result_1d = linear.forward(input_1d).unsqueeze();
        let result_2d = linear.forward(input_2d);

        assert_eq!(result_1d.into_data(), result_2d.into_data());
    }
}
