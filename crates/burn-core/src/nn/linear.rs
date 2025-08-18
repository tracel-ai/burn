use burn_tensor::module::linear;

use crate as burn;

use crate::config::Config;
use crate::module::Param;
use crate::module::{Content, DisplaySettings, Module, ModuleDisplay};
use crate::tensor::{Tensor, backend::Backend};

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
    /// The layout in which the linear parameters are stored.
    #[config(default = "LinearLayout::Row")]
    pub layout: LinearLayout,
}

#[derive(Config, Debug, Copy)]
/// The layout in which the linear parameters are stored.
///
/// This can have performance impacts.
pub enum LinearLayout {
    /// Parameters are stored in Row major.
    Row,
    /// Parameters are stored in Col major.
    Col,
}

/// Applies a linear transformation to the input tensor.
///
/// Should be created with [LinearConfig]
///
/// `O = IW + b`
#[derive(Module, Debug)]
#[module(custom_display)]
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
        let weight = match self.layout {
            LinearLayout::Row => {
                let shape = [self.d_input, self.d_output];
                self.initializer
                    .init_with(shape, Some(self.d_input), Some(self.d_output), device)
            }
            LinearLayout::Col => {
                let shape = [self.d_output, self.d_input];
                self.initializer
                    .init_with(shape, Some(self.d_output), Some(self.d_input), device)
                    // The param is already transposed when init. We re-transpose to have
                    // [d_output, d_input] while saving.
                    .save_mapper(|tensor| tensor.transpose())
                    // When loading from record we have to transpose.
                    .load_mapper(|tensor| tensor.transpose())
                    // When loading from initialization, we have to transpose.
                    .lazy_map(|tensor| tensor.transpose())
            }
        };
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
    /// # Arguments
    ///
    /// - `input` - The input tensor of shape `[..., d_input]`.
    ///
    /// # Shapes
    ///
    /// - input: `[..., d_input]`
    /// - output: `[..., d_output]`
    ///
    /// # Returns
    ///
    /// The transformed tensor of shape `[..., d_output]`.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        linear(
            input,
            self.weight.val(),
            self.bias.as_ref().map(|b| b.val()),
        )
    }
}

impl<B: Backend> ModuleDisplay for Linear<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let [d_input, d_output] = self.weight.shape().dims();
        content
            .add("d_input", &d_input)
            .add("d_output", &d_output)
            .add("bias", &self.bias.is_some())
            .optional()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use crate::tensor::{Shape, TensorData};
    use burn_tensor::ElementConversion;
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn initializer_default() {
        TestBackend::seed(0);

        let config = LinearConfig::new(5, 5);
        let k = (1.0 / config.d_input as f64).sqrt().elem::<FT>();
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
        linear.weight.to_data().assert_approx_eq::<FT>(
            &TensorData::zeros::<f32, _>(linear.weight.shape()),
            Tolerance::default(),
        );
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

        let result_1d = linear.forward(input_1d).unsqueeze::<2>();
        let result_2d = linear.forward(input_2d);

        assert_eq!(result_1d.into_data(), result_2d.into_data());
    }

    #[test]
    fn display() {
        let config = LinearConfig::new(3, 5);
        let linear = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            alloc::format!("{linear}"),
            "Linear {d_input: 3, d_output: 5, bias: true, params: 20}"
        );
    }
}
