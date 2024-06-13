use crate as burn;
use crate::config::Config;
use crate::module::Module;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

use crate::tensor::activation::leaky_relu;

/// Leaky ReLu layer.
///
/// Should be created with [LeakyReluConfig](LeakyReluConfig).
#[derive(Module, Clone, Debug)]
pub struct LeakyRelu {
    /// The negative slope.
    pub negative_slope: f64,
}
/// Configuration to create a [Leaky Relu](LeakyRelu) layer using the [init function](LeakyReluConfig::init).
#[derive(Config, Debug)]
pub struct LeakyReluConfig {
    /// The negative slope. Default is 0.01
    #[config(default = "0.01")]
    pub negative_slope: f64,
}
impl LeakyReluConfig {
    /// Initialize a new [Leaky Relu](LeakyRelu) Layer
    pub fn init(&self) -> LeakyRelu {
        LeakyRelu {
            negative_slope: self.negative_slope,
        }
    }
}

impl LeakyRelu {
    /// Forward pass for the Leaky ReLu layer.
    ///
    /// See [leaky_relu](crate::tensor::activation::leaky_relu) for more information.
    ///
    /// # Shapes
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        leaky_relu(input, self.negative_slope)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Data;
    use crate::TestBackend;

    #[test]
    fn test_leaky_relu_forward() {
        let device = <TestBackend as Backend>::Device::default();
        let model = LeakyReluConfig::new().init();
        let input = Tensor::<TestBackend, 2>::from_data(Data::from([[0.4410, -0.2507]]), &device);
        let out = model.forward(input);
        assert_eq!(out.to_data(), Data::from([[0.4410, -0.002507]]));
    }
    #[test]
    fn test_leaky_relu_forward_multi_dim() {
        let input = [
            [
                [-1.0222, 1.5810, 0.3457, -1.3530],
                [0.0231, 0.8681, 0.2473, -0.0377],
                [0.3520, -1.1199, 1.2219, 0.2804],
            ],
            [
                [1.0002, 0.7259, 0.8779, 0.2084],
                [1.5615, -0.1057, -0.4886, -1.5184],
                [-0.5523, -0.2741, -0.0210, -1.1352],
            ],
        ];
        let expected_output = [
            [
                [-1.0222e-02, 1.5810e+00, 3.457e-01, -1.3530e-02],
                [2.31e-02, 8.681e-01, 2.473e-01, -3.77e-04],
                [3.52e-01, -1.1199e-02, 1.2219e+00, 2.804e-01],
            ],
            [
                [1.0002e+00, 7.259e-01, 8.779e-01, 2.084e-01],
                [1.5615e+00, -1.057e-03, -4.886e-03, -1.5184e-02],
                [-5.523e-03, -2.741e-03, -2.1e-04, -1.1352e-02],
            ],
        ];

        let device = <TestBackend as Backend>::Device::default();
        let model = LeakyReluConfig::new().init();
        let input_data = Tensor::<TestBackend, 3>::from_data(Data::from(input), &device);
        let actual_output = model.forward(input_data);
        actual_output
            .to_data()
            .assert_approx_eq(&Data::from(expected_output), 4)
    }
}
