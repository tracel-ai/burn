use burn::module::Initializer;
use burn_core as burn;

use burn::config::Config;
use burn::module::Module;
use burn::module::Param;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Configuration to create a [GroupNorm](GroupNorm) layer using the [init function](GroupNormConfig::init).
#[derive(Debug, Config)]
pub struct GroupNormConfig {
    /// The number of groups to separate the channels into
    pub num_groups: usize,
    /// The number of channels expected in the input
    pub num_channels: usize,
    /// A value required for numerical stability. Default: 1e-5
    #[config(default = 1e-5)]
    pub epsilon: f64,
    /// A boolean value that when set to `true`, this module has learnable
    /// per-channel affine parameters initialized to ones (for weights)
    /// and zeros (for biases). Default: `true`
    #[config(default = true)]
    pub affine: bool,
}

/// Applies Group Normalization over a mini-batch of inputs as described in the paper [Group Normalization](https://arxiv.org/abs/1803.08494).
///
/// `Y = groupnorm(X) * γ + β`
///
/// Where:
/// - `X` is the input tensor
/// - `Y` is the output tensor
/// - `γ` is the learnable weight
/// - `β` is the learnable bias
///
/// Should be created using [GroupNormConfig](GroupNormConfig).
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct GroupNorm<B: Backend> {
    /// The learnable weight
    pub gamma: Option<Param<Tensor<B, 1>>>,
    /// The learnable bias
    pub beta: Option<Param<Tensor<B, 1>>>,
    /// The number of groups to separate the channels into
    pub num_groups: usize,
    /// The number of channels expected in the input
    pub num_channels: usize,
    /// A value required for numerical stability
    pub epsilon: f64,
    /// A boolean value that when set to `true`, this module has learnable
    pub affine: bool,
}

impl<B: Backend> ModuleDisplay for GroupNorm<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("num_groups", &self.num_groups)
            .add("num_channels", &self.num_channels)
            .add("epsilon", &self.epsilon)
            .add("affine", &self.affine)
            .optional()
    }
}

impl GroupNormConfig {
    /// Initialize a new [group norm](GroupNorm) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> GroupNorm<B> {
        assert_eq!(
            self.num_channels % self.num_groups,
            0,
            "The number of channels must be divisible by the number of groups"
        );

        let (gamma, beta) = if self.affine {
            let gamma = Initializer::Ones.init([self.num_channels], device);
            let beta = Initializer::Zeros.init([self.num_channels], device);

            (Some(gamma), Some(beta))
        } else {
            (None, None)
        };

        GroupNorm {
            num_groups: self.num_groups,
            num_channels: self.num_channels,
            gamma,
            beta,
            epsilon: self.epsilon,
            affine: self.affine,
        }
    }
}

impl<B: Backend> GroupNorm<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [GroupNorm](GroupNorm) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, num_channels, *]`
    /// - output: `[batch_size, num_channels, *]`
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        if input.shape()[1] != self.num_channels {
            panic!(
                "The number of channels in the input tensor should be equal to the number of channels in the GroupNorm module. Expected {}, got {}",
                self.num_channels,
                input.shape()[1]
            );
        }

        let gamma = self.gamma.as_ref().map(|x| x.val());
        let beta = self.beta.as_ref().map(|x| x.val());

        group_norm(
            input,
            gamma,
            beta,
            self.num_groups,
            self.epsilon,
            self.affine,
        )
    }
}

/// Applies Group Normalization over a mini-batch of inputs as described in the paper [Group Normalization](https://arxiv.org/abs/1803.08494).
///
/// `Y = groupnorm(X) * γ + β`
///
/// Where:
/// - `X` is the input tensor
/// - `Y` is the output tensor
/// - `γ` is the learnable weight
/// - `β` is the learnable bias
///
pub(crate) fn group_norm<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    gamma: Option<Tensor<B, 1>>,
    beta: Option<Tensor<B, 1>>,
    num_groups: usize,
    epsilon: f64,
    affine: bool,
) -> Tensor<B, D> {
    if (beta.is_none() || gamma.is_none()) && affine {
        panic!("Affine is set to true, but gamma or beta is None");
    }

    let shape = input.shape();
    if shape.num_elements() <= 2 {
        panic!(
            "input rank for GroupNorm should be at least 3, but got {}",
            shape.num_elements()
        );
    }

    let batch_size = shape[0];
    let num_channels = shape[1];

    let hidden_size = shape[2..].iter().product::<usize>() * num_channels / num_groups;
    let input = input.reshape([batch_size, num_groups, hidden_size]);

    let mean = input.clone().sum_dim(2) / hidden_size as f64;
    let input = input.sub(mean);

    let var = input.clone().square().sum_dim(2) / hidden_size as f64;
    let input_normalized = input.div(var.add_scalar(epsilon).sqrt());

    if affine {
        let mut affine_shape = [1; D];
        affine_shape[1] = num_channels;

        input_normalized
            .reshape(shape)
            .mul(gamma.clone().unwrap().reshape(affine_shape))
            .add(beta.clone().unwrap().reshape(affine_shape))
    } else {
        input_normalized.reshape(shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use alloc::format;
    use burn::tensor::TensorData;
    use burn::tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn group_norm_forward_affine_false() {
        let device = Default::default();
        let module = GroupNormConfig::new(2, 6)
            .with_affine(false)
            .init::<TestBackend>(&device);

        assert!(module.gamma.is_none());
        assert!(module.beta.is_none());

        let input = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([
                [
                    [-0.3034, 0.2726, -0.9659],
                    [-1.1845, -1.3236, 0.0172],
                    [1.9507, 1.2554, -0.8625],
                    [1.0682, 0.3604, 0.3985],
                    [-0.4957, -0.4461, -0.9721],
                    [1.5157, -0.1546, -0.5596],
                ],
                [
                    [-1.6698, -0.4040, -0.7927],
                    [0.3736, -0.0975, -0.1351],
                    [-0.9461, 0.5461, -0.6334],
                    [-1.0919, -0.1158, 0.1213],
                    [-0.9535, 0.1281, 0.4372],
                    [-0.2845, 0.3488, 0.5641],
                ],
            ]),
            &device,
        );

        let output = module.forward(input);

        let expected = TensorData::from([
            [
                [-0.1653, 0.3748, -0.7866],
                [-0.9916, -1.1220, 0.1353],
                [1.9485, 1.2965, -0.6896],
                [1.2769, 0.3628, 0.4120],
                [-0.7427, -0.6786, -1.3578],
                [1.8547, -0.3022, -0.8252],
            ],
            [
                [-1.9342, 0.0211, -0.5793],
                [1.2223, 0.4945, 0.4365],
                [-0.8163, 1.4887, -0.3333],
                [-1.7960, -0.0392, 0.3875],
                [-1.5469, 0.3998, 0.9561],
                [-0.3428, 0.7970, 1.1845],
            ],
        ]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn group_norm_forward_affine_true() {
        let device = Default::default();
        let module = GroupNormConfig::new(3, 6)
            .with_affine(true)
            .init::<TestBackend>(&device);

        let tolerance = Tolerance::permissive();
        module
            .gamma
            .as_ref()
            .expect("gamma should not be None")
            .val()
            .to_data()
            .assert_approx_eq::<FT>(&TensorData::ones::<f32, _>([6]), tolerance);

        module
            .beta
            .as_ref()
            .expect("beta should not be None")
            .val()
            .to_data()
            .assert_approx_eq::<FT>(&TensorData::zeros::<f32, _>([6]), tolerance);

        let input = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([
                [
                    [0.3345, 0.4429, 0.6639],
                    [0.5041, 0.4175, 0.8437],
                    [0.6159, 0.3758, 0.4071],
                    [0.5417, 0.5785, 0.7671],
                    [0.3837, 0.9883, 0.0420],
                    [0.4808, 0.8989, 0.6144],
                ],
                [
                    [0.3930, 0.2098, 0.0602],
                    [0.2298, 0.9425, 0.0333],
                    [0.7409, 0.8172, 0.8879],
                    [0.4846, 0.0486, 0.2029],
                    [0.6741, 0.9765, 0.6864],
                    [0.2827, 0.5534, 0.2125],
                ],
            ]),
            &device,
        );

        let output = module.forward(input);

        let expected = TensorData::from([
            [
                [-1.1694, -0.5353, 0.7572],
                [-0.1775, -0.6838, 1.8087],
                [0.5205, -1.3107, -1.0723],
                [-0.0459, 0.2351, 1.6734],
                [-0.5796, 1.3218, -1.6544],
                [-0.2744, 1.0406, 0.1459],
            ],
            [
                [0.2665, -0.3320, -0.8205],
                [-0.2667, 2.0612, -0.9085],
                [0.6681, 0.9102, 1.1345],
                [-0.1453, -1.5287, -1.0389],
                [0.4253, 1.5962, 0.4731],
                [-1.0903, -0.0419, -1.3623],
            ],
        ]);
        output
            .to_data()
            .assert_approx_eq::<FT>(&expected, tolerance);
    }

    #[test]
    fn display() {
        let config = GroupNormConfig::new(3, 6);
        let group_norm = config.init::<TestBackend>(&Default::default());

        assert_eq!(
            format!("{group_norm}"),
            "GroupNorm {num_groups: 3, num_channels: 6, epsilon: 0.00001, affine: true, params: 12}"
        );
    }
}
