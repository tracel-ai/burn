use crate as burn;

use crate::config::Config;
use crate::module::{Module, Param};
use crate::nn::norm::group_norm;
use crate::nn::Initializer;
use crate::tensor::{backend::Backend, Tensor};

/// Configuration to create a [InstanceNorm](InstanceNorm) layer using the [init function](InstanceNormConfig::init).
#[derive(Debug, Config)]
pub struct InstanceNormConfig {
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

/// Applies Instance Normalization over a tensor as described in the paper [Instance Normalization](https://arxiv.org/abs/1607.08022)
///
/// Should be created using [InstanceNormConfig](InstanceNormConfig).
#[derive(Module, Debug)]
pub struct InstanceNorm<B: Backend> {
    /// The learnable weight
    pub gamma: Option<Param<Tensor<B, 1>>>,
    /// The learnable bias
    pub beta: Option<Param<Tensor<B, 1>>>,

    num_channels: usize,
    epsilon: f64,
    affine: bool,
}

impl InstanceNormConfig {
    /// Initialize a new [instance norm](InstanceNorm) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> InstanceNorm<B> {
        let (gamma, beta) = if self.affine {
            let gamma = Initializer::Ones.init([self.num_channels], device);
            let beta = Initializer::Zeros.init([self.num_channels], device);

            (Some(gamma), Some(beta))
        } else {
            (None, None)
        };

        InstanceNorm {
            gamma,
            beta,
            num_channels: self.num_channels,
            epsilon: self.epsilon,
            affine: self.affine,
        }
    }
}

impl<B: Backend> InstanceNorm<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// See also [InstanceNormConfig](InstanceNormConfig) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, num_channels, *]`
    /// - output: `[batch_size, num_channels, *]`
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        // Instance norm is equivalent to group norm when the number of groups is equal to the number of channels.
        let num_groups = self.num_channels;

        let gamma = self.gamma.as_ref().map(|x| x.val());
        let beta = self.beta.as_ref().map(|x| x.val());

        group_norm(input, gamma, beta, num_groups, self.epsilon, self.affine)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Data;
    use crate::TestBackend;

    #[test]
    fn instance_norm_forward_affine_false() {
        let device = Default::default();
        let module = InstanceNormConfig::new(6)
            .with_affine(false)
            .init::<TestBackend>(&device);

        let input = Tensor::from_data(
            Data::from([
                [
                    [-0.3034, 0.2726, -0.9659],
                    [-1.1845, 1.4078, 0.9774],
                    [0.3963, -1.3738, 1.4125],
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

        output.to_data().assert_approx_eq(
            &Data::from([
                [
                    [0.0569, 1.1952, -1.2522],
                    [-1.3971, 0.8883, 0.5088],
                    [0.2183, -1.3192, 1.1009],
                    [1.4126, -0.7649, -0.6477],
                    [0.5999, 0.8091, -1.409],
                    [1.39, -0.4696, -0.9205],
                ],
                [
                    [-1.3492, 1.0417, 0.3075],
                    [1.411, -0.6243, -0.7867],
                    [-0.9363, 1.386, -0.4497],
                    [-1.3899, 0.4692, 0.9208],
                    [-1.3822, 0.4319, 0.9503],
                    [-1.3714, 0.3868, 0.9846],
                ],
            ]),
            3,
        );
    }

    #[test]
    fn instance_norm_forward_affine_true() {
        let device = Default::default();
        let module = InstanceNormConfig::new(6)
            .with_affine(true)
            .init::<TestBackend>(&device);

        let input = Tensor::from_data(
            Data::from([
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

        output.to_data().assert_approx_eq(
            &Data::from([
                [
                    [-1.06458, -0.2738, 1.33838],
                    [-0.45848, -0.92929, 1.38777],
                    [1.40388, -0.84877, -0.55511],
                    [-0.88515, -0.51245, 1.3976],
                    [-0.22397, 1.32124, -1.09727],
                    [-1.05468, 1.34316, -0.28848],
                ],
                [
                    [1.26372, -0.08229, -1.18144],
                    [-0.44049, 1.38403, -0.94354],
                    [-1.23979, 0.03109, 1.2087],
                    [1.32524, -1.08999, -0.23524],
                    [-0.75061, 1.4132, -0.66259],
                    [-0.45469, 1.38697, -0.93228],
                ],
            ]),
            3,
        );
    }
}
