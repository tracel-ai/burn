use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

/// Configuration to create a [GroupNorm](GroupNorm) layer.
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

/// Applies Group Normalization over a mini-batch of inputs.
///
/// `Y = groupnorm(X) * γ + β`
#[derive(Module, Debug)]
pub struct GroupNorm<B: Backend> {
    num_groups: usize,
    num_channels: usize,
    gamma: Option<Param<Tensor<B, 1>>>,
    beta: Option<Param<Tensor<B, 1>>>,
    epsilon: f64,
    affine: bool,
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
            let gamma = Tensor::ones([self.num_channels], device).into();
            let beta = Tensor::zeros([self.num_channels], device).into();

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

    /// Initialize a new [group norm](GroupNorm) module with a [record](GroupNormRecord).
    pub fn init_with<B: Backend>(&self, record: GroupNormRecord<B>) -> GroupNorm<B> {
        GroupNorm {
            num_groups: self.num_groups,
            num_channels: self.num_channels,
            gamma: record.gamma,
            beta: record.beta,
            epsilon: self.epsilon,
            affine: self.affine,
        }
    }
}

impl<B: Backend> GroupNorm<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any, d_model]`
    /// - output: `[..., any, d_model]`
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let shape = input.shape();
        if shape.num_elements() <= 2 {
            panic!(
                "input rank for GroupNorm should be at least 3, but got {}",
                shape.num_elements()
            );
        }

        let batch_size = shape.dims[0];
        let num_channels = shape.dims[1];

        if num_channels != self.num_channels {
            panic!(
                "expected {} channels but got {}",
                self.num_channels, num_channels
            );
        }

        let hidden_size =
            shape.dims[2..].iter().product::<usize>() * num_channels / self.num_groups;
        let input = input.reshape([batch_size, self.num_groups, hidden_size]);

        let mean = input.clone().sum_dim(2) / hidden_size as f64;
        let input = input.sub(mean);

        let var = input.clone().powf_scalar(2.).sum_dim(2) / hidden_size as f64;
        let input_normalized = input.div(var.sqrt().add_scalar(self.epsilon));

        if self.affine {
            let mut affine_shape = [1; D];
            affine_shape[1] = num_channels;

            input_normalized
                .reshape(shape)
                .mul(self.gamma.clone().unwrap().val().reshape(affine_shape))
                .add(self.beta.clone().unwrap().val().reshape(affine_shape))
        } else {
            input_normalized.reshape(shape)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_tensor::Data;

    #[test]
    fn group_norm_forward_affine_false() {
        let device = Default::default();
        let module = GroupNormConfig::new(2, 6)
            .with_affine(false)
            .init::<TestBackend>(&device);

        assert!(module.gamma.is_none());
        assert!(module.beta.is_none());

        let input = Tensor::from_data(
            Data::from([
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

        output.to_data().assert_approx_eq(
            &Data::from([
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
            ]),
            3,
        );
    }

    #[test]
    fn group_norm_forward_affine_true() {
        let device = Default::default();
        let module = GroupNormConfig::new(3, 6)
            .with_affine(true)
            .init::<TestBackend>(&device);

        module
            .gamma
            .as_ref()
            .expect("gamma should not be None")
            .val()
            .to_data()
            .assert_approx_eq(&Data::ones([6].into()), 3);

        module
            .beta
            .as_ref()
            .expect("beta should not be None")
            .val()
            .to_data()
            .assert_approx_eq(&Data::zeros([6]), 3);

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
            ]),
            3,
        );
    }
}
