use crate as burn;

use super::{GroupNorm, GroupNormConfig};
use crate::config::Config;
use crate::module::Module;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

/// Configuration to create a [LayerNorm](LayerNorm) layer.
#[derive(Config)]
pub struct LayerNormConfig {
    /// The number of channels expected in the input
    num_channels: usize,
    /// A value required for numerical stability. Default: 1e-5
    #[config(default = 1e-5)]
    epsilon: f64,
    /// A boolean value that when set to `true`, this module has learnable
    /// per-channel affine parameters initialized to ones (for weights)
    /// and zeros (for biases). Default: `true`
    #[config(default = true)]
    affine: bool,
}

/// Applies Layer Normalization over an input tensor as described in the paper [Layer Normalization](https://arxiv.org/abs/1607.06450).
///
/// `Y = norm(X) * γ + β`
#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    group_norm: GroupNorm<B>,
}

impl LayerNormConfig {
    /// Initialize a new [layer norm](LayerNorm) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> LayerNorm<B> {
        LayerNorm {
            group_norm: self.to_group_norm().init(device),
        }
    }

    /// Initialize a new [layer norm](LayerNorm) module with a [record](LayerNormRecord).
    pub fn init_with<B: Backend>(&self, record: LayerNormRecord<B>) -> LayerNorm<B> {
        LayerNorm {
            group_norm: self.to_group_norm().init_with(record.group_norm),
        }
    }

    fn to_group_norm(&self) -> GroupNormConfig {
        GroupNormConfig {
            // Group norm is equivalent to layer norm, when the number of groups is 1.
            num_groups: 1,
            num_channels: self.num_channels,
            epsilon: self.epsilon,
            affine: self.affine,
        }
    }
}

impl<B: Backend> LayerNorm<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any, d_model]`
    /// - output: `[..., any, d_model]`
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        self.group_norm.forward(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_tensor::Data;

    #[cfg(feature = "std")]
    use crate::{TestAutodiffBackend, TestBackend};

    #[cfg(not(feature = "std"))]
    use crate::TestBackend;

    #[test]
    fn layer_norm_forward() {
        let device = Default::default();
        let module = LayerNormConfig::new(10).init::<TestBackend>(&device);
        let input = Tensor::from_data(
            Data::from([[
                -0.6897, -2.7106, 2.2222, -1.0330, -0.8933, 1.1765, 0.0601, 1.5252, -0.3630, 0.6728,
            ]]),
            &device,
        );

        let output = module.forward(input);

        output.to_data().assert_approx_eq(
            &Data::from([[
                -0.4990, -1.9680, 1.6178, -0.7486, -0.6470, 0.8576, 0.0461, 1.1111, -0.2614, 0.4915,
            ]]),
            3,
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn layer_norm_backward() {
        let device = Default::default();
        let module = LayerNormConfig::new(2).init::<TestAutodiffBackend>(&device);
        let tensor_1 = Tensor::<TestAutodiffBackend, 2>::from_data(
            Data::from([[0.0, 1.0], [3.0, 4.0]]),
            &device,
        )
        .require_grad();
        let tensor_2 = Tensor::<TestAutodiffBackend, 2>::from_data(
            Data::from([[6.0, 7.0], [9.0, 10.0]]),
            &device,
        )
        .require_grad();

        let x = tensor_1.clone().matmul(tensor_2.clone());

        let output = module.forward(x);
        let grads = output.backward();

        let tensor_1_grad = tensor_1.grad(&grads).unwrap();
        let tensor_2_grad = tensor_2.grad(&grads).unwrap();

        tensor_1_grad
            .to_data()
            .assert_approx_eq(&Data::zeros(tensor_1_grad.shape()), 3);
        tensor_2_grad
            .to_data()
            .assert_approx_eq(&Data::zeros(tensor_2_grad.shape()), 3);
    }
}
