use crate as burn;

use crate::config;
use crate::module::Module;
use crate::module::{Forward, Param};
use crate::tensor::backend::Backend;
use crate::tensor::{ElementConversion, Shape, Tensor};

config!(
    /// Configuration to create a [LayerNorm](LayerNorm) layer.
    pub struct LayerNormConfig {
        /// The size of the input features.
        pub d_model: usize,
        /// A value required for numerical stability. Default: 1e-5
        #[config(default = 1e-5)]
        pub epsilon: f64,
    }
);

/// Applies Layer Normalization over an input tensor as described in the paper [Layer Normalization](https://arxiv.org/abs/1607.06450).
///
/// `Y = norm(X) * γ + β`
#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    gamma: Param<Tensor<B, 1>>,
    beta: Param<Tensor<B, 1>>,
    epsilon: f64,
}

impl<B: Backend> LayerNorm<B> {
    pub fn new(config: &LayerNormConfig) -> Self {
        let gamma = Tensor::ones(Shape::new([config.d_model]));
        let beta = Tensor::zeros(Shape::new([config.d_model]));

        Self {
            gamma: Param::new(gamma),
            beta: Param::new(beta),
            epsilon: config.epsilon,
        }
    }
}

impl<B: Backend, const D: usize> Forward<Tensor<B, D>, Tensor<B, D>> for LayerNorm<B> {
    fn forward(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let (var, mean) = input.var_mean_bias(D - 1);

        let input_normalized = input
            .sub(&mean)
            .div(&var.powf(0.5).add_scalar(&self.epsilon.to_elem()));

        input_normalized
            .mul(&self.gamma.unsqueeze())
            .add(&self.beta.unsqueeze())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TestADBackend, TestBackend};
    use burn_tensor::Data;

    #[test]
    fn layer_norm_forward() {
        let config = LayerNormConfig::new(10);
        let module = LayerNorm::<TestBackend>::new(&config);
        let input = Tensor::from_data(Data::from([[
            -0.6897, -2.7106, 2.2222, -1.0330, -0.8933, 1.1765, 0.0601, 1.5252, -0.3630, 0.6728,
        ]]));

        let output = module.forward(input);

        output.to_data().assert_approx_eq(
            &Data::from([[
                -0.4990, -1.9680, 1.6178, -0.7486, -0.6470, 0.8576, 0.0461, 1.1111, -0.2614, 0.4915,
            ]]),
            3,
        );
    }

    #[test]
    fn layer_norm_backward() {
        let config = LayerNormConfig::new(2);
        let module = LayerNorm::<TestADBackend>::new(&config);
        let tensor_1 = Tensor::<TestADBackend, 2>::from_data(Data::from([[0.0, 1.0], [3.0, 4.0]]));
        let tensor_2 = Tensor::<TestADBackend, 2>::from_data(Data::from([[6.0, 7.0], [9.0, 10.0]]));

        let x = tensor_1.matmul(&tensor_2);

        let output = module.forward(x);
        let grads = output.backward();

        let tensor_1_grad = tensor_1.grad(&grads).unwrap();
        let tensor_2_grad = tensor_2.grad(&grads).unwrap();
        let gamma_grad = module.gamma.grad(&grads).unwrap();
        let beta_grad = module.beta.grad(&grads).unwrap();

        gamma_grad
            .to_data()
            .assert_approx_eq(&Data::from([-2.0, 2.0]), 3);
        beta_grad
            .to_data()
            .assert_approx_eq(&Data::from([2.0, 2.0]), 3);
        tensor_1_grad
            .to_data()
            .assert_approx_eq(&Data::zeros(*tensor_1_grad.shape()), 3);
        tensor_2_grad
            .to_data()
            .assert_approx_eq(&Data::zeros(*tensor_2_grad.shape()), 3);
    }
}
