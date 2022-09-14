use burn_tensor::{ElementConversion, Shape};

use crate as burn;

use crate::macros::config;
use crate::module::Module;
use crate::module::{Forward, Param};
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

config!(
    pub struct LayerNormConfig {
        pub d_model: usize,
        pub epsilon: f32,
    }
);

#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    gamma: Param<Tensor<B, 1>>,
    beta: Param<Tensor<B, 1>>,
    epsilon: f32,
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
            .sub(&mean.detach())
            .div(&var.powf(0.5).add_scalar(&self.epsilon.to_elem()).detach());

        input_normalized
            .mul(&self.gamma.unsqueeze())
            .add(&self.beta.unsqueeze())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_tensor::Data;

    #[test]
    fn layer_norm_forward() {
        let config = LayerNormConfig {
            d_model: 10,
            epsilon: 1e-5,
        };
        let module = LayerNorm::<TestBackend>::new(&config);
        let input = Tensor::from_data(Data::from([[
            -0.6897, -2.7106, 2.2222, -1.0330, -0.8933, 1.1765, 0.0601, 1.5252, -0.3630, 0.6728,
        ]]));

        let output = module.forward(input);

        output.to_data().assert_approx_eq(
            &Data::from([[
                -0.4990, -1.9680, 1.6178, -0.7485, -0.6470, 0.8576, 0.0461, 1.1111, -0.2615, 0.4915,
            ]]),
            3,
        );
    }
}
