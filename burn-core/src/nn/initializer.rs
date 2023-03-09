use burn_tensor::Shape;

use crate::config::Config;
use crate::tensor::backend::Backend;
use crate::tensor::{Distribution, ElementConversion, Tensor};

use crate as burn;

#[derive(Config, Debug, PartialEq)]
pub enum Initializer {
    Uniform(f64, f64),
    UniformDefault,
    Normal(f64, f64),
    Constant(f64),
    Ones,
    Zeros,
    // TODO: add Xavier initialization
}

impl Initializer {
    pub fn init<B: Backend, const D: usize, S: Into<Shape<D>>>(&self, shape: S) -> Tensor<B, D> {
        match self {
            Self::Uniform(a, b) => Tensor::<B, D>::random(
                shape,
                Distribution::Uniform((*a).elem::<B::FloatElem>(), (*b).elem::<B::FloatElem>()),
            ),
            Self::UniformDefault => unimplemented!("The caller should implement the default"),
            Self::Normal(mean, std) => {
                Tensor::<B, D>::random(shape, Distribution::Normal(*mean, *std))
            }
            Self::Constant(value) => Tensor::<B, D>::zeros(shape) + *value, //TODO replace with fill()
            Self::Ones => Tensor::<B, D>::ones(shape),
            Self::Zeros => Tensor::<B, D>::zeros(shape),
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use burn_tensor::Data;

    pub type TB = burn_ndarray::NdArrayBackend<f32>;

    #[test]
    fn initializer_uniform_init() {
        // seed random generator
        TB::seed(0);
        let (a, b) = (0.0, 1.0);
        let uniform: Tensor<TB, 4> = Initializer::Uniform(a, b).init([2, 2, 2, 2]);
        for item in uniform.to_data().value.iter() {
            if *item < a as f32 || *item > b as f32 {
                panic!("Element ({item}) is not within range ({a},{b})");
            }
        }
    }

    #[test]
    #[should_panic]
    fn initializer_uniform_default_init() {
        let _: Tensor<TB, 4> = Initializer::UniformDefault.init([2, 2, 2, 2]);
    }

    #[test]
    fn initializer_normal_init() {
        // seed random generator
        TB::seed(0);
        let (mean, std) = (0.0, 1.0);
        let normal: Tensor<TB, 1> = Initializer::Normal(mean, std).init([1000]);
        let (var_act, mean_act) = normal.var_mean(0);
        var_act
            .to_data()
            .assert_approx_eq(&Data::from([std as f32]), 1);
        mean_act
            .to_data()
            .assert_approx_eq(&Data::from([mean as f32]), 1);
    }

    #[test]
    fn initializer_constant_init() {
        let value = 5.0;
        let constants: Tensor<TB, 4> = Initializer::Constant(value).init([2, 2, 2, 2]);
        constants
            .sum()
            .to_data()
            .assert_approx_eq(&Data::from([value as f32 * 16.0]), 3);
    }

    #[test]
    fn initializer_zeros_init() {
        let zeros: Tensor<TB, 4> = Initializer::Zeros.init([2, 2, 2, 2]);
        zeros
            .sum()
            .to_data()
            .assert_approx_eq(&Data::from([0.0]), 3);
    }

    #[test]
    fn initializer_ones_init() {
        let ones: Tensor<TB, 4> = Initializer::Ones.init([2, 2, 2, 2]);
        ones.sum()
            .to_data()
            .assert_approx_eq(&Data::from([16.0]), 3);
    }
}
