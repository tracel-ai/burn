use burn_tensor::Shape;
use libm::sqrt;

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
    XavierUniform(f64),
    XavierNormal(f64),
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
            Self::XavierUniform(gain) => xavier_uniform(gain, shape),
            Self::XavierNormal(gain) => xavier_normal(gain, shape),
        }
    }
}

fn xavier_uniform<B: Backend, const D: usize, S: Into<Shape<D>>>(
    gain: &f64,
    shape: S,
) -> Tensor<B, D> {
    let shape = shape.into();
    let bound = sqrt(3.0) * xavier_std(gain, &shape);
    Tensor::<B, D>::random(
        shape,
        Distribution::Uniform(
            (-bound).elem::<B::FloatElem>(),
            bound.elem::<B::FloatElem>(),
        ),
    )
}

fn xavier_normal<B: Backend, const D: usize, S: Into<Shape<D>>>(
    gain: &f64,
    shape: S,
) -> Tensor<B, D> {
    let shape = shape.into();
    let std = xavier_std(gain, &shape);
    Tensor::<B, D>::random(shape, Distribution::Normal(0.0, std))
}

fn xavier_std<const D: usize>(gain: &f64, shape: &Shape<D>) -> f64 {
    assert!(D >= 2);
    let fan_sum: usize = shape.dims.iter().take(2).sum();
    let receptive_field_size: usize = shape.dims.iter().skip(2).product();
    gain * sqrt(2.0 / (fan_sum * receptive_field_size) as f64)
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

        let var_act: f32 = var_act.into_scalar().elem();
        let mean_act: f32 = mean_act.into_scalar().elem();

        assert!(
            var_act > 0.9 && var_act < 1.1,
            "Expected variance to be between 1.0 += 0.1, but got {var_act}"
        );
        assert!(
            mean_act > -0.1 && mean_act < 0.1,
            "Expected mean to be between 0.0 += 0.1, but got {mean_act}"
        );
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

    #[test]
    fn initializer_xavier_uniform_init() {
        TB::seed(0);
        let gain = 2.;
        let (fan_in, fan_out) = (5, 6);
        let bound = gain * sqrt(6. / (fan_in + fan_out) as f64);
        let xavier_uniform: Tensor<TB, 2> =
            Initializer::XavierUniform(gain).init([fan_in, fan_out]);
        for item in xavier_uniform.to_data().value.iter() {
            if *item < -bound as f32 || *item > bound as f32 {
                panic!("Element ({item}) is not within range (-{bound},{bound})");
            }
        }
    }

    #[test]
    fn initializer_xavier_uniform_init_with_receptive_field() {
        TB::seed(0);
        let gain = 2.;
        let (fan_in, fan_out) = (5, 6);
        let (rec_field_1, rec_field_2) = (3, 4);
        let bound = gain * sqrt(6. / ((fan_in + fan_out) * rec_field_1 * rec_field_2) as f64);
        let xavier_uniform: Tensor<TB, 4> =
            Initializer::XavierUniform(gain).init([fan_in, fan_out, rec_field_1, rec_field_2]);
        for item in xavier_uniform.to_data().value.iter() {
            if *item < -bound as f32 || *item > bound as f32 {
                panic!("Element ({item}) is not within range (-{bound},{bound})");
            }
        }
    }

    #[test]
    fn initializer_xavier_normal_init() {
        TB::seed(0);
        let gain = 2.;
        let (fan_in, fan_out) = (1000, 10);
        let expected_mean = 0_f64;
        let expected_var = (gain * sqrt(2. / (fan_in as f64 + fan_out as f64))).powf(2.);
        let xavier_normal: Tensor<TB, 2> = Initializer::XavierNormal(gain).init([fan_in, fan_out]);
        let (actual_vars, actual_means) = xavier_normal.var_mean(0);
        for i in 0..fan_out {
            let actual_var = actual_vars.to_data().value[i] as f64;
            let actual_mean = actual_means.to_data().value[i] as f64;

            assert!(
                (expected_var - actual_var).abs() <= 0.1,
                "Expected variance to be between {expected_var} += 0.1, but got {actual_var}"
            );
            assert!(
                (expected_mean - actual_mean).abs() <= 0.1,
                "Expected mean to be between {expected_mean} += 0.1, but got {actual_mean}"
            );
        }
    }
}
