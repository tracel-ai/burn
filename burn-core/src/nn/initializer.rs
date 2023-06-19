use burn_tensor::Shape;
use libm::sqrt;

use crate::config::Config;
use crate::tensor::backend::Backend;
use crate::tensor::{Distribution, ElementConversion, Tensor};

use crate as burn;

/// Enum specifying with what values a tensor should be initialized
#[derive(Config, Debug, PartialEq)]
pub enum Initializer {
    /// Fills tensor with specified value everywhere
    Constant { value: f64 },
    /// Fills tensor with 1s everywhere
    Ones,
    /// Fills tensor with 0s everywhere
    Zeros,
    /// Fills tensor with values drawn uniformly between specified values
    Uniform { min: f64, max: f64 },
    /// Fills tensor with values drawn from normal distribution with specified mean and std
    Normal { mean: f64, std: f64 },
    /// Fills tensor with values according to the uniform version of Kaiming initialization
    KaimingUniform { gain: f64, fan_out_only: bool },
    /// Fills tensor with values according to the uniform version of Kaiming initialization
    KaimingNormal { gain: f64, fan_out_only: bool },
    /// Fills tensor with values according to the uniform version of Xavier Glorot initialization described in [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
    XavierUniform { gain: f64 },
    /// Fills tensor with values according to the normal version of Xavier Glorot initialization described in [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
    XavierNormal { gain: f64 },
}

#[derive(Debug, Default)]
/// TODO DOC
pub struct InitializerOptions {
    fan_in: Option<usize>,
    fan_out: Option<usize>,
}

impl InitializerOptions {
    /// TODO DOC
    pub fn with_fan_in(mut self, fan_in: usize) -> Self {
        self.fan_in = Some(fan_in);
        self
    }

    /// TODO DOC
    pub fn with_fan_out(mut self, fan_out: usize) -> Self {
        self.fan_out = Some(fan_out);
        self
    }
}

impl Initializer {
    /// TODO DOC
    pub fn init<B: Backend, const D: usize, S: Into<Shape<D>>>(
        &self,
        shape: S,
        options: InitializerOptions,
    ) -> Tensor<B, D> {
        let shape = shape.into();
        match self {
            Initializer::Constant { value } => Tensor::<B, D>::zeros(shape) + *value, // TODO replace with fill()
            Initializer::Ones => Tensor::<B, D>::ones(shape),
            Initializer::Zeros => Tensor::<B, D>::zeros(shape),
            Initializer::Uniform { min, max } => uniform_draw(shape, *min, *max),
            Initializer::Normal { mean, std } => normal_draw(shape, *mean, *std),
            Initializer::KaimingUniform { gain, fan_out_only } => {
                let a = sqrt(3.0) * *gain * self.kaiming_std(*fan_out_only, options);
                uniform_draw(shape, -a, a)
            }
            Initializer::KaimingNormal { gain, fan_out_only } => {
                let std = *gain * self.kaiming_std(*fan_out_only, options);
                normal_draw(shape, 0.0, std)
            }
            Initializer::XavierUniform { gain } => {
                let a = sqrt(3.0) * *gain * self.xavier_std(options);
                uniform_draw(shape, -a, a)
            }
            Initializer::XavierNormal { gain } => {
                let std = *gain * self.xavier_std(options);
                normal_draw(shape, 0.0, std)
            }
        }
    }

    fn kaiming_std(&self, fan_out_only: bool, options: InitializerOptions) -> f64 {
        let fan = if fan_out_only {
            options.fan_out
        } else {
            options.fan_in
        };
        let fan = fan.expect("Can't use Kaiming initialization without specifying fan");

        1.0 / sqrt(fan as f64)
    }

    fn xavier_std(&self, options: InitializerOptions) -> f64 {
        let fan_in = options
            .fan_in
            .expect("Can't use Xavier initialization without specifying fan in");
        let fan_out = options
            .fan_out
            .expect("Can't use Xavier initialization without specifying fan out");
        sqrt(2.0 / (fan_in + fan_out) as f64)
    }
}

fn uniform_draw<B: Backend, const D: usize, S: Into<Shape<D>>>(
    shape: S,
    low: f64,
    high: f64,
) -> Tensor<B, D> {
    let distribution =
        Distribution::Uniform(low.elem::<B::FloatElem>(), high.elem::<B::FloatElem>());
    Tensor::<B, D>::random(shape, distribution)
}

fn normal_draw<B: Backend, const D: usize, S: Into<Shape<D>>>(
    shape: S,
    mean: f64,
    std: f64,
) -> Tensor<B, D> {
    let distribution = Distribution::Normal(mean, std);
    Tensor::<B, D>::random(shape, distribution)
}

#[cfg(test)]
mod tests {
    use super::*;

    use burn_tensor::Data;

    pub type TB = burn_ndarray::NdArrayBackend<f32>;

    fn assert_normal_init(expected_mean: f64, expected_var: f64, tensor: &Tensor<TB, 2>) {
        let (actual_vars, actual_means) = tensor.clone().var_mean(0);

        for i in 0..tensor.shape().dims[0] {
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

    #[test]
    fn initializer_uniform_init() {
        TB::seed(0);

        let (min, max) = (0.0, 1.0);
        let uniform = Initializer::Uniform { min, max };
        let tensor: Tensor<TB, 4> = uniform.init([2, 2, 2, 2], InitializerOptions::default());

        tensor.into_data().assert_within_range(min..max);
    }

    #[test]
    fn initializer_normal_init() {
        // seed random generator
        TB::seed(0);
        let (mean, std) = (0.0, 1.0);
        let normal: Tensor<TB, 1> =
            Initializer::Normal { mean, std }.init([1000], InitializerOptions::default());
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
        let constants: Tensor<TB, 4> =
            Initializer::Constant { value }.init([2, 2, 2, 2], InitializerOptions::default());
        constants
            .sum()
            .to_data()
            .assert_approx_eq(&Data::from([value as f32 * 16.0]), 3);
    }

    #[test]
    fn initializer_zeros_init() {
        let zeros: Tensor<TB, 4> =
            Initializer::Zeros.init([2, 2, 2, 2], InitializerOptions::default());
        zeros
            .sum()
            .to_data()
            .assert_approx_eq(&Data::from([0.0]), 3);
    }

    #[test]
    fn initializer_ones_init() {
        let ones: Tensor<TB, 4> =
            Initializer::Ones.init([2, 2, 2, 2], InitializerOptions::default());
        ones.sum()
            .to_data()
            .assert_approx_eq(&Data::from([16.0]), 3);
    }

    #[test]
    fn initializer_kaiming_uniform_init() {
        TB::seed(0);

        let gain = 2. as f64;
        let (fan_in, fan_out) = (5, 6);
        let k = gain * sqrt(3.0 / fan_in as f64);

        let tensor: Tensor<TB, 2> = Initializer::KaimingUniform {
            gain,
            fan_out_only: false,
        }
        .init(
            [fan_out, fan_in],
            InitializerOptions::default().with_fan_in(fan_in),
        );
        tensor.into_data().assert_within_range(-k..k);
    }

    #[test]
    fn initializer_kaiming_normal_init() {
        TB::seed(0);

        let gain = 2.;
        let (fan_in, fan_out) = (1000, 10);
        let expected_mean = 0_f64;

        let expected_var = (gain * sqrt(1. / (fan_in as f64))).powf(2.);
        let tensor: Tensor<TB, 2> = Initializer::KaimingNormal {
            gain,
            fan_out_only: false,
        }
        .init(
            [fan_out, fan_in],
            InitializerOptions::default().with_fan_in(fan_in),
        );
        assert_normal_init(expected_mean, expected_var, &tensor)
    }

    #[test]
    fn initializer_kaiming_uniform_init_bias() {
        TB::seed(0);

        let gain = 2. as f64;
        let shape = [3];
        let fan_in = 5;
        let k = gain * sqrt(3.0 / fan_in as f64);

        let tensor: Tensor<TB, 1> = Initializer::KaimingUniform {
            gain,
            fan_out_only: false,
        }
        .init(shape, InitializerOptions::default().with_fan_in(fan_in));
        tensor.into_data().assert_within_range(-k..k);
    }

    #[test]
    fn initializer_kaiming_uniform_init_fan_out() {
        TB::seed(0);

        let gain = 2. as f64;
        let (fan_in, fan_out) = (5, 6);
        let k = gain * sqrt(3.0 / fan_out as f64);

        let tensor: Tensor<TB, 2> = Initializer::KaimingUniform {
            gain,
            fan_out_only: true,
        }
        .init(
            [fan_out, fan_in],
            InitializerOptions::default().with_fan_out(fan_out),
        );
        tensor.into_data().assert_within_range(-k..k);
    }

    #[test]
    #[should_panic]
    fn initializer_kaiming_uniform_no_fan() {
        TB::seed(0);

        let gain = 2. as f64;
        let (fan_in, fan_out) = (5, 6);

        let _: Tensor<TB, 2> = Initializer::KaimingUniform {
            gain,
            fan_out_only: false,
        }
        .init([fan_out, fan_in], InitializerOptions::default());
    }

    #[test]
    fn initializer_xavier_uniform_init() {
        TB::seed(0);

        let gain = 2.;
        let (fan_in, fan_out) = (5, 6);
        let bound = gain * sqrt(6. / (fan_in + fan_out) as f64);
        let tensor: Tensor<TB, 2> = Initializer::XavierUniform { gain }.init(
            [fan_out, fan_in],
            InitializerOptions::default()
                .with_fan_in(fan_in)
                .with_fan_out(fan_out),
        );

        tensor.into_data().assert_within_range(-bound..bound);
    }

    #[test]
    fn initializer_xavier_normal_init() {
        TB::seed(0);

        let gain = 2.;
        let (fan_in, fan_out) = (1000, 10);
        let expected_mean = 0_f64;

        let expected_var = (gain * sqrt(2. / (fan_in as f64 + fan_out as f64))).powf(2.);
        let tensor: Tensor<TB, 2> = Initializer::XavierNormal { gain }.init(
            [fan_out, fan_in],
            InitializerOptions::default()
                .with_fan_in(fan_in)
                .with_fan_out(fan_out),
        );
        assert_normal_init(expected_mean, expected_var, &tensor)
    }

    #[test]
    #[should_panic]
    fn initializer_xavier_uniform_no_fan() {
        TB::seed(0);

        let gain = 2.;
        let (fan_in, fan_out) = (5, 6);
        let _: Tensor<TB, 2> = Initializer::XavierUniform { gain }
            .init([fan_out, fan_in], InitializerOptions::default());
    }
}
