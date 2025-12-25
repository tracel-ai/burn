use crate::tensor::Shape;

use crate::config::Config;
use crate::module::{Param, ParamId};
use crate::tensor::backend::Backend;
use crate::tensor::{Distribution, Tensor};

use crate as burn;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// Enum specifying with what values a tensor should be initialized
#[derive(Config, Debug, PartialEq)]
pub enum Initializer {
    /// Fills tensor with specified value everywhere
    Constant {
        /// The value to fill the tensor with
        value: f64,
    },
    /// Fills tensor with 1s everywhere
    Ones,
    /// Fills tensor with 0s everywhere
    Zeros,
    /// Fills tensor with values drawn uniformly between specified values
    Uniform {
        /// The minimum value to draw from
        min: f64,

        /// The maximum value to draw from
        max: f64,
    },
    /// Fills tensor with values drawn from normal distribution with specified mean and std
    Normal {
        /// The mean of the normal distribution
        mean: f64,

        /// The standard deviation of the normal distribution
        std: f64,
    },
    /// Fills tensor with values according to the uniform version of Kaiming initialization
    KaimingUniform {
        /// The gain to use in initialization formula
        gain: f64,

        /// Whether to use fan out only in initialization formula
        fan_out_only: bool,
    },
    /// Fills tensor with values according to the uniform version of Kaiming initialization
    KaimingNormal {
        /// The gain to use in initialization formula
        gain: f64,

        /// Whether to use fan out only in initialization formula
        fan_out_only: bool,
    },
    /// Fills tensor with values according to the uniform version of Xavier Glorot initialization
    /// described in [Understanding the difficulty of training deep feedforward neural networks
    /// ](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
    XavierUniform {
        /// The gain to use in initialization formula
        gain: f64,
    },
    /// Fills tensor with values according to the normal version of Xavier Glorot initialization
    /// described in [Understanding the difficulty of training deep feedforward neural networks
    /// ](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
    XavierNormal {
        /// The gain to use in initialization formula
        gain: f64,
    },
    /// Fills tensor with values according to the (semi) orthogonal initialization
    /// described in [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks`
    ///  - [Saxe, A. et al. (2013)](https://arxiv.org/abs/1312.6120)
    Orthogonal {
        /// The gain to use in initialization formula
        gain: f64,
    },
}

impl Initializer {
    /// Inits a tensor parameter of given shape with values depending on initializer kind.
    ///
    /// # Params
    ///
    /// - shape: Shape of the initiated tensor.
    pub fn init<B: Backend, const D: usize, S: Into<Shape>>(
        &self,
        shape: S,
        device: &B::Device,
    ) -> Param<Tensor<B, D>> {
        self.init_with(shape, None, None, device)
    }

    /// Inits a tensor parameter of given shape with values depending on initializer kind.
    ///
    /// # Params
    ///
    /// - shape: Shape of the initiated tensor.
    pub fn init_with<B: Backend, const D: usize, S: Into<Shape>>(
        &self,
        shape: S,
        fan_in: Option<usize>,
        fan_out: Option<usize>,
        device: &B::Device,
    ) -> Param<Tensor<B, D>> {
        let device = device.clone();
        let shape: Shape = shape.into();
        let config = self.clone();
        let shape_for_closure = shape.clone();

        Param::uninitialized(
            ParamId::new(),
            move |device, require_grad| {
                B::memory_persistent_allocations(device, (), move |_| {
                    let mut tensor = config.init_tensor(shape.clone(), fan_in, fan_out, device);

                    if require_grad {
                        tensor = tensor.require_grad();
                    }

                    tensor
                })
            },
            device,
            true,
            shape_for_closure,
        )
    }

    fn init_tensor<B: Backend, const D: usize, S: Into<Shape>>(
        &self,
        shape: S,
        fan_in: Option<usize>,
        fan_out: Option<usize>,
        device: &B::Device,
    ) -> Tensor<B, D> {
        let shape = shape.into();
        match self {
            Initializer::Constant { value } => Tensor::<B, D>::full(shape, *value, device),
            Initializer::Ones => Tensor::<B, D>::ones(shape, device),
            Initializer::Zeros => Tensor::<B, D>::zeros(shape, device),
            Initializer::Uniform { min, max } => uniform_draw(shape, *min, *max, device),
            Initializer::Normal { mean, std } => normal_draw(shape, *mean, *std, device),
            Initializer::KaimingUniform { gain, fan_out_only } => {
                let a = 3.0f64.sqrt() * *gain * self.kaiming_std(*fan_out_only, fan_in, fan_out);
                uniform_draw(shape, -a, a, device)
            }
            Initializer::KaimingNormal { gain, fan_out_only } => {
                let std = *gain * self.kaiming_std(*fan_out_only, fan_in, fan_out);
                normal_draw(shape, 0.0, std, device)
            }
            Initializer::XavierUniform { gain } => {
                let a = 3.0f64.sqrt() * *gain * self.xavier_std(fan_in, fan_out);
                uniform_draw(shape, -a, a, device)
            }
            Initializer::XavierNormal { gain } => {
                let std = *gain * self.xavier_std(fan_in, fan_out);
                normal_draw(shape, 0.0, std, device)
            }
            Initializer::Orthogonal { gain } => {
                // following the implementation in pytorch:
                // https://github.com/pytorch/pytorch/blob/v2.7.0/torch/nn/init.py#L574

                assert!(
                    D >= 2,
                    "Expected D (in Tensor<B, D>) to be greater or equal 2; (D >= 2)"
                );

                let rows: usize = shape.dims::<D>()[0];
                let cols: usize = shape.num_elements() / rows;

                let mut t: Tensor<B, 2> = normal_draw([rows, cols], 0.0, 1.0, device);

                if rows < cols {
                    t = t.transpose();
                }

                let (q, r) = crate::tensor::linalg::qr_decomposition(t);
                let [r_rows, r_cols] = r.clone().dims();

                let diag_r = Tensor::<B, 2>::ones([1, r_rows], device)
                    .matmul(Tensor::<B, 2>::eye(r_cols, device).mul(r.clone()));

                let ph = diag_r.clone().sign();

                let mut q = q.mul(ph);

                if rows < cols {
                    q = q.transpose();
                }

                q.reshape(shape).mul_scalar(*gain)
            }
        }
    }

    fn kaiming_std(
        &self,
        fan_out_only: bool,
        fan_in: Option<usize>,
        fan_out: Option<usize>,
    ) -> f64 {
        let fan = if fan_out_only { fan_out } else { fan_in };
        let fan = fan.expect(
            "Can't use Kaiming initialization without specifying fan. Use init_with method.",
        );

        1.0 / (fan as f64).sqrt()
    }

    fn xavier_std(&self, fan_in: Option<usize>, fan_out: Option<usize>) -> f64 {
        let fan_in = fan_in.expect(
            "Can't use Xavier initialization without specifying fan in. Use init_with method and \
             provide fan_in.",
        );
        let fan_out = fan_out.expect(
            "Can't use Xavier initialization without specifying fan out. Use init_with method and \
             provide fan_out.",
        );
        (2.0 / (fan_in + fan_out) as f64).sqrt()
    }
}

fn uniform_draw<B: Backend, const D: usize, S: Into<Shape>>(
    shape: S,
    low: f64,
    high: f64,
    device: &B::Device,
) -> Tensor<B, D> {
    let distribution = Distribution::Uniform(low, high);
    Tensor::<B, D>::random(shape, distribution, device)
}

fn normal_draw<B: Backend, const D: usize, S: Into<Shape>>(
    shape: S,
    mean: f64,
    std: f64,
    device: &B::Device,
) -> Tensor<B, D> {
    let distribution = Distribution::Normal(mean, std);
    Tensor::<B, D>::random(shape, distribution, device)
}

#[cfg(test)]
mod tests {
    use super::*;

    use burn_tensor::{ElementConversion, TensorData};
    use num_traits::Pow;

    pub type TB = burn_ndarray::NdArray<f32>;
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TB>;

    fn assert_normal_init(expected_mean: f64, expected_var: f64, tensor: &Tensor<TB, 2>) {
        let (actual_vars, actual_means) = tensor.clone().var_mean(0);
        let actual_vars = actual_vars.to_data();
        let actual_vars = actual_vars.as_slice::<FT>().unwrap();
        let actual_means = actual_means.to_data();
        let actual_means = actual_means.as_slice::<FT>().unwrap();

        for i in 0..tensor.shape().dims[0] {
            let actual_var = actual_vars[i] as f64;
            let actual_mean = actual_means[i] as f64;

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
        let device = Default::default();
        TB::seed(&device, 0);

        let (min, max) = (0.0, 1.0);
        let uniform = Initializer::Uniform { min, max };
        let tensor: Tensor<TB, 4> = uniform.init([2, 2, 2, 2], &Default::default()).into_value();

        tensor
            .into_data()
            .assert_within_range::<FT>(min.elem()..max.elem());
    }

    #[test]
    fn initializer_normal_init() {
        // seed random generator
        let device = Default::default();
        TB::seed(&device, 0);

        let (mean, std) = (0.0, 1.0);
        let normal: Tensor<TB, 1> = Initializer::Normal { mean, std }
            .init([1000], &Default::default())
            .into_value();
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
        let constants: Tensor<TB, 4> = Initializer::Constant { value }
            .init([2, 2, 2, 2], &Default::default())
            .into_value();
        constants.sum().to_data().assert_approx_eq::<FT>(
            &TensorData::from([value as f32 * 16.0]),
            Tolerance::default(),
        );
    }

    #[test]
    fn initializer_zeros_init() {
        let zeros: Tensor<TB, 4> = Initializer::Zeros
            .init([2, 2, 2, 2], &Default::default())
            .into_value();
        zeros
            .sum()
            .to_data()
            .assert_approx_eq::<FT>(&TensorData::from([0.0]), Tolerance::default());
    }

    #[test]
    fn initializer_ones_init() {
        let ones: Tensor<TB, 4> = Initializer::Ones
            .init([2, 2, 2, 2], &Default::default())
            .into_value();
        ones.sum()
            .to_data()
            .assert_approx_eq::<FT>(&TensorData::from([16.0]), Tolerance::default());
    }

    #[test]
    fn initializer_kaiming_uniform_init() {
        let device = Default::default();
        TB::seed(&device, 0);

        let gain = 2_f64;
        let (fan_in, fan_out) = (5, 6);
        let k = (gain * (3.0 / fan_in as f64).sqrt()).elem::<FT>();

        let tensor: Tensor<TB, 2> = Initializer::KaimingUniform {
            gain,
            fan_out_only: false,
        }
        .init_with([fan_out, fan_in], Some(fan_in), None, &Default::default())
        .into_value();
        tensor.into_data().assert_within_range(-k..k);
    }

    #[test]
    fn initializer_kaiming_normal_init() {
        let device = Default::default();
        TB::seed(&device, 0);

        let gain = 2.;
        let (fan_in, fan_out) = (1000, 10);
        let expected_mean = 0_f64;

        let expected_var = (gain * (1. / (fan_in as f64)).sqrt()).pow(2.);
        let tensor: Tensor<TB, 2> = Initializer::KaimingNormal {
            gain,
            fan_out_only: false,
        }
        .init_with([fan_out, fan_in], Some(fan_in), None, &Default::default())
        .into_value();
        assert_normal_init(expected_mean, expected_var, &tensor)
    }

    #[test]
    fn initializer_kaiming_uniform_init_bias() {
        let device = Default::default();
        TB::seed(&device, 0);

        let gain = 2_f64;
        let shape = [3];
        let fan_in = 5;
        let k = (gain * (3.0 / fan_in as f64).sqrt()).elem::<FT>();

        let tensor: Tensor<TB, 1> = Initializer::KaimingUniform {
            gain,
            fan_out_only: false,
        }
        .init_with(shape, Some(fan_in), None, &Default::default())
        .into_value();
        tensor.into_data().assert_within_range(-k..k);
    }

    #[test]
    fn initializer_kaiming_uniform_init_fan_out() {
        let device = Default::default();
        TB::seed(&device, 0);

        let gain = 2_f64;
        let (fan_in, fan_out) = (5, 6);
        let k = (gain * (3.0 / fan_out as f64).sqrt()).elem::<FT>();

        let tensor: Tensor<TB, 2> = Initializer::KaimingUniform {
            gain,
            fan_out_only: true,
        }
        .init_with([fan_out, fan_in], None, Some(fan_out), &Default::default())
        .into_value();
        tensor.into_data().assert_within_range(-k..k);
    }

    #[test]
    #[should_panic]
    fn initializer_kaiming_uniform_no_fan() {
        let device = Default::default();
        TB::seed(&device, 0);

        let gain = 2_f64;
        let (fan_in, fan_out) = (5, 6);

        let _: Tensor<TB, 2> = Initializer::KaimingUniform {
            gain,
            fan_out_only: false,
        }
        .init([fan_out, fan_in], &Default::default())
        .into_value();
    }

    #[test]
    fn initializer_xavier_uniform_init() {
        let device = Default::default();
        TB::seed(&device, 0);

        let gain = 2.;
        let (fan_in, fan_out) = (5, 6);
        let bound = (gain * (6. / (fan_in + fan_out) as f64).sqrt()).elem::<FT>();
        let tensor: Tensor<TB, 2> = Initializer::XavierUniform { gain }
            .init_with(
                [fan_out, fan_in],
                Some(fan_in),
                Some(fan_out),
                &Default::default(),
            )
            .into_value();

        tensor.into_data().assert_within_range(-bound..bound);
    }

    #[test]
    fn initializer_xavier_normal_init() {
        let device = Default::default();
        TB::seed(&device, 0);

        let gain = 2.;
        let (fan_in, fan_out) = (1000, 10);
        let expected_mean = 0_f64;

        let expected_var = (gain * (2. / (fan_in as f64 + fan_out as f64)).sqrt()).powf(2.);
        let tensor: Tensor<TB, 2> = Initializer::XavierNormal { gain }
            .init_with(
                [fan_out, fan_in],
                Some(fan_in),
                Some(fan_out),
                &Default::default(),
            )
            .into_value();
        assert_normal_init(expected_mean, expected_var, &tensor)
    }

    #[test]
    #[should_panic]
    fn initializer_xavier_uniform_no_fan() {
        let device = Default::default();
        TB::seed(&device, 0);

        let gain = 2.;
        let (fan_in, fan_out) = (5, 6);
        let _: Tensor<TB, 2> = Initializer::XavierUniform { gain }
            .init([fan_out, fan_in], &Default::default())
            .into_value();
    }

    #[test]
    fn test_qr_decomposition() {
        let device = Default::default();
        TB::seed(&device, 0);

        // test values follow the example from https://pytorch.org/docs/stable/generated/torch.linalg.qr.html#torch.linalg.qr
        let a = Tensor::<TB, 2>::from_floats(
            [[12., -51., 4.], [6., 167., -68.], [-4., 24., -41.]],
            &Default::default(),
        );
        let qr = crate::tensor::linalg::qr_decomposition(a.clone());

        // Q @ R should reconstruct input `a`
        let q_matmul_r = qr.0.clone().matmul(qr.1.clone());

        // assert that the difference between input (`a`) and Q @ R is (almost) zero
        q_matmul_r
            .into_data()
            .assert_approx_eq::<FT>(&a.into_data(), Tolerance::rel_abs(0.1, 0.1));
    }

    #[test]
    fn initializer_orthogonal_correct() {
        let device = Default::default();
        TB::seed(&device, 0);

        let gain = 1.;

        // test 2D tensor
        let size = 10;
        let q: Tensor<TB, 2> = Initializer::Orthogonal { gain }
            .init([size, size], &Default::default())
            .into_value();
        let eye = Tensor::<TB, 2>::eye(size, &Default::default());

        // Q.T @ Q should be close to identity matrix
        q.clone()
            .transpose()
            .matmul(q)
            .into_data()
            .assert_approx_eq::<FT>(&eye.into_data(), Tolerance::rel_abs(0.1, 0.1));
    }

    #[test]
    fn initializer_orthogonal_init() {
        let device = Default::default();
        TB::seed(&device, 0);

        let gain = 1.;

        // test 2D tensor
        let shape = [25, 30];
        let t: Tensor<TB, 2> = Initializer::Orthogonal { gain }
            .init(shape, &Default::default())
            .into_value();
        let dims = t.dims();
        assert_eq!(
            shape, dims,
            "Expected the shape of the input tensor to match the shape of the output. ({shape:?}, {dims:?})"
        );

        // test 3D tensor
        let shape = [24, 6, 85];
        let t: Tensor<TB, 3> = Initializer::Orthogonal { gain }
            .init(shape, &Default::default())
            .into_value();
        let dims = t.dims();
        assert_eq!(
            shape, dims,
            "Expected the shape of the input tensor to match the shape of the output. ({shape:?}, {dims:?})"
        );
    }

    #[test]
    #[should_panic]
    fn initializer_orthogonal_init_1d() {
        let device = Default::default();
        TB::seed(&device, 0);

        let gain = 1.;

        // test 1D tensor
        let shape = [3];
        let _: Tensor<TB, 1> = Initializer::Orthogonal { gain }
            .init(shape, &Default::default())
            .into_value();
    }
}
