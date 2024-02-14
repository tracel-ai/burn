use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

#[derive(Config, Debug)]
pub struct InstanceNormConfig {
    /// The number of features.
    pub num_features: usize,
    /// A value required for numerical stability. Default: 1e-5
    #[config(default = 1e-5)]
    pub eps: f64,
    /// Momentum used to update the metrics. Default: 0.1
    #[config(default = 0.1)]
    pub momentum: f64,
    /// A boolean value that when set to `true`, this module has learnable
    /// affine parameters. Default: `false`
    #[config(default = false)]
    pub affine: bool,
}

#[derive(Module, Debug)]
pub struct InstanceNorm<B: Backend> {
    num_features: usize,
    eps: f64,
    momentum: f64,
    affine: bool,
    gamma: Option<Param<Tensor<B, 1>>>,
    beta: Option<Param<Tensor<B, 1>>>,
}
