use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::Param;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

/// Configuration to create a [InstanceNorm1d](InstanceNorm1d) layer.
#[derive(Config)]
pub struct InstanceNorm1dConfig {
    /// Number of features in the input tensor.
    pub num_features: i64,
    /// Epsilon value to avoid division by zero.
    pub eps: f64,
    /// Momentum value to update running statistics.
    pub momentum: f64,
    /// Whether to use the input tensor's mean and variance to normalize the tensor.
    pub affine: bool,
    /// Whether to use the input tensor's mean and variance to normalize the tensor.
    pub track_running_stats: bool,
}
