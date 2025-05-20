/// Burn implementation of the DropPath (Stochastic Depth) regularization layer.
///
/// Papers:
/// DropBlock: A regularization method for convolutional networks (https://arxiv.org/abs/1810.12890)
///
/// Deep Networks with Stochastic Depth (https://arxiv.org/abs/1603.09382)
///
/// Inspired by the python implementation from the timm library:
/// https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/drop.py
use burn_core as burn;
use burn_core::config::Config;
use burn_core::module::Module;
use burn_tensor::backend::Backend;
use burn_tensor::{Distribution, Tensor};

fn check_probability(prob: f64) -> f64 {
    if !(0.0..=1.0).contains(&prob) {
        panic!("Probability should be between 0 and 1, but got {}", prob);
    }
    prob
}

/// DropPath (stochastic depth) regularization.
///
/// ## Arguments
///
/// * `x`: Input tensor.
/// * `drop_prob`: Probability of dropping a path.
/// * `training`: Whether the model is in training mode.
/// * `scale_by_keep`: Whether to scale the output by `1 / (1 - drop_prob)`
///
/// ## Returns
///
/// * Output tensor with the same shape as the input tensor.
#[must_use]
pub fn drop_path<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    drop_prob: f64,
    training: bool,
    scale_by_keep: bool,
) -> Tensor<B, D> {
    _drop_path_sample(
        x,
        drop_prob,
        training,
        scale_by_keep,
        |shape, keep_prob, device| {
            Tensor::<B, D>::random(shape, Distribution::Bernoulli(keep_prob), device)
        },
    )
}

/// Internal implementation of DropPath.
///
/// Deferred to a separate function to allow for testing sampling.
///
/// ## Arguments
///
/// * `x`: Input tensor.
/// * `drop_prob`: Probability of dropping a path.
/// * `training`: Whether the model is in training mode.
/// * `scale_by_keep`: Whether to scale the output by `1 / (1 - drop_prob)`
/// * `sample`: Sampling function to generate the random tensor.
///
/// ## Returns
///
/// * Output tensor with the same shape as the input tensor.
#[inline(always)]
#[must_use]
fn _drop_path_sample<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    drop_prob: f64,
    training: bool,
    scale_by_keep: bool,
    sample: fn([usize; D], f64, &B::Device) -> Tensor<B, D>,
) -> Tensor<B, D> {
    check_probability(drop_prob);

    if !training || drop_prob == 0.0 {
        return x;
    }

    let keep_prob = 1.0 - drop_prob;

    let mut shape = [1; D];
    shape[0] = x.dims()[0];

    let random_tensor = sample(shape, keep_prob, &x.device());

    let random_tensor = if keep_prob > 0.0 && scale_by_keep {
        random_tensor.div_scalar(keep_prob)
    } else {
        random_tensor
    };

    x * random_tensor
}

/// Common introspection interface for DropPath modules.
pub trait DropPathMeta {
    fn drop_prob(&self) -> f64;
    fn keep_prob(&self) -> f64 {
        1.0 - self.drop_prob()
    }
    fn scale_by_keep(&self) -> bool;
}

/// Configuration for the DropPath module.
#[derive(Config, Debug)]
pub struct DropPathConfig {
    #[config(default = 0.0)]
    pub drop_prob: f64,

    #[config(default = true)]
    pub scale_by_keep: bool,
}

impl DropPathMeta for DropPathConfig {
    fn drop_prob(&self) -> f64 {
        self.drop_prob
    }

    fn scale_by_keep(&self) -> bool {
        self.scale_by_keep
    }
}

impl DropPathConfig {
    /// Initializes a new DropPath module.
    #[must_use]
    pub fn init(&self) -> DropPath {
        DropPath {
            drop_prob: check_probability(self.drop_prob),
            scale_by_keep: self.scale_by_keep,
        }
    }
}

/// The DropPath module.
///
/// Burn Module that implements the DropPath (Stochastic Depth) regularization.
#[derive(Module, Clone, Debug)]
pub struct DropPath {
    /// Probability of dropping a path.
    pub drop_prob: f64,

    /// Whether to scale the output by `1 / (1 - drop_prob)`.
    pub scale_by_keep: bool,
}

impl DropPathMeta for DropPath {
    fn drop_prob(&self) -> f64 {
        self.drop_prob
    }

    fn scale_by_keep(&self) -> bool {
        self.scale_by_keep
    }
}

impl DropPath {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [DropPath](DropPath) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    #[must_use]
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let training = B::ad_enabled();
        drop_path(input, self.drop_prob, training, self.scale_by_keep)
    }
}
