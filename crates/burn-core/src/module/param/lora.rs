use super::{Param, Reparameterization};
use crate as burn;
use crate::module::Module;
use burn_tensor::Tensor;

/// A LoRA (Low-Rank Adaptation) adapter attached to a frozen weight [parameter](Param).
///
/// When present on a `Param<Tensor<2>>`, the parameter materializes its effective value as
/// `base + scale * (a @ b)`, where `base` is the frozen (and optionally quantized) weight and
/// `a`/`b` are the trainable low-rank factors. The frozen base is the stored value of the
/// parameter; the adapter factors are surfaced to the optimizer, autodiff and record systems as
/// regular parameters with their own [`ParamId`](super::ParamId)s through the module
/// visitor/mapper traversal.
#[derive(Debug, Module)]
pub struct LoraAdapter {
    /// Down-projection factor with shape `[d_in, rank]` (trainable).
    pub a: Param<Tensor<2>>,
    /// Up-projection factor with shape `[rank, d_out]` (trainable).
    pub b: Param<Tensor<2>>,
    /// Scaling factor applied to the low-rank product, typically `alpha / rank`.
    pub scale: f64,
}

impl Reparameterization for LoraAdapter {
    const NAME: &'static str = "lora";

    fn materialize<const D: usize>(&self, base: Tensor<D>) -> Tensor<D> {
        let delta = self.delta().reshape(base.shape());
        // The factors set the precision the compose runs at. A dense base was
        // matched to them when they were built; a packed base dequantizes to
        // the device default, which is not theirs whenever the model computes
        // at another precision.
        let base = if base.dtype() != delta.dtype() {
            base.dequantize().cast(delta.dtype())
        } else {
            base
        };
        base + delta
    }
}

impl LoraAdapter {
    /// Compute the low-rank delta `scale * (a @ b)` with shape `[d_in, d_out]`.
    ///
    /// `a` and `b` are read through [`Param::val`], so the delta always reflects the current
    /// (optimizer-updated) factors and keeps them as autodiff leaves for backpropagation.
    pub fn delta(&self) -> Tensor<2> {
        self.a.val().matmul(self.b.val()).mul_scalar(self.scale)
    }
}
