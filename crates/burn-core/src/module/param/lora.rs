use super::Param;
use burn_tensor::Tensor;

/// A LoRA (Low-Rank Adaptation) adapter attached to a frozen weight [parameter](Param).
///
/// When present on a `Param<Tensor<2>>`, the parameter materializes its effective value as
/// `base + scale * (a @ b)`, where `base` is the frozen (and optionally quantized) weight and
/// `a`/`b` are the trainable low-rank factors. The frozen base is the stored value of the
/// parameter; the adapter factors are surfaced to the optimizer, autodiff and record systems as
/// regular parameters with their own [`ParamId`](super::ParamId)s through the module
/// visitor/mapper traversal.
#[derive(Clone, Debug)]
pub struct LoraAdapter {
    /// Down-projection factor with shape `[d_in, rank]` (trainable).
    pub a: Param<Tensor<2>>,
    /// Up-projection factor with shape `[rank, d_out]` (trainable).
    pub b: Param<Tensor<2>>,
    /// Scaling factor applied to the low-rank product, typically `alpha / rank`.
    pub scale: f64,
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
