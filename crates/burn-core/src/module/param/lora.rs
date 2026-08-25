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
        // Compose at the factors' dtype: cast a mismatched dense base, but leave
        // a packed base untouched so mixed addition dequantizes it directly to
        // the delta's dtype rather than through the device default.
        let base = if base.dtype().is_float() && base.dtype() != delta.dtype() {
            base.cast(delta.dtype())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_device;
    use burn_tensor::DType;

    #[test]
    fn materialize_casts_a_dense_base_to_the_factor_dtype() {
        let device = test_device();
        let base = Tensor::<2>::ones([4, 4], (&device, DType::F32));
        let adapter = LoraAdapter {
            a: Param::from_tensor(Tensor::<2>::ones([4, 2], (&device, DType::F16))),
            b: Param::from_tensor(Tensor::<2>::zeros([2, 4], (&device, DType::F16))),
            scale: 1.0,
        };

        let materialized = adapter.materialize(base);

        assert_eq!(materialized.dtype(), DType::F16);
    }
}
