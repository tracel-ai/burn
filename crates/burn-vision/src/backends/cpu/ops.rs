use crate::VisionOps;

#[cfg(feature = "autodiff")]
use burn_autodiff::{checkpoint::strategy::CheckpointStrategy, Autodiff};
#[cfg(feature = "candle")]
use burn_candle::{Candle, FloatCandleElement, IntCandleElement};
#[cfg(feature = "ndarray")]
use burn_ndarray::{FloatNdArrayElement, IntNdArrayElement, NdArray, QuantElement};
#[cfg(feature = "tch")]
use burn_tch::{LibTorch, TchElement};

#[cfg(feature = "ndarray")]
impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> VisionOps<Self>
    for NdArray<E, I, Q>
{
}

#[cfg(feature = "candle")]
impl<F: FloatCandleElement, I: IntCandleElement> VisionOps<Self> for Candle<F, I> {}
#[cfg(feature = "tch")]
impl<E: TchElement, Q: burn_tch::QuantElement> VisionOps<Self> for LibTorch<E, Q> {}
#[cfg(feature = "autodiff")]
impl<B: burn_tensor::backend::Backend + VisionOps<B>, C: CheckpointStrategy> VisionOps<Self>
    for Autodiff<B, C>
{
    fn connected_components(
        img: burn_tensor::ops::BoolTensor<Self>,
        connectivity: crate::Connectivity,
    ) -> burn_tensor::ops::IntTensor<Self> {
        B::connected_components(img, connectivity)
    }

    fn connected_components_with_stats(
        img: burn_tensor::ops::BoolTensor<Self>,
        connectivity: crate::Connectivity,
        opts: crate::ConnectedStatsOptions,
    ) -> (
        burn_tensor::ops::IntTensor<Self>,
        crate::ConnectedStatsPrimitive<Self>,
    ) {
        let (labels, stats) = B::connected_components_with_stats(img, connectivity, opts);
        let stats = crate::ConnectedStatsPrimitive::<Self> {
            area: stats.area,
            left: stats.left,
            top: stats.top,
            right: stats.right,
            bottom: stats.bottom,
        };
        (labels, stats)
    }
}
