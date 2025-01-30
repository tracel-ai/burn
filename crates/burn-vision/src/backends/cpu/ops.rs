use crate::{
    backends::cpu, ConnectedStatsOptions, ConnectedStatsPrimitive, Connectivity, VisionOps,
};
use burn_ndarray::{FloatNdArrayElement, IntNdArrayElement, NdArray, QuantElement};
use burn_tensor::ops::{BoolTensor, IntTensor};

impl<E, I, Q> VisionOps<Self> for NdArray<E, I, Q>
where
    E: FloatNdArrayElement,
    I: IntNdArrayElement,
    Q: QuantElement,
{
    fn connected_components(img: BoolTensor<Self>, connectivity: Connectivity) -> IntTensor<Self> {
        cpu::connected_components::<Self>(img, connectivity)
    }

    fn connected_components_with_stats(
        img: BoolTensor<Self>,
        connectivity: Connectivity,
        opts: ConnectedStatsOptions,
    ) -> (IntTensor<Self>, ConnectedStatsPrimitive<Self>) {
        cpu::connected_components_with_stats::<Self>(img, connectivity, opts)
    }
}
