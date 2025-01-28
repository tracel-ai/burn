use crate::{BoolElement, FloatElement, IntElement, JitBackend, JitRuntime};
use burn_tensor::ops::{BoolTensor, IntTensor};
use burn_vision::{
    cpu_impl, ConnectedStatsOptions, ConnectedStatsPrimitive, Connectivity, VisionOps,
};

use super::connected_components::hardware_accelerated;

impl<R, F, I, BT> VisionOps<Self> for JitBackend<R, F, I, BT>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn connected_components(img: BoolTensor<Self>, connectivity: Connectivity) -> IntTensor<Self> {
        hardware_accelerated::<R, F, I, BT>(
            img.clone(),
            ConnectedStatsOptions::none(),
            connectivity,
        )
        .map(|it| it.0)
        .unwrap_or_else(|_| cpu_impl::connected_components::<Self>(img, connectivity))
    }

    fn connected_components_with_stats(
        img: BoolTensor<Self>,
        connectivity: Connectivity,
        opts: ConnectedStatsOptions,
    ) -> (IntTensor<Self>, ConnectedStatsPrimitive<Self>) {
        hardware_accelerated::<R, F, I, BT>(img.clone(), opts, connectivity).unwrap_or_else(|_| {
            cpu_impl::connected_components_with_stats::<Self>(img, connectivity, opts)
        })
    }
}
