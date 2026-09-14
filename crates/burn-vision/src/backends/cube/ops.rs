use crate::{
    BoolVisionOps, ConnectedStatsOptions, ConnectedStatsPrimitive, Connectivity, FloatVisionOps,
    IntVisionOps, VisionBackend, backends::cpu,
};
use burn_cubecl::CubeBackend;

use burn_core::backend::{
    TensorMetadata,
    ops::IntTensorOps,
    tensor::{BoolTensor, IntTensor},
};
use burn_core::tensor::IntDType;

use super::connected_components::hardware_accelerated;

impl BoolVisionOps for CubeBackend {
    fn connected_components(
        img: BoolTensor<Self>,
        connectivity: Connectivity,
        out_dtype: IntDType,
    ) -> IntTensor<Self> {
        if img.shape().num_elements() == 0 {
            return Self::int_zeros(img.shape(), &img.device(), out_dtype);
        }
        hardware_accelerated(
            img.clone(),
            ConnectedStatsOptions::none(),
            connectivity,
            out_dtype.into(),
        )
        .map(|it| it.0)
        .unwrap_or_else(|_| {
            let device = &img.device();
            Self::int_from_data(
                cpu::connected_components::<Self>(img, connectivity, out_dtype),
                device,
            )
        })
    }

    fn connected_components_with_stats(
        img: BoolTensor<Self>,
        connectivity: Connectivity,
        opts: ConnectedStatsOptions,
        out_dtype: IntDType,
    ) -> (IntTensor<Self>, ConnectedStatsPrimitive<Self>) {
        let device = &img.device();
        let capacity = crate::ops::connected_components_capacity(&img.shape());
        if img.shape().num_elements() == 0 {
            let zeros = |shape| Self::int_zeros(shape, device, out_dtype);
            return (
                zeros(img.shape()),
                ConnectedStatsPrimitive {
                    area: zeros([capacity].into()),
                    left: zeros([capacity].into()),
                    top: zeros([capacity].into()),
                    right: zeros([capacity].into()),
                    bottom: zeros([capacity].into()),
                    max_label: zeros([1].into()),
                },
            );
        }
        hardware_accelerated(img.clone(), opts, connectivity, out_dtype.into()).unwrap_or_else(
            |_| {
                let (labels, stats) = cpu::connected_components_with_stats_capacity::<Self>(
                    img,
                    connectivity,
                    out_dtype,
                    Some(capacity),
                );
                (Self::int_from_data(labels, device), stats)
            },
        )
    }
}

impl IntVisionOps for CubeBackend {}
impl FloatVisionOps for CubeBackend {}
impl VisionBackend for CubeBackend {}
