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

/// Connected components run on the host: the image is read back and labeled by the CPU
/// implementation, then uploaded again.
impl BoolVisionOps for CubeBackend {
    fn connected_components(
        img: BoolTensor<Self>,
        connectivity: Connectivity,
        out_dtype: IntDType,
    ) -> IntTensor<Self> {
        let device = &img.device();
        if img.shape().num_elements() == 0 {
            return Self::int_zeros(img.shape(), device, out_dtype);
        }
        Self::int_from_data(
            cpu::connected_components::<Self>(img, connectivity, out_dtype),
            device,
        )
    }

    fn connected_components_with_stats(
        img: BoolTensor<Self>,
        connectivity: Connectivity,
        opts: ConnectedStatsOptions,
        out_dtype: IntDType,
    ) -> (IntTensor<Self>, ConnectedStatsPrimitive<Self>) {
        let device = &img.device();
        if img.shape().num_elements() == 0 {
            let zeros = |shape| Self::int_zeros(shape, device, out_dtype);
            return (
                zeros(img.shape()),
                ConnectedStatsPrimitive {
                    area: zeros([1].into()),
                    left: zeros([1].into()),
                    top: zeros([1].into()),
                    right: zeros([1].into()),
                    bottom: zeros([1].into()),
                    max_label: zeros([1].into()),
                },
            );
        }
        let (labels, stats) =
            cpu::connected_components_with_stats::<Self>(img, connectivity, opts, out_dtype);
        (Self::int_from_data(labels, device), stats)
    }
}

impl IntVisionOps for CubeBackend {}
impl FloatVisionOps for CubeBackend {}
impl VisionBackend for CubeBackend {}
