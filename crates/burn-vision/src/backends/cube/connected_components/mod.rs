mod hardware_accelerated;

/// Should eventually make this a full op, but the kernel is too specialized on ints and plane ops
/// to really use it in a general case. Needs more work to use as a normal tensor method.
mod prefix_sum;

use burn_core::{
    backend::cubecl::dtype_to_storage_type,
    tensor::{DType, Shape},
};
use burn_cubecl::{
    CubeBackend, CubeRuntime,
    ops::numeric::{full_device_dtype, zeros_client},
    tensor::CubeTensor,
};
use cubecl::prelude::InputScalar;
pub use hardware_accelerated::*;

use crate::{ConnectedStatsOptions, ConnectedStatsPrimitive, dispatch_int_dtype};

pub(crate) fn stats_from_opts<R: CubeRuntime>(
    l: CubeTensor<R>,
    opts: ConnectedStatsOptions,
    int_dtype: DType,
) -> ConnectedStatsPrimitive<CubeBackend<R>> {
    let [height, width] = l.meta.shape().dims();
    let shape = Shape::new([height * width]);
    let zeros = || zeros_client::<R>(l.client.clone(), l.device.clone(), shape.clone(), int_dtype);

    let max = dispatch_int_dtype!(int_dtype.into(), |I| InputScalar::new(
        I::MAX,
        dtype_to_storage_type(int_dtype)
    ));
    let max = || {
        full_device_dtype::<R>(
            l.client.clone(),
            shape.clone(),
            l.device.clone(),
            max,
            int_dtype,
        )
    };
    let dummy = || {
        CubeTensor::new_contiguous(
            l.client.clone(),
            l.device.clone(),
            shape.clone(),
            l.handle.clone(),
            l.dtype,
        )
    };
    ConnectedStatsPrimitive {
        area: (opts != ConnectedStatsOptions::none())
            .then(zeros)
            .unwrap_or_else(dummy),
        left: opts.bounds_enabled.then(max).unwrap_or_else(dummy),
        top: opts.bounds_enabled.then(max).unwrap_or_else(dummy),
        right: opts.bounds_enabled.then(zeros).unwrap_or_else(dummy),
        bottom: opts.bounds_enabled.then(zeros).unwrap_or_else(dummy),
        max_label: zeros_client::<R>(
            l.client.clone(),
            l.device.clone(),
            Shape::new([1]),
            int_dtype,
        ),
    }
}
