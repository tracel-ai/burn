mod hardware_accelerated;

/// Should eventually make this a full op, but the kernel is too specialized on ints and plane ops
/// to really use it in a general case. Needs more work to use as a normal tensor method.
mod prefix_sum;

use burn_cubecl::{
    ops::numeric::{full_device, zeros_device},
    tensor::CubeTensor,
    BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement,
};
use burn_tensor::Shape;
pub use hardware_accelerated::*;

use crate::{ConnectedStatsOptions, ConnectedStatsPrimitive};

pub(crate) fn stats_from_opts<R, F, I, BT>(
    l: CubeTensor<R>,
    opts: ConnectedStatsOptions,
) -> ConnectedStatsPrimitive<CubeBackend<R, F, I, BT>>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let [height, width] = l.shape.dims();
    let shape = Shape::new([height * width]);
    let zeros = || zeros_device::<R, I>(l.client.clone(), l.device.clone(), shape.clone());
    let max = I::max_value();
    let max = || full_device::<R, I>(l.client.clone(), shape.clone(), l.device.clone(), max);
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
        max_label: zeros_device::<R, I>(l.client.clone(), l.device.clone(), Shape::new([1])),
    }
}
