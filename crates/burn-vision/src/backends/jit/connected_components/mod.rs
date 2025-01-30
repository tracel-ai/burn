mod bke;
mod hardware_accelerated;

use burn_jit::{
    ops::numeric::{full_device, zeros_device},
    tensor::JitTensor,
    BoolElement, FloatElement, IntElement, JitBackend, JitRuntime,
};
use burn_tensor::Shape;
pub use hardware_accelerated::*;

use crate::{ConnectedStatsOptions, ConnectedStatsPrimitive};

pub(crate) fn stats_from_opts<R, F, I, BT>(
    l: JitTensor<R>,
    opts: ConnectedStatsOptions,
) -> ConnectedStatsPrimitive<JitBackend<R, F, I, BT>>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let [batches, height, width] = l.shape.dims();
    let shape = Shape::new([batches, height * width]);
    let zeros = || zeros_device::<R, u32>(l.client.clone(), l.device.clone(), shape.clone());
    let max = || full_device::<R, u32>(l.client.clone(), shape.clone(), l.device.clone(), u32::MAX);
    let dummy = || {
        JitTensor::new_contiguous(
            l.client.clone(),
            l.device.clone(),
            shape.clone(),
            l.handle.clone(),
            l.dtype,
        )
    };
    ConnectedStatsPrimitive {
        area: opts.area_enabled.then(zeros).unwrap_or_else(dummy),
        left: opts.left_enabled.then(max).unwrap_or_else(dummy),
        top: opts.top_enabled.then(max).unwrap_or_else(dummy),
        right: opts.right_enabled.then(zeros).unwrap_or_else(dummy),
        bottom: opts.bottom_enabled.then(zeros).unwrap_or_else(dummy),
    }
}
