use alloc::vec::Vec;
use cubecl::std::throughput::measure_peak_throughput;

pub use cubecl::throughput::{ThroughputKey, ThroughputValue};

use crate::CubeRuntime;

/// Measure peak throughput on `device` for each of the given `keys`.
///
/// Runs cubecl-std's [`measure_peak_throughput`] against the runtime's compute
/// client. Generic over any [`CubeRuntime`]; concrete backends (cuda, wgpu, ...)
/// wrap this with their runtime type.
pub fn device_throughput<R: CubeRuntime>(
    device: &R::Device,
    keys: &[ThroughputKey],
) -> Vec<ThroughputValue> {
    let client = R::client(device);
    keys.iter()
        .map(|key| measure_peak_throughput::<R>(&client, *key))
        .collect()
}
