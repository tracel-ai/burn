#![cfg_attr(docsrs, feature(doc_cfg))]
extern crate alloc;

use burn_cubecl::CubeBackend;

pub use cubecl::hip::AmdDevice as RocmDevice;

use cubecl::{
    hip::HipRuntime,
    throughput::{ThroughputError, ThroughputKey, ThroughputValue},
};

#[cfg(not(feature = "fusion"))]
pub type Rocm = CubeBackend<HipRuntime>;

#[cfg(feature = "fusion")]
pub type Rocm = burn_fusion::Fusion<CubeBackend<HipRuntime>>;

/// Measure peak throughput on a ROCm `device` for each of the given `keys`.
///
/// One result per key, in order; a key the device has no peak for carries the
/// [`ThroughputError`] saying why.
pub fn device_throughput(
    device: &RocmDevice,
    keys: &[ThroughputKey],
) -> alloc::vec::Vec<Result<ThroughputValue, ThroughputError>> {
    cubecl::std::throughput::device_throughput::<HipRuntime>(device, keys)
}
