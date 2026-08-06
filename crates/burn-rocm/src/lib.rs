#![cfg_attr(docsrs, feature(doc_cfg))]
extern crate alloc;

use burn_cubecl::CubeBackend;

pub use cubecl::hip::AmdDevice as RocmDevice;

use cubecl::{
    hip::HipRuntime,
    throughput::{ThroughputKey, ThroughputValue},
};

#[cfg(not(feature = "fusion"))]
pub type Rocm = CubeBackend<HipRuntime>;

#[cfg(feature = "fusion")]
pub type Rocm = burn_fusion::Fusion<CubeBackend<HipRuntime>>;

/// Measure peak throughput on a ROCm `device` for each of the given `keys`.
pub fn device_throughput(
    device: &RocmDevice,
    keys: &[ThroughputKey],
) -> alloc::vec::Vec<ThroughputValue> {
    cubecl::std::throughput::device_throughput::<HipRuntime>(device, keys)
}
