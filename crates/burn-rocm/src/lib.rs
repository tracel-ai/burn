#![cfg_attr(docsrs, feature(doc_cfg))]
extern crate alloc;

use burn_cubecl::CubeBackend;

pub use cubecl::hip::AmdDevice as RocmDevice;

use cubecl::{
    hip::HipRuntime,
    throughput::{ThroughputError, ThroughputKey, ThroughputValue},
};

/// The cubecl backend, under the name of the runtime this crate compiles in.
/// Every cubecl backend is the same type — a tensor's device is what says which
/// runtime it runs on.
pub type Rocm = burn_cubecl::Cube;
