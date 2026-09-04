#![cfg_attr(docsrs, feature(doc_cfg))]
extern crate alloc;

pub use cubecl::hip::AmdDevice as RocmDevice;

/// The cubecl backend, under the name of the runtime this crate compiles in.
/// Every cubecl backend is the same type — a tensor's device is what says which
/// runtime it runs on.
pub type Rocm = burn_cubecl::Cube;
