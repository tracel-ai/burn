#![cfg_attr(docsrs, feature(doc_cfg))]
extern crate alloc;

use burn_cubecl::CubeBackend;

pub use cubecl::hip::AmdDevice as RocmDevice;

use cubecl::hip::HipRuntime;

#[cfg(not(feature = "fusion"))]
pub type Rocm = CubeBackend<HipRuntime>;

#[cfg(feature = "fusion")]
pub type Rocm = burn_fusion::Fusion<CubeBackend<HipRuntime>>;
