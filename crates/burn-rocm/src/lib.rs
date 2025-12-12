#![cfg_attr(docsrs, feature(doc_cfg))]
extern crate alloc;

use burn_cubecl::CubeBackend;

pub use cubecl::hip::AmdDevice as RocmDevice;

use cubecl::hip::HipRuntime;

#[cfg(not(feature = "fusion"))]
pub type Rocm<F = f32, I = i32, B = u8> = CubeBackend<HipRuntime, F, I, B>;

#[cfg(feature = "fusion")]
pub type Rocm<F = f32, I = i32, B = u8> = burn_fusion::Fusion<CubeBackend<HipRuntime, F, I, B>>;
