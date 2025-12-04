#![cfg(target_os = "linux")]
#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;

use burn_cubecl::CubeBackend;
pub use cubecl::cpu::CpuDevice;
use cubecl::cpu::CpuRuntime;

#[cfg(not(feature = "fusion"))]
pub type Cpu<F = f32, I = i32> = CubeBackend<CpuRuntime, F, I, u8>;

#[cfg(feature = "fusion")]
pub type Cpu<F = f32, I = i32> = burn_fusion::Fusion<CubeBackend<CpuRuntime, F, I, u8>>;
