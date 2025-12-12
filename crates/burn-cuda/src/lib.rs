#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;

use burn_cubecl::CubeBackend;
pub use cubecl::cuda::CudaDevice;
use cubecl::cuda::CudaRuntime;

#[cfg(not(feature = "fusion"))]
pub type Cuda<F = f32, I = i32> = CubeBackend<CudaRuntime, F, I, u8>;

#[cfg(feature = "fusion")]
pub type Cuda<F = f32, I = i32> = burn_fusion::Fusion<CubeBackend<CudaRuntime, F, I, u8>>;
