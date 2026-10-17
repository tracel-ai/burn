#![cfg_attr(docsrs, feature(doc_cfg))]

//! The CUDA runtime for [Burn](https://github.com/tracel-ai/burn)'s CubeCL backend, for NVIDIA
//! GPUs.
//!
//! Applications enable Burn's `cuda` feature and create a device with `Device::cuda(index)`;
//! they do not need this crate directly. [`Cuda`] is the backend type under the name of this
//! runtime: every CubeCL runtime shares the same backend, and a tensor's device says which one it
//! runs on.
//!
//! The CUDA driver is loaded at runtime, so building does not require the CUDA toolkit.
//!
//! # Feature flags
//!
//! - `std` (default): standard library support.
//! - `fusion` (default): kernel fusion.
//! - `autotune` (default): benchmark kernel variants at runtime and keep the fastest.
//! - `tracing`: instrument operations with the `tracing` crate.

extern crate alloc;

pub use cubecl::cuda::CudaDevice;

/// The cubecl backend, under the name of the runtime this crate compiles in.
/// Every cubecl backend is the same type — a tensor's device is what says which
/// runtime it runs on.
pub type Cuda = burn_cubecl::Cube;

#[cfg(all(test, not(target_os = "macos")))]
mod tests {
    use super::*;
    use burn_backend::{Backend, BoolStore, DType, DeviceOps};

    #[test]
    fn should_support_dtypes() {
        type B = Cuda;
        let device = cubecl::Device::Cuda(CudaDevice::default());
        let scheme = device.defaults().quantization.scheme;

        assert!(B::supports_dtype(&device, DType::F32));
        assert!(B::supports_dtype(&device, DType::Flex32));
        assert!(B::supports_dtype(&device, DType::F16));
        assert!(B::supports_dtype(&device, DType::BF16));
        assert!(B::supports_dtype(&device, DType::I64));
        assert!(B::supports_dtype(&device, DType::I32));
        assert!(B::supports_dtype(&device, DType::I16));
        assert!(B::supports_dtype(&device, DType::I8));
        assert!(B::supports_dtype(&device, DType::U64));
        assert!(B::supports_dtype(&device, DType::U32));
        assert!(B::supports_dtype(&device, DType::U16));
        assert!(B::supports_dtype(&device, DType::U8));
        assert!(B::supports_dtype(&device, DType::Bool(BoolStore::Native)));
        assert!(B::supports_dtype(&device, DType::QFloat(scheme)));

        // Currently not registered in supported types
        assert!(!B::supports_dtype(&device, DType::F64));
    }
}
