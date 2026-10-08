#![cfg_attr(docsrs, feature(doc_cfg))]

//! The CubeCL CPU runtime for [Burn](https://github.com/tracel-ai/burn): the GPU backends' kernels,
//! compiled for the CPU through LLVM.
//!
//! Applications enable Burn's `cpu` feature and create a device with `Device::cpu()`; they do not
//! need this crate directly. [`Cpu`] is the backend type under the name of this runtime: every
//! CubeCL runtime shares the same backend, and a tensor's device says which one it runs on. LLVM
//! is bundled, so no system installation is needed.
//!
//! This is one of two independent CPU backends. `burn-flex` (feature `flex`) is a pure-Rust eager
//! backend that also supports `no_std` and WebAssembly; this one brings kernel fusion, autotuning
//! and custom CubeCL kernels to the CPU.
//!
//! # Feature flags
//!
//! - `std` (default): standard library support.
//! - `fusion` (default): kernel fusion.
//! - `autotune` (default): benchmark kernel variants at runtime and keep the fastest.
//! - `tracing`: instrument operations with the `tracing` crate.

extern crate alloc;

pub use cubecl::cpu::CpuDevice;

/// The cubecl backend, under the name of the runtime this crate compiles in.
/// Every cubecl backend is the same type — a tensor's device is what says which
/// runtime it runs on.
pub type Cpu = burn_cubecl::Cube;

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{Backend, BoolStore, DType, DeviceOps};

    #[test]
    fn should_support_dtypes() {
        type B = Cpu;
        let device = cubecl::Device::Cpu(CpuDevice);
        let scheme = device.defaults().quantization.scheme;

        assert!(B::supports_dtype(&device, DType::F64));
        assert!(B::supports_dtype(&device, DType::F32));
        assert!(B::supports_dtype(&device, DType::F16));
        assert!(B::supports_dtype(&device, DType::I64));
        assert!(B::supports_dtype(&device, DType::I32));
        assert!(B::supports_dtype(&device, DType::I16));
        assert!(B::supports_dtype(&device, DType::I8));
        assert!(B::supports_dtype(&device, DType::U64));
        assert!(B::supports_dtype(&device, DType::U32));
        assert!(B::supports_dtype(&device, DType::U16));
        assert!(B::supports_dtype(&device, DType::U8));
        assert!(B::supports_dtype(&device, DType::QFloat(scheme)));

        // Currently not registered in supported types
        assert!(!B::supports_dtype(&device, DType::Flex32));
        assert!(!B::supports_dtype(&device, DType::Bool(BoolStore::Native)));
        // BF16 is dropped: the LLVM dialect has no bfloat type to compute with.
        assert!(!B::supports_dtype(&device, DType::BF16));
    }
}
