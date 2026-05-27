#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;

use burn_cubecl::CubeBackend;
pub use cubecl::cpu::CpuDevice;
use cubecl::cpu::CpuRuntime;

#[cfg(not(feature = "fusion"))]
pub type Cpu = CubeBackend<CpuRuntime>;

#[cfg(feature = "fusion")]
pub type Cpu = burn_fusion::Fusion<CubeBackend<CpuRuntime>>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{Backend, BoolStore, DType, DeviceOps};

    #[test]
    fn should_support_dtypes() {
        type B = Cpu;
        let device = CpuDevice;
        let scheme = device.defaults().quantization.scheme;

        assert!(B::supports_dtype(&device, DType::F64));
        assert!(B::supports_dtype(&device, DType::F32));
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
        assert!(B::supports_dtype(&device, DType::QFloat(scheme)));

        // Currently not registered in supported types
        assert!(!B::supports_dtype(&device, DType::Flex32));
        assert!(!B::supports_dtype(&device, DType::Bool(BoolStore::Native)));
    }
}
