#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;

use burn_cubecl::CubeBackend;
pub use cubecl::cpu::CpuDevice;
use cubecl::cpu::CpuRuntime;

#[cfg(not(feature = "fusion"))]
pub type Cpu<F = f32, I = i32> = CubeBackend<CpuRuntime, F, I, u8>;

#[cfg(feature = "fusion")]
pub type Cpu<F = f32, I = i32> = burn_fusion::Fusion<CubeBackend<CpuRuntime, F, I, u8>>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::{Backend, DType, QTensorPrimitive};
    use burn_cubecl::tensor::CubeTensor;

    #[test]
    fn should_support_dtypes() {
        type B = Cpu;
        let device = Default::default();

        assert!(B::supports_dtype(&device, DType::F64));
        assert!(B::supports_dtype(&device, DType::F32));
        assert!(B::supports_dtype(&device, DType::F16));
        assert!(B::supports_dtype(&device, DType::BF16)); // does it actually work?
        assert!(B::supports_dtype(&device, DType::I64));
        assert!(B::supports_dtype(&device, DType::I32));
        assert!(B::supports_dtype(&device, DType::I16));
        assert!(B::supports_dtype(&device, DType::I8));
        assert!(B::supports_dtype(&device, DType::U64));
        assert!(B::supports_dtype(&device, DType::U32));
        assert!(B::supports_dtype(&device, DType::U16));
        assert!(B::supports_dtype(&device, DType::U8));
        assert!(B::supports_dtype(
            &device,
            DType::QFloat(CubeTensor::<CpuRuntime>::default_scheme())
        ));

        // Currently not registered in supported types
        assert!(!B::supports_dtype(&device, DType::Flex32));
        assert!(!B::supports_dtype(&device, DType::Bool));
    }
}
