#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;

use burn_cubecl::CubeBackend;
pub use cubecl::cuda::CudaDevice;
use cubecl::cuda::CudaRuntime;

#[cfg(not(feature = "fusion"))]
pub type Cuda<F = f32, I = i32> = CubeBackend<CudaRuntime, F, I, u8>;

#[cfg(feature = "fusion")]
pub type Cuda<F = f32, I = i32> = burn_fusion::Fusion<CubeBackend<CudaRuntime, F, I, u8>>;

#[cfg(test)]
mod tests {
    use super::*;
    use burn_cubecl::{CubeBackend, tensor::CubeTensor};
    use burn_tensor::{
        DType,
        backend::{Backend, QTensorPrimitive},
    };
    //use half::{bf16, f16};

    pub type TestRuntime = cubecl::cuda::CudaRuntime;

    // TODO: Add tests for bf16
    //burn_cubecl::testgen_all!([bf16, f16, f32], [i8, i16, i32, i64], [u8, u32]);
    burn_cubecl::testgen_all!([f32], [i32], [u32]);

    #[test]
    fn should_support_dtypes() {
        type B = Cuda;
        let device = Default::default();

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
        assert!(B::supports_dtype(&device, DType::Bool));
        assert!(B::supports_dtype(
            &device,
            DType::QFloat(CubeTensor::<TestRuntime>::default_scheme())
        ));

        // Currently not registered in supported types
        assert!(!B::supports_dtype(&device, DType::F64));
    }
}
