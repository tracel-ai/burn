use burn_tensor::{Float, TensorKind, TensorMetadata, backend::Backend};
use bytemuck::Pod;

use crate::base::{ComplexTensorBackend, SplitLayout, element::Complex};
#[derive(Debug, Clone)]
pub struct SplitComplexTensor<B, const D: usize, K= Float>
where
    B: Backend,
    K: TensorKind<B> {
    pub real: K::Primitive,
    pub imag: K::Primitive,
}

impl<B: Backend, const D: usize, K> TensorMetadata for SplitComplexTensor<B, D, K>
where
    K: TensorKind<B> {
    fn shape(&self) -> burn_tensor::Shape {
        self.real.shape()
    }

    fn rank(&self) -> usize {
        self.shape().num_dims()
    }
    
    fn dtype(&self) -> burn_std::DType {
        match self.real.dtype() {
            burn_std::DType::F32 => burn_std::DType::Complex32,
            burn_std::DType::F64 => burn_std::DType::Complex64,
            dtype => panic!("Unsupported dtype for complex tensor: {dtype:?}"),
        }
    }
}

impl<B: Backend, const D: usize, K> ComplexTensorBackend for SplitComplexTensor<B, D, K>
where
    B::FloatElem: burn_tensor::ElementComparison + Pod {
    type InnerBackend = B;

    type ComplexScalar = Complex<B::FloatElem>;

    type Layout = SplitLayout<Self>;
    
    fn complex_from_data(data: burn_tensor::TensorData, device: &burn_tensor::Device<Self>) -> super::ComplexTensor<Self> {
        todo!()
    }
}

