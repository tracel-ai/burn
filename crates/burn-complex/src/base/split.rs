use burn_tensor::{ElementComparison, Float, TensorData, TensorKind, TensorMetadata, backend::Backend};
use bytemuck::Pod;

use crate::base::{ComplexTensor, ComplexTensorBackend, ComplexTensorOps, Layout, SplitLayout, element::Complex};

impl<T: TensorMetadata + 'static> Layout for SplitLayout<T> {
    type ComplexTensorPrimitive = SplitComplexTensor<T>;
}

#[derive(Debug, Clone)]
pub struct SplitComplexTensor<P: TensorMetadata> {
    pub real: P,
    pub imag: P,
}

impl<T: TensorMetadata+ 'static> TensorMetadata for SplitComplexTensor<T>
{
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

/// A newtype that wraps a real backend B and exposes a split-layout complex backend.
pub struct SplitBackend<B: Backend>(core::marker::PhantomData<B>);

impl<B: Backend> ComplexTensorBackend for SplitBackend<B>
where
    <B as Backend>::FloatElem: ElementComparison + Pod,
    B::FloatTensorPrimitive: TensorMetadata + 'static,
{
    type InnerBackend = B;
    type ComplexScalar = Complex<B::FloatElem>;
    type Layout = SplitLayout<B::FloatTensorPrimitive>;

    fn complex_from_data(
        data: TensorData,
        device: &B::Device,
    ) -> ComplexTensor<Self> {
        // ComplexTensor<Self> = Complex<SplitComplexTensor<B::FloatTensorPrimitive>>
        // i.e. Complex { re: SplitComplexTensor { real, imag } }
        todo!()
    }
}

impl<B> ComplexTensorOps<SplitBackend<B>> for SplitBackend<B>
where
    B: Backend,
    <B as Backend>::FloatElem: ElementComparison + Pod,
{
    fn to_complex(tensor: super::FloatTensor<SplitBackend<B>>) -> ComplexTensor<SplitBackend<B>> {
        todo!()
    }

    async fn complex_into_data(
        tensor: ComplexTensor<SplitBackend<B>>,
    ) -> Result<TensorData, burn_tensor::backend::ExecutionError> {
        todo!()
    }

    fn complex_not_equal_elem(lhs: ComplexTensor<SplitBackend<B>>, rhs: <SplitBackend<B> as ComplexTensorBackend>::ComplexScalar)
    -> super::BoolTensor<SplitBackend<B>> {
        todo!()
    }
}

