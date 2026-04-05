use burn_tensor::{ElementComparison, Float, TensorData, TensorKind, TensorMetadata, backend::Backend};
use bytemuck::Pod;

use crate::base::{ComplexDevice, ComplexTensor, ComplexTensorBackend, ComplexTensorOps, Layout, SplitLayout, element::Complex};

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

    fn complex_from_real_data(
        data: TensorData,
        device: &B::Device,
    ) -> ComplexTensor<Self> {
        // ComplexTensor<Self> = Complex<SplitComplexTensor<B::FloatTensorPrimitive>>
        // i.e. Complex { re: SplitComplexTensor { real, imag } }
        let real = B::float_from_data(data, device);
        let imag = B::float_zeros(real.shape().clone(), device, real.dtype().into());
        SplitComplexTensor { real, imag }
    }
    
    fn complex_from_imag_data(data: TensorData, device: &<Self::InnerBackend as Backend>::Device) -> ComplexTensor<Self> {
        let imag = B::float_from_data(data, device);
        let real = B::float_zeros(imag.shape().clone(), device, imag.dtype().into());
        SplitComplexTensor { real, imag }
    }
    // Should these be a result
    fn complex_from_interleaved_data(data: TensorData, device: &<Self::InnerBackend as Backend>::Device) -> ComplexTensor<Self> {
        // This is a bit more complex, we need to deinterleave the data into real and imag parts.
        // We can do this by creating two new tensors and copying the data into them.
        let interleaved = B::float_from_data(data, device);
        let shape = interleaved.shape();
        assert!(shape.num_dims() > 0, "Interleaved tensor must have at least one dimension");
        // let Some(last_dim) = shape.dims().iter().last() else {
        //     panic!("Interleaved tensor must have at least one dimension");
        // };
        // assert!( last_dim % 2 == 0, "Last dimension of interleaved tensor must be even");
        // // need to check if shape dims are 0-indexed or 1-indexed
        // let real_shape = shape.clone().reshape([shape.num_dims() - 1, last_dim / 2]).unwrap();
        // let imag_shape = real_shape.clone();
        // let real = B::float_zeros(real_shape, device, interleaved.dtype().into());
        // let imag = B::float_zeros(imag_shape, device, interleaved.dtype().into());
        // interleaved.iter().chunks(2).zip(real.iter_mut()).zip(imag.iter_mut()).for_each(|(((re, im), re_out), im_out)| {
        //     *re_out = *re;
        //     *im_out = *im;
        // });
        todo!("What now?")
    }
    
    fn complex_from_split_data(real_data: TensorData, imag_data: TensorData, device: &<Self::InnerBackend as Backend>::Device) -> ComplexTensor<Self> {
        let real = B::float_from_data(real_data, device);
        let imag = B::float_from_data(imag_data, device);
        assert_eq!(real.shape(), imag.shape(), "Real and imaginary parts must have the same shape");
        assert_eq!(real.dtype(), imag.dtype(), "Real and imaginary parts must have the same dtype");
        SplitComplexTensor { real, imag }
    }
}

impl<B> ComplexTensorOps<SplitBackend<B>> for SplitBackend<B>
where
    B: Backend,
    <B as Backend>::FloatElem: ElementComparison + Pod,
{
    fn to_complex(tensor: super::FloatTensor<SplitBackend<B>>) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            imag: B::float_zeros(tensor.shape().clone(), &<Self as ComplexTensorBackend>::InnerBackend::float_device(&tensor), tensor.dtype().into()),
            real: tensor,
        }
    }

    fn real(tensor: ComplexTensor<SplitBackend<B>>) -> super::FloatTensor<SplitBackend<B>> {
        tensor.real
    }
    fn imag(tensor: ComplexTensor<SplitBackend<B>>) -> super::FloatTensor<SplitBackend<B>> {
        tensor.imag
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
    
    async fn complex_into_real_data(tensor: ComplexTensor<SplitBackend<B>>, device: &ComplexDevice<SplitBackend<B>>) -> Result<TensorData, burn_tensor::backend::ExecutionError> {
      B::float_into_data(tensor.real).await   
    }
    
    async fn complex_into_imag_data(tensor: ComplexTensor<SplitBackend<B>>, device: &ComplexDevice<SplitBackend<B>>) -> Result<TensorData, burn_tensor::backend::ExecutionError> {
        B::float_into_data(tensor.imag).await
    }
    
    async fn complex_into_interleaved_data(tensor: ComplexTensor<SplitBackend<B>>, device: &ComplexDevice<SplitBackend<B>>) -> Result<TensorData, burn_tensor::backend::ExecutionError> {
        
        
        
        // let interleaved_shape = {
        //     let mut dims = tensor.real.shape().dims().to_vec();
        //     *dims.last_mut().unwrap() *= 2;
        //     burn_tensor::Shape::from(dims);
        // };
        todo!()
    }
    
    async fn complex_into_split_data(tensor: ComplexTensor<SplitBackend<B>>, device: &ComplexDevice<SplitBackend<B>>) -> Result<(TensorData, TensorData), burn_tensor::backend::ExecutionError> {
        let real_data = B::float_into_data(tensor.real).await?;
        let imag_data = B::float_into_data(tensor.imag).await?;
        Ok((real_data, imag_data))
    }
}

