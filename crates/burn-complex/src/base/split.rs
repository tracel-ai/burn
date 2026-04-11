use crate::{
    base::{
        ComplexDevice, ComplexTensor, ComplexTensorBackend, ComplexTensorOps, Layout, SplitLayout,
        element::Complex,
    },
    utils::real_to_complex_dtype,
};
use burn_std::{Bytes, DType};
use burn_tensor::{
    ElementComparison, Float, TensorData, TensorKind, TensorMetadata,
    backend::{Backend, DeviceOps},
    get_device_settings,
    ops::FloatTensorOps,
};
use bytemuck::Pod;

impl<T: TensorMetadata + 'static> Layout for SplitLayout<T> {
    type ComplexTensorPrimitive = SplitComplexTensor<T>;
}

#[derive(Debug, Clone)]
pub struct SplitComplexTensor<P: TensorMetadata> {
    pub real: P,
    pub imag: P,
}

impl<T: TensorMetadata + 'static> TensorMetadata for SplitComplexTensor<T> {
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

    fn complex_from_real_data(data: TensorData, device: &B::Device) -> ComplexTensor<Self> {
        // ComplexTensor<Self> = Complex<SplitComplexTensor<B::FloatTensorPrimitive>>
        // i.e. Complex { re: SplitComplexTensor { real, imag } }
        let real = B::float_from_data(data, device);
        // https://github.com/rust-lang/rust/issues/54628
        let imag = B::float_from_data(
            TensorData::from_bytes_vec(
                vec![0u8; real.shape().num_elements() * real.dtype().size()],
                real.shape().clone(),
                real.dtype(),
            ),
            device,
        );
        SplitComplexTensor { real, imag }
    }

    fn complex_from_imag_data(
        data: TensorData,
        device: &<Self::InnerBackend as Backend>::Device,
    ) -> ComplexTensor<Self> {
        let imag = B::float_from_data(data, device);
        // https://github.com/rust-lang/rust/issues/54628
        let real = B::float_from_data(
            TensorData::from_bytes_vec(
                vec![0u8; imag.shape().num_elements() * imag.dtype().size()],
                imag.shape().clone(),
                imag.dtype(),
            ),
            device,
        );
        SplitComplexTensor { real, imag }
    }
    // Should these be a result
    fn complex_from_interleaved_data(
        data: TensorData,
        device: &<Self::InnerBackend as Backend>::Device,
    ) -> ComplexTensor<Self> {
        let mut real_bytes: Vec<u8> = Vec::with_capacity(data.bytes.len() / 2);
        let mut imag_bytes: Vec<u8> = Vec::with_capacity(data.bytes.len() / 2);

        let element_size = data.dtype.size();
        let complex_pair_size = 2 * element_size;

        // Iterate through the bytes in chunks of (Real + Imag)
        for chunk in data.bytes.chunks_exact(complex_pair_size) {
            let (real_part, imag_part) = chunk.split_at(element_size);

            real_bytes.extend_from_slice(real_part);
            imag_bytes.extend_from_slice(imag_part);
        }
        SplitComplexTensor {
            real: B::float_from_data(
                TensorData::from_bytes_vec(real_bytes, data.shape.clone(), data.dtype),
                device,
            ),
            imag: B::float_from_data(
                TensorData::from_bytes_vec(imag_bytes, data.shape, data.dtype),
                device,
            ),
        }
    }

    fn complex_from_split_data(
        real_data: TensorData,
        imag_data: TensorData,
        device: &<Self::InnerBackend as Backend>::Device,
    ) -> ComplexTensor<Self> {
        let real = B::float_from_data(real_data, device);
        let imag = B::float_from_data(imag_data, device);
        assert_eq!(
            real.shape(),
            imag.shape(),
            "Real and imaginary parts must have the same shape"
        );
        assert_eq!(
            real.dtype(),
            imag.dtype(),
            "Real and imaginary parts must have the same dtype"
        );
        SplitComplexTensor { real, imag }
    }
}
type FlOps<B> = <SplitBackend<B> as ComplexTensorBackend>::InnerBackend;
impl<B> ComplexTensorOps<SplitBackend<B>> for SplitBackend<B>
where
    B: Backend,
    <B as Backend>::FloatElem: ElementComparison + Pod,
{
    fn to_complex(tensor: super::FloatTensor<SplitBackend<B>>) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            imag: B::float_zeros(
                tensor.shape().clone(),
                &<Self as ComplexTensorBackend>::InnerBackend::float_device(&tensor),
                tensor.dtype().into(),
            ),
            real: tensor,
        }
    }

    fn real(tensor: ComplexTensor<SplitBackend<B>>) -> super::FloatTensor<SplitBackend<B>> {
        tensor.real
    }
    fn imag(tensor: ComplexTensor<SplitBackend<B>>) -> super::FloatTensor<SplitBackend<B>> {
        tensor.imag
    }

    fn complex_not_equal_elem(
        lhs: ComplexTensor<SplitBackend<B>>,
        rhs: <SplitBackend<B> as ComplexTensorBackend>::ComplexScalar,
    ) -> super::BoolTensor<SplitBackend<B>> {
        todo!()
    }

    async fn complex_into_real_data(
        tensor: ComplexTensor<SplitBackend<B>>,
    ) -> Result<TensorData, burn_tensor::backend::ExecutionError> {
        B::float_into_data(tensor.real).await
    }

    async fn complex_into_imag_data(
        tensor: ComplexTensor<SplitBackend<B>>,
    ) -> Result<TensorData, burn_tensor::backend::ExecutionError> {
        B::float_into_data(tensor.imag).await
    }

    async fn complex_into_interleaved_data(
        tensor: ComplexTensor<SplitBackend<B>>,
    ) -> Result<TensorData, burn_tensor::backend::ExecutionError> {
        let real_data = B::float_into_data(tensor.real).await?;
        let imag_data = B::float_into_data(tensor.imag).await?;
        let element_size = real_data.dtype.size();
        let mut interleaved_bytes = Vec::with_capacity(real_data.bytes.len() * 2);
        for (real_chunk, imag_chunk) in real_data
            .bytes
            .chunks_exact(element_size)
            .zip(imag_data.bytes.chunks_exact(element_size))
        {
            interleaved_bytes.extend_from_slice(real_chunk);
            interleaved_bytes.extend_from_slice(imag_chunk);
        }
        Ok(TensorData::from_bytes_vec(
            interleaved_bytes,
            real_data.shape,
            real_to_complex_dtype(real_data.dtype),
        ))
    }

    async fn complex_into_split_data(
        tensor: ComplexTensor<SplitBackend<B>>,
    ) -> Result<(TensorData, TensorData), burn_tensor::backend::ExecutionError> {
        let real_data = B::float_into_data(tensor.real).await?;
        let imag_data = B::float_into_data(tensor.imag).await?;
        Ok((real_data, imag_data))
    }

    fn complex_device(tensor: &ComplexTensor<SplitBackend<B>>) -> ComplexDevice<SplitBackend<B>> {
        <<SplitBackend<B> as ComplexTensorBackend>::InnerBackend as FloatTensorOps<
            <SplitBackend<B> as ComplexTensorBackend>::InnerBackend,
        >>::float_device(&tensor.real)
    }

    fn complex_add(
        lhs: ComplexTensor<SplitBackend<B>>,
        rhs: ComplexTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: FlOps::<B>::float_add(lhs.real, rhs.real),
            imag: FlOps::<B>::float_add(lhs.imag, rhs.imag),
        }
    }

    fn complex_sub(
        lhs: ComplexTensor<SplitBackend<B>>,
        rhs: ComplexTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: FlOps::<B>::float_sub(lhs.real, rhs.real),
            imag: FlOps::<B>::float_sub(lhs.imag, rhs.imag),
        }
    }

    fn complex_mul(
        lhs: ComplexTensor<SplitBackend<B>>,
        rhs: ComplexTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor {
            real: FlOps::<B>::float_sub(
                FlOps::<B>::float_mul(lhs.real.clone(), rhs.real.clone()),
                FlOps::<B>::float_mul(lhs.imag.clone(), rhs.imag.clone()),
            ),
            imag: FlOps::<B>::float_add(
                FlOps::<B>::float_mul(lhs.real, rhs.imag),
                FlOps::<B>::float_mul(rhs.real, lhs.imag),
            ),
        }
    }

    fn complex_div(
        lhs: ComplexTensor<SplitBackend<B>>,
        rhs: ComplexTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        // (a + i b) / (c + i d) == [(a + i b) * (c - i d)] / (c*c + d*d)
        //   == [(a*c + b*d) / (c*c + d*d)] + i [(b*c - a*d) / (c*c + d*d)]

        let norm_sqr = SplitBackend::<B>::complex_squared_norm(rhs.clone());

        SplitComplexTensor {
        real: FlOps::<B>::float_div(
            FlOps::<B>::float_add(
                FlOps::<B>::float_mul(lhs.real.clone(), rhs.real.clone()),
                FlOps::<B>::float_mul(lhs.imag.clone(), rhs.imag.clone()),
            ),
            norm_sqr.clone(),
        ),
        imag: FlOps::<B>::float_div(
            FlOps::<B>::float_sub(
                FlOps::<B>::float_mul(lhs.imag.clone(), rhs.real.clone()),
                FlOps::<B>::float_mul(lhs.real.clone(), rhs.imag.clone()),
            ),
            norm_sqr.clone(),
        ),

        }
    }
    fn complex_abs(tensor: ComplexTensor<SplitBackend<B>>) -> super::FloatTensor<SplitBackend<B>> {
        todo!()
    }

    fn complex_from_parts(
        real: super::FloatTensor<SplitBackend<B>>,
        imag: super::FloatTensor<SplitBackend<B>>,
    ) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor { real, imag }
    }
    
    fn complex_exp(tensor: ComplexTensor<SplitBackend<B>>) -> ComplexTensor<SplitBackend<B>> {
        todo!()
    }

    fn complex_log(tensor: ComplexTensor<SplitBackend<B>>) -> ComplexTensor<SplitBackend<B>> {
        todo!()
    }
    
    
    
    fn complex_squared_norm(tensor: ComplexTensor<SplitBackend<B>>) -> super::FloatTensor<SplitBackend<B>> {
        let real_sq = FlOps::<B>::float_mul(tensor.real.clone(), tensor.real.clone());
        let imag_sq = FlOps::<B>::float_mul(tensor.imag.clone(), tensor.imag.clone());
        FlOps::<B>::float_add(real_sq, imag_sq)

    }
    
    fn complex_from_polar(magnitude: super::FloatTensor<SplitBackend<B>>, phase: super::FloatTensor<SplitBackend<B>>) -> ComplexTensor<SplitBackend<B>> {
        SplitComplexTensor{
            real: FlOps::<B>::float_mul(magnitude.clone(), FlOps::<B>::float_cos(phase.clone())),
            imag: FlOps::<B>::float_mul(magnitude, FlOps::<B>::float_sin(phase)),
            
        }
    }
}
