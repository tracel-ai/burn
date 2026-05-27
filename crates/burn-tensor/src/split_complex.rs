use std::marker::PhantomData;

use alloc::vec::Vec;
use burn_backend::Layout;
use burn_backend::TypedDevice;
use burn_backend::ops::ComplexTensorOps;
use burn_backend::ops::FloatTensorOps;
use burn_backend::try_read_sync;
use burn_backend::{
    Backend, BackendTypes, DTypeUsageSet, ExecutionError, TensorMetadata
};
use burn_backend::{ComplexTensor, ComplexTensorBackend, DefaultComplexOps, tensor::Device as BackendDevice};
use burn_dispatch::Dispatch;
use burn_std::AsIndex;
use burn_std::DType;
use burn_std::Distribution;
use burn_std::Element;
use burn_std::ElementConversion;
use burn_std::SliceArg;
use burn_std::complex_utils;
use burn_std::complex_utils::real_to_complex_dtype;
use burn_std::{Complex, ComplexElement, ElementComparison, Scalar, TensorData, cast::ToElement};
use burn_std::{ComplexDType, FloatDType, IndexingUpdateOp, Shape, SplitTensorData};
use bytemuck::Pod;
use crate::bridge::BridgeKind;
use crate::atan2_impl;

#[derive(Debug, Clone)]
pub struct SplitComplexTensor<const D: usize, K=Float> 
where K: Basic {
    _kind: PhantomData<K>,
    pub(crate) real: BridgeTensor,
    pub(crate) imag: BridgeTensor,
}

/// Indicates that the underlying implementation has separate real and imaginary tensors.
pub struct SplitLayout<B> {
    _marker: core::marker::PhantomData<B>,
}

impl<B> Layout for SplitLayout<B> {}

impl<const D: usize, K> SplitComplexTensor<D, K>
where K: Basic,
{
    pub fn new(real: BridgeTensor, imag: BridgeTensor) -> Self {
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
        Self {
            _kind: core::marker::PhantomData,
            real,
            imag,
        }
    }

    pub fn inner_dtype(&self) -> burn_std::DType {
        self.real.dtype()
    }

    pub fn from_parts_data(real: TensorData, imag: TensorData, device: &Device) -> Self {
        let dtype = real.dtype;
        let shape = &real.shape;
        assert_eq!(
            shape,
            &imag.shape,
            "Real and imaginary parts must have the same shape"
        );
        assert_eq!(
            dtype,
            imag.dtype,
            "Real and imaginary parts must have the same dtype"
        );
        let real_tensor = K::from_data(real, device, dtype);
        let imag_tensor = K::from_data(imag, device, dtype);
        
        Self::new(real_tensor, imag_tensor)
    }

    

    pub fn from_real_data(data: TensorData, device: &Device) -> Self {
        let shape = data.shape.clone();
        let dtype = data.dtype;
        Self {
            _kind: core::marker::PhantomData,
            real: K::from_data(data, device, dtype),
            imag: K::zeros(shape, device, dtype.into()),
        }
    }

    pub fn from_imag_data(data: TensorData, device: &Device) -> Self {
        let shape = data.shape.clone();
        let dtype = data.dtype;
        Self {
            _kind: core::marker::PhantomData,
            real: K::zeros(shape, device, dtype.into()),
            imag: K::from_data(data, device, dtype),
        }
    }

    pub fn from_split_data(data: SplitTensorData, device: &Device) -> Self {
        let SplitTensorData {
            real_bytes: real,
            imag_bytes: imag,
            shape,
            dtype,
        } = data;

        Self {
            _kind: core::marker::PhantomData,
            real: K::from_data(
                TensorData::from_bytes(real, shape.clone(), dtype),
                device,
                dtype,
            ),
            imag: K::from_data(TensorData::from_bytes(imag, shape, dtype), device, dtype),
        }
    }
    pub fn real(self) -> BridgeTensor {
        self.real
    }
    pub fn imag(self) -> BridgeTensor {
        self.imag
    }
    pub fn real_ref(&self) -> &BridgeTensor {
        &self.real
    }
    pub fn imag_ref(&self) -> &BridgeTensor {
        &self.imag
    }
    pub fn into_parts(self) -> (Tensor<D, K>, Tensor<D, K>) {
        (Tensor::new(self.real), Tensor::new(self.imag))
    }
}

impl<const D: usize, K> TensorMetadata for SplitComplexTensor<D,K>
where K: Basic,
{
    fn shape(&self) -> burn_std::Shape {
        self.real.shape()
    }

    fn rank(&self) -> usize {
        self.shape().num_dims()
    }

    fn dtype(&self) -> burn_std::DType {
        burn_std::complex_utils::real_to_complex_dtype(self.inner_dtype())
    }
    
}

impl<const D: usize, K> SplitComplexTensor<D, K>
where K: Basic + Numeric,
{
    pub fn squared_norm(self) -> Tensor<D, K> {
        let real_sq = K::mul(self.real.clone(), self.real);
        let imag_sq = K::mul(self.imag.clone(), self.imag);
        Tensor::new(K::add(real_sq, imag_sq))
    }
}

#[derive(Debug, Clone)]
/// A newtype that wraps a real backend B and exposes a split-layout complex backend.
pub struct SplitBackend<B: Backend, const D: usize>(core::marker::PhantomData<B>);

impl<B: Backend, const D: usize> BackendTypes for SplitBackend<B, D> {
    type Device = B::Device;

    type FloatTensorPrimitive = B::FloatTensorPrimitive;    

    type IntTensorPrimitive = B::IntTensorPrimitive;


    type BoolTensorPrimitive = B::BoolTensorPrimitive;

    type QuantizedTensorPrimitive = B::QuantizedTensorPrimitive;

    fn dtype_usage(device: &Self::Device, dtype: burn_std::DType) -> DTypeUsageSet {
        B::dtype_usage(device, dtype)
    }

    fn device_count(type_id: u16) -> usize {
        B::device_count(type_id)
    }

    type ComplexTensorPrimitive = SplitComplexTensor<D,Float>;
}

impl<B, const D: usize> TypedDevice<Self> for SplitBackend<B, D>
where
    B: Backend,
{
    fn complex_device(tensor: &ComplexTensor<Self>) -> <Self as BackendTypes>::Device {
        //Need to figure out how to resolve the inner backends Device type
        
        let (kind, tensor) = tensor.real.as_parts();
        match kind {
            BridgeKind::Float => Device::new(Dispatch::float_device(tensor)),
            //BridgeKind::QFloat => Device::new(Dispatch::q_device(tensor)),
            _ => panic!("Should be Float primitive kind"),
        };
        todo!()
        
    }
}

impl<B, const D: usize> ComplexTensorBackend for SplitBackend<B, D>
where
    B: Backend,
    B: BackendTypes,
{
    type InnerBackend = B;
    type Layout = SplitLayout<B>;

    fn complex_from_real_data(data: TensorData, device: &B::Device) -> ComplexTensor<Self> {
        // ComplexTensor<Self> = Complex<SplitComplexTensor<B::FloatTensorPrimitive>>
        // i.e. Complex { re: SplitComplexTensor { real, imag } }
        Self::ComplexTensorPrimitive::from_real_data(data, device.into())
    }

    fn complex_from_imag_data(data: TensorData, device: &B::Device) -> ComplexTensor<Self> {
        Self::ComplexTensorPrimitive::from_imag_data(data, device.into())
    }
    // Should these be a result
    fn complex_from_interleaved_data(data: TensorData, device: &B::Device) -> ComplexTensor<Self> {
        Self::ComplexTensorPrimitive::from_split_data(
            burn_std::complex_utils::split_from_interleaved_data(data),
            device.into(),
        )
    }

    fn complex_from_parts_data(
        real_data: TensorData,
        imag_data: TensorData,
        device: &B::Device,
    ) -> ComplexTensor<Self> {
        ComplexTensor::<Self>::from_parts(real_data, imag_data, device.into())
    }
}

impl<B, const D: usize,F> ComplexTensorOps<SplitBackend<B, D>> for SplitBackend<B, D>
where
    B: Backend,
    B: BackendTypes<FloatElem = F, ComplexScalar = Complex<F>>,
{
    fn to_complex(tensor: B::FloatTensorPrimitive) -> ComplexTensor<SplitBackend<B, D>> {
        let shape = tensor.shape().clone();
        let dtype = tensor.dtype().into();
        let device = &<Self as ComplexTensorBackend>::InnerBackend::float_device(&tensor);
        ComplexTensor::<SplitBackend<B, D>>::new(
            tensor.into(),
            Dispatch::float_zeros(shape, device.into(), dtype).into(),
        )
    }

    fn complex_real(tensor: ComplexTensor<SplitBackend<B, D>>) -> B::FloatTensorPrimitive {
        tensor.real.into()
    }
    fn complex_imag(tensor: ComplexTensor<SplitBackend<B, D>>) -> B::FloatTensorPrimitive {
        tensor.imag.into()
    }

    fn complex_not_equal_elem(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        lhs.not_equal_elem(rhs).into()
    }

    fn complex_equal_elem(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        lhs.equal_elem(rhs).into()
    }

    fn complex_equal(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        lhs.equal(rhs).into()
    }

    fn complex_not_equal(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        lhs.not_equal(rhs).into()
    }

    async fn complex_into_real_data(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> Result<TensorData, ExecutionError> {
        B::float_into_data(tensor.real.into()).await
            
            
            
    }

    async fn complex_into_imag_data(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> Result<TensorData, ExecutionError> {
        B::float_into_data(tensor.imag.into()).await
    }

    async fn complex_into_interleaved_data(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> Result<TensorData, ExecutionError> {
        let real_data = B::float_into_data(tensor.real.into()).await?;
        let imag_data = B::float_into_data(tensor.imag.into()).await?;
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
            burn_std::complex_utils::real_to_complex_dtype(real_data.dtype),
        ))
    }

    async fn complex_into_split_data(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> Result<SplitTensorData, ExecutionError> {
        let (real, imag) = (
            tensor.real,
            tensor.imag,
        );
        let shape = real.shape().clone();
        let dtype = real.dtype();
        Ok(SplitTensorData {
            real_bytes: K::into_data(real).await?.bytes,
            imag_bytes: K::into_data(imag).await?.bytes,
            shape,
            dtype,
        })
    }

    fn complex_add(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        lhs.add(rhs)
    }

    fn complex_sub(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        lhs.sub(rhs)
    }

    fn complex_mul(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        lhs.mul(rhs)
    }

    fn complex_div(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        lhs.div(rhs)
    }
    fn abs(tensor: ComplexTensor<SplitBackend<B, D>>) -> B::FloatTensorPrimitive {
        //todo! https://github.com/tracel-ai/burn/issues/4836
        Float::sqrt(tensor.squared_norm().primitive).into()
    }

    fn complex_from_parts(real: TensorData, imag: TensorData) -> ComplexTensor<SplitBackend<B, D>> {
        ComplexTensor::<SplitBackend<B, D>>::from_parts_data(real, imag, &B::default_device())
    }

    fn complex_exp(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.exp()
    }

    fn complex_log(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.log()
    }

    fn complex_squared_norm(tensor: ComplexTensor<SplitBackend<B, D>>) -> B::FloatTensorPrimitive {
        tensor.squared_norm().into()
    }

    fn complex_from_polar(
        magnitude: B::FloatTensorPrimitive,
        phase: B::FloatTensorPrimitive,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        ComplexTensor::<SplitBackend<B, D>>::from_polar(
            magnitude,
            phase,
        )
    }

    fn complex_gather(
        dim: usize,
        tensor: ComplexTensor<SplitBackend<B, D>>,
        indices: B::IntTensorPrimitive,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.gather(dim, indices.into())
    }

    fn complex_scatter_add(
        dim: usize,
        tensor: ComplexTensor<SplitBackend<B, D>>,
        indices: B::IntTensorPrimitive,
        values: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.scatter(dim, indices.into(), values, IndexingUpdateOp::Add)
    }

    fn complex_random(
        shape: burn_std::Shape,
        distribution: burn_std::Distribution,
        device: &BackendDevice<B>,
        dtype: ComplexDType,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        ComplexTensor::<Self>::random(shape, distribution, TensorCreationOptions::new(device.into()))
    }

    fn complex_to_device(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        device: &BackendDevice<B>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.to_device(device.into())
    }

    fn complex_reshape(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        shape: burn_std::Shape,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            K::reshape(tensor.real,shape.clone()),
            K::reshape(tensor.imag,shape),
        )
    }

    fn complex_transpose(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.transpose()
    }

    fn complex_neg(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.neg()
    }

    fn conj(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        // conj(a + bi) = a - bi
        SplitComplexTensor::conj(tensor)
    }

    fn complex_arg(tensor: ComplexTensor<SplitBackend<B, D>>) -> B::FloatTensorPrimitive {
        // arg(a + bi) = atan2(b, a)
        tensor.phase().into()
    }

    fn complex_powc(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // z^w = exp(w * ln(z))
        let log_lhs = lhs.log();
        let product = SplitBackend::<B, D>::complex_mul(rhs, log_lhs);
        product.exp()
    }

    fn complex_sqrt(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.sqrt()
    }

    fn complex_sin(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.sin()
    }

    fn complex_cos(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.cos()
    }

    fn complex_tan(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.tan()
    }

    fn complex_acos(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.acos()
    }

    fn complex_acosh(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.acosh()
    }

    fn complex_asin(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.asin()
    }

    fn complex_asinh(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.asinh()
    }

    fn complex_atan(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.atan()
    }

    fn complex_atanh(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.atanh()
    }

    fn complex_slice(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        slices: &[burn_std::Slice],
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.slice(slices)
    }

    fn complex_slice_assign(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        ranges: &[burn_std::Slice],
        value: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.slice_assign(ranges, value)
    }

    fn complex_swap_dims(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim1: usize,
        dim2: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.swap_dims(dim1, dim2)
    }

    fn complex_repeat_dim(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
        times: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.repeat_dim(dim, times)
    }

    fn complex_cat(
        tensors: Vec<ComplexTensor<SplitBackend<B, D>>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::cat(tensors, dim)
    }

    fn complex_any(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        tensor.any().into()
    }

    fn complex_any_dim(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        tensor.any_dim(dim).into()
    }

    fn complex_all(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        tensor.all().into()
    }

    fn complex_all_dim(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        tensor.all_dim(dim).into()
    }

    fn complex_permute(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        axes: &[usize],
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.permute(axes.into())
    }

    fn complex_expand(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        shape: burn_std::Shape,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(K::expand(tensor.real,shape.clone()), K::expand(tensor.imag,shape))
    }

    fn complex_flip(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        axes: &[usize],
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(K::flip(tensor.real,axes), K::flip(tensor.imag,axes))
    }

    fn complex_unfold(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            K::unfold(tensor.real,dim, size, step),
            K::unfold(tensor.imag,dim, size, step),
        )
    }

    fn complex_select(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
        indices: B::IntTensorPrimitive,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.select(dim, indices.into())
    }

    fn complex_sum(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.sum()
    }

    fn complex_sum_dim(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.sum_dim(dim)
    }

    fn complex_prod(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.prod()
    }

    fn complex_prod_dim(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.prod_dim(dim)
    }

    fn complex_mean(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.mean()
    }

    fn complex_mean_dim(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.mean_dim(dim)
    }

    fn complex_remainder(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        lhs.remainder(rhs)
    }

    fn complex_remainder_scalar(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: Scalar,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        lhs.remainder_scalar(rhs)
    }

    fn complex_mask_where(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        mask: B::BoolTensorPrimitive,
        source: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.mask_where(mask.into(), source)
    }

    fn complex_mask_fill(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        mask: B::BoolTensorPrimitive,
        value: Scalar,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.mask_fill(mask.into(), value)
    }

    fn complex_sign(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.sign()
    }

    fn complex_matmul(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        lhs.matmul(rhs)
    }

    fn complex_cumsum(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.cumsum(dim)
    }

    fn complex_cumprod(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.cumprod(dim)
    }

    fn complex_select_add(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
        indices: B::IntTensorPrimitive,
        values: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        tensor.select_add(dim, indices, values)
    }

    fn complex_powc_scalar(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: Scalar,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // z^c = exp(c * ln(z)) where c = a + bi is a scalar
        // (a + bi) * (u + vi) = (au - bv) + (av + bu)i
        assert_eq!(rhs, burn_std::Scalar::Complex(_), "rhs must be a complex scalar");
        let rhs = rhs.elem::<Complex<f64>>();
        let a = burn_std::Scalar::Float(rhs.real);
        let b = burn_std::Scalar::Float(rhs.imag);
        let ln_z = lhs.log();
        SplitComplexTensor::new(
            ln_z.real * a - ln_z.imag * b,
            ln_z.real * b + ln_z.imag * a,
        ).exp()
    }

    fn complex_powf(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: B::FloatTensorPrimitive,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        lhs.powf(rhs)       
    }

    fn complex_powf_scalar(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: Scalar,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // z^w = exp(w * ln(z)) where w is a real scalar
        lhs.powf_scalar(rhs.elem::<f64>())
    }

    fn complex_scatter_nd(
        _tensor: ComplexTensor<SplitBackend<B, D>>,
        _indices: B::IntTensorPrimitive,
        _value: ComplexTensor<SplitBackend<B, D>>,
        _reduction: IndexingUpdateOp,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // can't implement scatter_nd via trait as SplitTensor isn't generic over rank
        // In order to be api compatible with Tensor<B,D,K> anything that requires more than 1 generic const rank
        // needs to be implemented directly on the SplitTensor type.
        unreachable!("inlined into scatter_nd for SplitTensor")
    }

    async fn complex_into_data(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> Result<SplitTensorData, ExecutionError> {
        <SplitBackend<B, D> as ComplexTensorBackend>::Layout::complex_into_data(tensor).await
    }
}

impl<B, F> DefaultComplexOps<B> for SplitLayout<B>
where
    B: ComplexTensorBackend<Layout = SplitLayout<B>>,
    B: BackendTypes<FloatTensorPrimitive = F>,
    F: burn_backend::TensorMetadata,
{
    type OutTensorData = SplitTensorData;
    fn zeros(shape: Shape, device: &BackendDevice<B>, _dtype: ComplexDType) -> ComplexTensor<B> {
        // let real = B::InnerBackend::float_from_data(
        //     TensorData::zeros::<<B::InnerBackend as BackendTypes>::FloatElem, _>(&shape),
        //     device,
        // );
        // let imag = B::InnerBackend::float_from_data(
        //     TensorData::zeros::<<B::InnerBackend as BackendTypes>::FloatElem, _>(shape),
        //     device,
        // );
        // // ComplexTensor<B> = Complex<T> via SplitLayout
        // <B as BackendTypes>::ComplexTensorPrimitive::new(real, imag)
        unimplemented!("placeholder")
    }

    fn ones(shape: Shape, device: &BackendDevice<B>, _dtype: ComplexDType) -> ComplexTensor<B> {
        // let real = TensorData::ones::<<B::InnerBackend as BackendTypes>::FloatElem, _>(&shape);
        // let imag = TensorData::zeros::<<B::InnerBackend as BackendTypes>::FloatElem, _>(shape);
        // ComplexTensor::<B>::from_parts_data(
        //     TensorData::ones(&shape),
        //     TensorData::zeros(&shape),
        // )
            unimplemented!("placeholder")
        
    }

    fn full(
        shape: Shape,
        fill_value: Scalar,
        device: &BackendDevice<B>,
    ) -> ComplexTensor<B> {
        // match fill_value {
        //     Scalar::Complex(c) => {
        //         let real = TensorData::full(&shape, c.real());
        //         let imag = TensorData::full(&shape, c.imag());
        //         ComplexTensor::<B>::from_parts_data(real, imag)
        //     }
        //     x => {
        //         ComplexTensor::<B>::from_real_data(
        //         TensorData::full(&shape, x.into()),
        //         )
        //     }
        // }
        unimplemented!("placeholder")
    }

    async fn complex_into_data(
        tensor: ComplexTensor<B>,
    ) -> Result<Self::OutTensorData, ExecutionError> {
        B::complex_into_split_data(tensor).await
    }
}

impl<const D: usize, K> SplitComplexTensor<D,K>
where
    K: Basic,
{
    /// Create an empty complex tensor of the given shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `options` - Options to control creation, including device and dtype.
    pub fn empty<S: Into<Shape>>(shape: S, options: impl Into<TensorCreationOptions>) -> Self {
        let opt = options.into();
        let shape = shape.into();
        //check!(TensorCheck::creation_ops::<D>("Empty", &shape));
        Self::zeros(shape, &opt.device.into())
    }

    /// Returns a complex tensor containing the elements selected from the given slices.
    ///
    /// # Arguments
    ///
    /// * `slices` - The slices to select from.
    ///
    /// # Panics
    ///
    /// If the number of slices exceeds the tensor's dimensions.
    pub fn slice<S>(self, slices: S) -> Self
    where
        S: SliceArg,
    {
        let shape = self.shape();
        let slices = slices.into_slices(&shape);

        // Validate slices
        //check!(TensorCheck::slice::<D>(&shape, &slices));

        // Calculate output shape and check for empty slices
        let mut output_dims = shape.clone();
        for (dim, slice) in slices.iter().enumerate() {
            output_dims[dim] = slice.output_size(shape[dim]);
        }

        // Return empty tensor if any dimension is 0 (empty slice)
        if output_dims.contains(&0) {
            return Self::empty(output_dims, &self.device().into());
        }
        SplitComplexTensor::new(K::slice(self.real, &slices), K::slice(self.imag, &slices))
    }

    /// Create a complex tensor of the given shape where each element is zero.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `options` - Options to control creation, including device and dtype.
    pub fn zeros<S: Into<Shape>>(shape: S, options: impl Into<TensorCreationOptions>) -> Self {
        let options = options.into();
        let shape = shape.into();
        let device = &options.device;
        let dtype = options.dtype();
        match dtype {
            DType::Complex32 | DType::Complex64 => {
                let dtype = burn_std::complex_utils::complex_to_real_dtype(dtype);
                Self::new(
                    K::zeros(shape.clone(), device.into(), dtype.into()),
                    K::zeros(shape, device.into(), dtype.into()),
                )
            }
            _ => panic!("Unsupported complex dtype"),
        }
    }

    /// Create a complex tensor of the given shape where each element is one.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `options` - Options to control creation, including device and dtype.
    pub fn ones<S: Into<Shape>>(shape: S, options: impl Into<TensorCreationOptions>) -> Self {
        let options = options.into();
        let shape = shape.into();
        
        let dtype= options.dtype();
        match dtype {
            DType::Complex32 | DType::Complex64 => {
                let dtype = burn_std::complex_utils::complex_to_real_dtype(dtype);
                Self::ones(shape, options)
            }
            _ => panic!("Unsupported complex dtype"),
        }
    }
}
//BasicOps
impl<const D: usize, K> SplitComplexTensor<D,K>
where K: Basic,
{
    /// Select complex tensor elements along the given dimension corresponding to the given indices.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to select from.
    /// * `indices` - The indices of the elements to select.
    pub fn select(self, dim: usize, indices: Tensor<1, Int>) -> Self {
        // Uses your existing `select` name.
        SplitComplexTensor::new(
            K::select(self.real,dim, indices.primitive.clone()),
            K::select(self.imag,dim, indices.primitive),
        )
    }

    /// Returns the dimensions of the current tensor.
    ///
    /// # Example
    /// ```rust
    /// use burn_std::backend::Backend;
    /// use burn_std::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///   let device = Default::default();
    ///   let tensor = Tensor::<B, 3>::ones([2, 3, 4], &device);
    ///   let dims = tensor.dims(); // [2, 3, 4]
    ///   println!("{dims:?}");
    /// }
    /// ```
    pub fn dims(&self) -> [usize; D] {
        Self::shape(self).dims()
    }

    /// Returns the shape of the current tensor.
    ///
    /// # Example
    /// ```rust
    /// use burn_std::backend::Backend;
    /// use burn_std::Tensor;
    ///
    /// fn example<B: Backend>() {
    ///    let device = Default::default();
    ///    let tensor = Tensor::<B, 3>::ones([2, 3, 4], &device);
    ///    // Shape { dims: [2, 3, 4] }
    ///    let shape = tensor.shape();
    /// }
    /// ```
    pub fn shape(&self) -> Shape {
        self.real.shape()
    }

    /// Assign the selected complex tensor elements along the given dimension corresponding to
    /// the given indices from the value tensor to the original tensor.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension along which to select.
    /// * `indices` - The indices to select from the tensor.
    /// * `values` - The complex values to assign to the selected indices.
    /// * `update` - The operation used to update the existing values at the indexed positions.
    ///
    /// # Panics
    ///
    /// If `update` is not `IndexingUpdateOp::Add`. Other operations are currently not implemented.
    pub fn select_assign(
        self,
        dim: usize,
        indices: Tensor<1, Int>,
        values: Self,
        update: IndexingUpdateOp,
    ) -> Self {
        match update {
            IndexingUpdateOp::Add => {
                let indices = indices.primitive;
                SplitComplexTensor::new(
                    K::select_assign(self.real,dim, indices.clone(), values.real, update.clone()),
                    K::select_assign(self.imag,dim, indices, values.imag, update),
                )
            }
            _ => unimplemented!(),
        }
    }

    /// Reshape the complex tensor to have the given shape.
    ///
    /// The tensor has the same data and number of elements as the input.
    ///
    /// A `-1` in the shape is used to infer the remaining dimensions, e.g.: `[2, -1]`
    /// will reshape the tensor with [2, 3, 4] dimensions to [2, 12].
    ///
    /// # Arguments
    /// - `shape`: The new shape of the tensor.
    pub fn reshape<const D2: usize, S: ReshapeArgs<D2>>(
        self,
        shape: S,
    ) -> SplitComplexTensor<D2, K> {
        // Convert reshape args to shape
        let shape = shape.into_shape::<D2>(self.shape());
        let (real, imag) = (self.real, self.imag);
        SplitComplexTensor::new(K::reshape(real,shape.clone()), K::reshape(imag,shape))
    }

    /// Transpose the complex tensor.
    ///
    /// For a 2D tensor, this is the standard matrix transpose. For `D > 2`, the transpose is
    /// applied on the last two dimensions.
    pub fn transpose(self) -> Self {
        SplitComplexTensor::new(K::transpose(self.real), K::transpose(self.imag))
    }

    /// Swaps two dimensions of a complex tensor.
    ///
    /// # Arguments
    ///
    /// * `dim1` - The first dimension to swap.
    /// * `dim2` - The second dimension to swap.
    pub fn swap_dims(self, dim1: usize, dim2: usize) -> Self {
        SplitComplexTensor::new(
            K::swap_dims(self.real,dim1, dim2),
            K::swap_dims(self.imag,dim1, dim2),
        )
    }
    

    /// Returns the device of the current complex tensor.
    pub fn device(&self) -> Device {
        K::device(&self.real)
    }

    /// Move the complex tensor to the given device.
    pub fn to_device(self, device: &Device) -> Self {
        SplitComplexTensor::new(K::to_device(self.real,device), K::to_device(self.imag,device))
    }

    /// Converts the data of the current complex tensor asynchronously.
    ///
    /// Returns the data as interleaved real and imaginary values.
    ///
    /// # Note
    ///
    /// For better performance, prefer using a [Transaction](burn_std::Transaction) when reading multiple
    /// tensors at once. This may improve laziness, especially if executed on a different
    /// thread in native environments.
    pub async fn into_data_async(self) -> Result<TensorData, ExecutionError> {
        self.into_interleaved_data().await
    }

    /// Create a complex tensor from the given interleaved complex data on the given device.
    ///
    /// # Arguments
    ///
    /// * `data` - The interleaved complex data (alternating real and imaginary values).
    /// * `options` - Options to control creation, including device and dtype.
    pub fn from_data<T>(data: T, options: impl Into<TensorCreationOptions>) -> Self
    where
        T: Into<TensorData>,
    {
        let data = data.into();
        let opt = options.into();
        Self::from_split_data(burn_std::complex_utils::split_from_interleaved_data(data), &opt.device)
    }

    /// Repeat the complex tensor along the given dimension.
    ///
    /// # Arguments
    /// - `dim`: The dimension to repeat.
    /// - `times`: The number of times to repeat the tensor along the given dimension.
    pub fn repeat_dim(self, dim: usize, times: usize) -> Self {
        SplitComplexTensor::new(
            K::repeat_dim(self.real,dim, times),
            K::repeat_dim(self.imag,dim, times),
        )
    }
    /// Applies element-wise equal comparison.
    ///
    /// # Returns
    ///
    /// A boolean tensor that is `true` where the two complex elements are equal and `false` elsewhere.
    pub fn equal(self, rhs: Self) -> Tensor<D, Bool> {
        let real_cmp = K::equal(self.real, rhs.real);
        let imag_cmp = K::equal(self.imag, rhs.imag);
        bool_and_impl(real_cmp, imag_cmp)
    }

    /// Applies element-wise non-equality comparison.
    ///
    /// # Returns
    ///
    /// A boolean tensor that is `true` where the two complex elements are not equal and `false` elsewhere.
    pub fn not_equal(self, rhs: Self) -> Tensor<D, Bool> {
        let real_cmp = K::not_equal(self.real,
            rhs.real,
        );
        let imag_cmp = K::not_equal(self.imag,
            rhs.imag,
        );
        bool_or_impl(real_cmp, imag_cmp)
    }

    /// Concatenates all complex tensors into a new one along the given dimension.
    ///
    /// # Panics
    ///
    /// - If `dim` is higher than the rank.
    /// - If `tensors` is an empty vector.
    /// - If all tensors don't have the same shape (the dimension `dim` is ignored).
    pub fn cat(tensors: Vec<Self>, dim: usize) -> Self {
        let (reals, imags): (Vec<_>, Vec<_>) =
            tensors.into_iter().map(|t| (t.real, t.imag)).unzip();
        SplitComplexTensor::new(K::cat(reals, dim), K::cat(imags, dim))
    }

    /// Tests if any element in the complex tensor evaluates to non-zero (i.e., true).
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element, `true` if any element is non-zero, `false` otherwise.
    pub fn any(self) -> Tensor<1, Bool> {
        let real_any = K::any(self.real);
        let imag_any = K::any(self.imag);
        bool_or_impl(real_any, imag_any)
    }

    /// Tests if any element in the complex tensor evaluates to non-zero along a given dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input, except in the `dim` axis where
    /// the size is 1, containing `true` if any element along that dimension is non-zero.
    pub fn any_dim(self, dim: usize) -> Tensor<D, Bool> {
        let real_any = K::any_dim(self.real,dim);
        let imag_any = K::any_dim(self.imag,dim);
        bool_or_impl(real_any, imag_any)
    }

    /// Tests if all elements in the complex tensor evaluate to non-zero (i.e., true).
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element, `true` if all elements are non-zero, `false` otherwise.
    pub fn all(self) -> Tensor<1, Bool> {
        let real_nonzero = K::not_equal_elem(self
            .real, burn_std::Scalar::Float(0.0));
        let imag_nonzero = K::not_equal_elem(self
            .imag,burn_std::Scalar::Float(0.0));
        let elem_nonzero = bool_and_impl(real_nonzero, imag_nonzero);
        elem_nonzero.all()
    }

    /// Tests if all elements in the complex tensor evaluate to non-zero along a given dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to test.
    ///
    /// # Returns
    ///
    /// A boolean tensor with the same shape as the input, except in the `dim` axis where
    /// the size is 1, containing `true` if all elements along that dimension are non-zero.
    pub fn all_dim(self, dim: usize) -> Tensor<D, Bool> {
        let real_nonzero = K::not_equal_elem(self.real, burn_std::Scalar::Float(0.0));
        let imag_nonzero = K::not_equal_elem(self.imag, burn_std::Scalar::Float(0.0));
        Tensor::new(K::all_dim(bool_and_impl(real_nonzero, imag_nonzero), dim))
        
    }

    /// Permute the dimensions of the complex tensor.
    ///
    /// This is a no-op when the resolved `axes` match the current order.
    ///
    /// # Arguments
    ///
    /// * `axes` - The new order of the dimensions. The length of the axes must equal the
    ///   number of dimensions. The values must be unique and in the range of the number of
    ///   dimensions. Negative values are used as an offset from the end.
    ///
    /// # Returns
    ///
    /// The tensor with the dimensions permuted.
    pub fn permute<Dim>(self, axes: [Dim; D]) -> Self
    where
        Dim: AsIndex,
    {
        let mut no_op = true;
        let mut fixed_axes = [0; D];
        for (i, axis) in axes.into_iter().enumerate() {
            let dim = axis.expect_dim_index(D);
            no_op &= dim == i;
            fixed_axes[i] = dim;
        }

        if no_op {
            self
        } else {
            SplitComplexTensor::new(K::permute(self.real, &fixed_axes), K::permute(self.imag, &fixed_axes))
        }
    }

    // pub fn expand(self, shape: Shape) -> Self {
    //     self.expand(, shape)
    // }

    /// Broadcast the complex tensor to the given shape.
    ///
    /// Only singleton dimensions can be expanded to a larger size. Other dimensions must have
    /// the same size (which can be inferred with `-1`).
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape to broadcast the tensor to. Can contain -1 for dimensions that
    ///   should be inferred.
    ///
    /// # Panics
    ///
    /// If the tensor cannot be broadcasted to the given shape.
    pub fn expand<const D2: usize, S: BroadcastArgs<D, D2>>(
        self,
        shape: S,
    ) -> SplitComplexTensor<D2, K> {
        let shape = shape.into_shape(&self.shape());
        // check!(TensorCheck::expand::<D, D2>(
        //     "expand",
        //     &self.shape(),
        //     &shape,
        // ));
        let (real, imag) = (self.real, self.imag);
        SplitComplexTensor::<D2,K>::new(K::expand(real,shape.clone()), K::expand(imag,shape))
    }

    // pub fn flip(self, axes: &[usize]) -> Self {
    //     self.flip(, axes)
    // }

    /// Reverse the order of elements in the complex tensor along the given dimensions.
    ///
    /// # Arguments
    ///
    /// * `axes` - The dimensions to reverse. The values must be unique and in the range of the
    ///   number of dimensions. Negative values are used as an offset from the end.
    pub fn flip<const N: usize>(self, axes: [isize; N]) -> Self {
        // Convert the axes to usize and handle negative values without using vector
        let mut transformed_axes: [usize; N] = [0; N];
        for (i, &x) in axes.iter().enumerate() {
            transformed_axes[i] = if x < 0 {
                (D as isize + x) as usize
            } else {
                x as usize
            };
        }
        let (real, imag) = (self.real, self.imag);
        // Check if the axes are valid
        //check!(TensorCheck::flip(D, &transformed_axes));

        SplitComplexTensor::<D,K>::new(K::flip(real, &transformed_axes), K::flip(imag, &transformed_axes))
    }
    /// Returns a view of the complex tensor with an additional dimension of size `size`
    /// obtained by slicing the tensor along `dim` with step `step`.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to unfold.
    /// * `size` - The size of each unfolded window.
    /// * `step` - The step between each window.
    ///
    /// # Returns
    ///
    /// A tensor with shape `[..., windows, ..., size]` where the extra `size` dimension
    /// is appended at the end.
    pub fn unfold<const D2: usize, I: AsIndex>(
        self,
        dim: I,
        size: usize,
        step: usize,
    ) -> SplitComplexTensor<D2,K> {
        let dim = dim.expect_dim_index(D);
        let (real, imag) = (self.real, self.imag);
        SplitComplexTensor::new(K::unfold(real, dim, size, step), K::unfold(imag, dim, size, step))
    }

    /// Assigns values to a slice of the complex tensor and returns the updated tensor.
    ///
    /// # Arguments
    ///
    /// * `slices` - The slice specification indicating where to assign.
    /// * `values` - Tensor with complex values to assign (must match the selected slice dimensions).
    ///
    /// # Panics
    ///
    /// - If slices exceed tensor dimensions.
    /// - If values dimensions don't match the selected slice shape.
    pub fn slice_assign<S>(self, slices: S, values: Self) -> Self
    where
        S: SliceArg,
    {
        let shape = self.shape();
        let slices = slices.into_slices(&shape);

        // Check if any slice produces 0 elements (empty assignment).
        // Empty assignments are no-ops and would cause issues in backend implementations.
        let is_empty_assignment = slices
            .iter()
            .enumerate()
            .any(|(i, slice)| slice.output_size(shape[i]) == 0);

        if is_empty_assignment {
            return self;
        }

        let values_shape = values.shape();
        for (i, slice) in slices
            .iter()
            .enumerate()
            .take(slices.len().min(shape.num_dims()))
        {
            let range = slice.to_range(shape[i]);
            assert!(
                range.end <= shape[i],
                "slice_assign: range ({}..{}) exceeds tensor size {} at dim {}",
                range.start,
                range.end,
                shape[i],
                i,
            );
            let expected = range.end - range.start;
            assert_eq!(
                values_shape[i], expected,
                "slice_assign: values shape {} does not match slice length {} at dim {}",
                values_shape[i], expected, i,
            );
        }

        SplitComplexTensor::new(
            K::slice_assign(self.real, &slices, values.real),
            K::slice_assign(self.imag, &slices, values.imag),
        )
    }

    /// Update the complex tensor with the value tensor where the mask is true.
    ///
    /// This is similar to [`mask_fill`](Self::mask_fill), however the value is a tensor
    /// instead of a scalar.
    ///
    /// # Arguments
    ///
    /// * `mask` - A boolean tensor with the same shape as the input tensor.
    /// * `source` - The complex tensor to use for replacement where the mask is true.
    pub fn mask_where(self, mask: Tensor<D, Bool>, source: Self) -> Self {
        SplitComplexTensor::new(
            K::mask_where(self.real, mask.primitive.clone(), source.real),
            K::mask_where(self.imag, mask.primitive, source.imag),
        )
    }

    /// Update the complex tensor with the scalar value where the mask is true.
    ///
    /// This is similar to [`mask_where`](Self::mask_where), however the value is a scalar
    /// instead of a tensor.
    ///
    /// # Arguments
    ///
    /// * `mask` - A boolean tensor with the same shape as the input tensor.
    /// * `value` - The scalar value to assign where the mask is true.
    pub fn mask_fill<E: ElementConversion>(self, mask: Tensor<D, Bool>, value: E) -> Self {
        use burn_std::cast::ToElement;
        let value = value.elem::<Complex<f64>>();
        let mask = mask.primitive;
        SplitComplexTensor::new(
            K::mask_fill(self.real, mask.clone(), burn_std::Scalar::Float(value.real())),
            K::mask_fill(self.imag, mask, burn_std::Scalar::Float(value.imag())),
        )
    }

    /// Gather complex tensor elements corresponding to the given indices from the specified dimension.
    ///
    /// Example using a 3D tensor:
    ///
    /// `output[i, j, k] = input[indices[i, j, k], j, k]; // dim = 0`
    /// `output[i, j, k] = input[i, indices[i, j, k], k]; // dim = 1`
    /// `output[i, j, k] = input[i, j, indices[i, j, k]]; // dim = 2`
    ///
    /// # Warning
    ///
    /// Not all backends have runtime bound checks for the indices, so make sure they are valid.
    /// Otherwise, out of bounds indices could lead to unexpected results instead of panicking.
    pub fn gather(self, dim: usize, indices: Tensor<D, Int>) -> Self {
        // check!(TensorCheck::gather::<D>(
        //     dim,
        //     &self.shape(),
        //     &indices.shape()
        // ));
        let indices= indices.primitive;
        SplitComplexTensor::new(
            K::gather(dim, self.real.into(), indices.clone()).into(),
            K::gather(dim, self.imag.into(), indices).into(),
        )
    }

    /// Assign the gathered elements corresponding to the given indices along the specified dimension
    /// from the value tensor to the original complex tensor.
    ///
    /// # Arguments
    ///
    /// * `dim` - The axis along which to scatter elements.
    /// * `indices` - The indices of the elements to scatter.
    /// * `values` - The complex values to scatter into the tensor.
    /// * `update` - The operation used to update the existing values at the indexed positions.
    ///
    /// # Warning
    ///
    /// Not all backends have runtime bound checks for the indices, so make sure they are valid.
    /// Otherwise, out of bounds indices could lead to unexpected results instead of panicking.
    ///
    /// # Panics
    ///
    /// If `update` is not `IndexingUpdateOp::Add`. Other operations are currently not implemented.
    pub fn scatter(
        self,
        dim: usize,
        indices: Tensor<D, Int>,
        values: Self,
        update: burn_std::IndexingUpdateOp,
    ) -> Self {
        match update {
            IndexingUpdateOp::Add => {
                Self::new(
                    K::scatter(dim, self.real, indices.primitive.clone(), values.real, update),
                    K::scatter(dim, self.imag, indices.primitive, values.imag, update),
        )
            }
            _ => unimplemented!(),
        }
    }

    /// Applies element-wise equal comparison with a scalar and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The element to compare each complex element against.
    pub fn equal_elem<E: Element>(self, other: E) -> Tensor<D, Bool> {
        
        let rhs = other.elem::<Complex<f64>>();
        let (lhs_real, lhs_imag) = (self.real, self.imag);
        let rhs_real = rhs.real();
        let rhs_imag = rhs.imag();
        let real_cmp = K::equal_elem(
            lhs_real,
            burn_std::Scalar::Float(rhs_real),
        );
        let imag_cmp = K::equal_elem(
            lhs_imag,
            burn_std::Scalar::Float(rhs_imag),
        );
        bool_and_impl(real_cmp, imag_cmp)
    }

    /// Applies element-wise non-equality comparison with a scalar and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The element to compare each complex element against.
    pub fn not_equal_elem<E: Element>(self, other: E) -> Tensor<D, Bool> {
        let rhs = other.elem::<Complex<f64>>();
        let (lhs_real, lhs_imag) = (self.real, self.imag);
        let rhs_real = rhs.real();
        let rhs_imag = rhs.imag();
        let real_cmp = K::not_equal_elem(
            lhs_real,
            burn_std::Scalar::Float(rhs_real),
        );
        let imag_cmp = K::not_equal_elem(
            lhs_imag,
            burn_std::Scalar::Float(rhs_imag),
        );
        bool_or_impl(real_cmp, imag_cmp)
    }

    /// Create a complex tensor of the given shape where each element is equal to the provided value.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `fill_value` - The complex value to fill the tensor with.
    /// * `options` - Options to control creation, including device and dtype.
    pub fn full<S: Into<Shape>, E: ElementConversion>(
        shape: S,
        fill_value: E,
        options: impl Into<TensorCreationOptions>,
    ) -> Self {
        //TODO: figure out how to map dtype so that it doesn't just assume Complex<f64>
        let e = E::elem::<Complex<f64>>(fill_value);
        let shape = shape.into();
        SplitComplexTensor::from_parts_data(
            TensorData::full(&shape, e.real()),
            TensorData::full(&shape, e.imag()),
            &options.into().device,
        )
    }

    /// Multi-dimensional scatter: update the complex tensor at locations given by `indices`
    /// using the specified `update` operation.
    ///
    /// The size of `indices`'s last axis (call it `K`) indexes the leading `K` dims of `self`;
    /// the batch shape `indices.shape[0..M-1]` is preserved. `values` has shape
    /// `indices.shape[0..M-1] ++ self.shape[K..D]`. Constraints: `K <= D` and `M >= 1`.
    ///
    /// # Arguments
    ///
    /// * `indices` - The indices of the elements to scatter.
    /// * `values` - The complex values to scatter into the tensor.
    /// * `update` - The operation used to update the existing values at the indexed positions.
    ///
    /// # Warning
    ///
    /// Not all backends have runtime bound checks for the indices, so make sure they are valid.
    /// Otherwise, out of bounds indices could lead to unexpected results instead of panicking.
    pub fn scatter_nd<const M: usize, const DV: usize>(
        self,
        indices: Tensor<M, Int>,
        values: SplitComplexTensor<DV, K>,
        update: IndexingUpdateOp,
    ) -> Self {
        // check!(TensorCheck::scatter_nd::<D, M, DV>(
        //     &self.shape(),
        //     &indices.shape(),
        //     &values.shape()
        // ));
        let indices = indices.primitive;
        let SplitComplexTensor::<D, K> { real, imag, .. } = self;
        let SplitComplexTensor::<DV, K> {
            real: real_values,
            imag: imag_values,
            ..
        } = values;
        SplitComplexTensor::new(
            K::scatter_nd(real,indices.clone(), real_values, update),
            K::scatter_nd(imag,indices, imag_values, update),
        )
    }

    /// Multi-dimensional gather: collect complex slices from the tensor at multi-index
    /// locations specified by `indices`.
    ///
    /// The size of `indices`'s last axis (call it `K`) indexes the leading `K` dims of `self`;
    /// the batch shape `indices.shape[0..M-1]` is preserved. The output has shape
    /// `indices.shape[0..M-1] ++ self.shape[K..D]`. Constraints: `K <= D` and `M >= 1`.
    ///
    /// # Warning
    ///
    /// Not all backends have runtime bound checks for the indices, so make sure they are valid.
    /// Otherwise, out of bounds indices could lead to unexpected results instead of panicking.
    pub fn gather_nd<const M: usize, const DV: usize>(
        self,
        indices: Tensor<M, Int>,
    ) -> SplitComplexTensor<DV, K> {
        let indices = indices.primitive;
        let SplitComplexTensor::<D, K> { real, imag, .. } = self;
        //check!(TensorCheck::gather_nd::<D, M, DV>(&indices.shape()));
        SplitComplexTensor::new(K::gather_nd(real,indices.clone()), K::gather_nd(imag,indices))
    }
}
//impl<B, F> Numeric<B> for SplitComplexTensor<F>
impl<const D: usize, K:Numeric> SplitComplexTensor<D,K>
{
    /// Applies element-wise addition operation.
    ///
    /// `y = x2 + x1`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex tensor to add.
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, rhs: Self) -> Self {
        SplitComplexTensor::new(K::add(self.real, rhs.real), K::add(self.imag, rhs.imag))
    }

    /// Applies element-wise addition operation with a scalar.
    ///
    /// `y = x + s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to add, element-wise.
    pub fn add_scalar(self, rhs: burn_std::Scalar) -> Self {
        let device = self.device();
        let shape = self.shape();
        let scalar_complex = rhs.elem::<Complex<f64>>();
        let scalar_tensor = Self::full(shape, scalar_complex, &device);
        self.add(scalar_tensor)
    }

    /// Applies element-wise subtraction operation.
    ///
    /// `y = x2 - x1`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex tensor to subtract.
    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, rhs: Self) -> Self {
        SplitComplexTensor::new(K::sub(self.real, rhs.real), K::sub(self.imag, rhs.imag))
    }

    /// Applies element-wise subtraction operation with a scalar.
    ///
    /// `y = x - s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to subtract, element-wise.
    pub fn sub_scalar(self, rhs: burn_std::Scalar) -> Self {
        let device = self.device();
        let shape = self.shape();
        let scalar_complex = rhs.elem::<Complex<f64>>();
        let scalar_tensor = Self::full(shape, scalar_complex, &device);
        self.sub(scalar_tensor)
    }

    /// Applies element-wise division operation.
    ///
    /// `y = x2 / x1`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex tensor to divide by.
    #[allow(clippy::should_implement_trait)]
    pub fn div(self, rhs: Self) -> Self {
        // (a + i b) / (c + i d) == [(a + i b) * (c - i d)] / (c*c + d*d)
        //   == [(a*c + b*d) / (c*c + d*d)] + i [(b*c - a*d) / (c*c + d*d)]
        let norm_sqr = rhs.clone().squared_norm().primitive;
        SplitComplexTensor::new(
            K::div(K::add(K::mul(self.real.clone(), rhs.real.clone()), K::mul(self.imag.clone(), rhs.imag.clone())), norm_sqr.clone()),
            K::div(K::sub(K::mul(self.imag, rhs.real), K::mul(self.real, rhs.imag)), norm_sqr),
        )
    }

    /// Applies element-wise division operation with a scalar.
    ///
    /// `y = x / s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to divide by, element-wise.
    pub fn div_scalar(self, rhs: burn_std::Scalar) -> Self {
        let device = self.device();
        let shape = self.shape();
        let scalar_complex = rhs.elem::<Complex<f64>>();
        let scalar_tensor = Self::full(shape, scalar_complex, TensorCreationOptions::new(device));
        self.div(scalar_tensor)
    }

    /// Applies element-wise the remainder operation.
    ///
    /// `y = x2 % x1`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex tensor to compute the remainder with.
    pub fn remainder(self, rhs: Self) -> Self {
        // Componentwise remainder (matching Complex<E> Rem impl)
        SplitComplexTensor::new(K::remainder(self.real, rhs.real), K::remainder(self.imag, rhs.imag))
    }

    /// Applies element-wise the remainder operation with a scalar.
    ///
    /// `y = x % s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to compute the remainder with, element-wise.
    pub fn remainder_scalar(self, rhs: burn_std::Scalar) -> Self {
        use burn_std::cast::ToElement;
        let rhs = rhs.elem::<Complex<f64>>();
        SplitComplexTensor::new(
            K::remainder_scalar(self.real, burn_std::Scalar::Float(rhs.real)),
            K::remainder_scalar(self.imag, burn_std::Scalar::Float(rhs.imag)),
        )
    }

    /// Applies element-wise multiplication operation.
    ///
    /// `y = x2 * x1`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex tensor to multiply.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, rhs: Self) -> Self {
        // (a + i b) * (c + i d) == (a*c - b*d) + i (a*d + b*c)
        SplitComplexTensor::new(
            K::sub(K::mul(self.real.clone(), rhs.real.clone()), K::mul(self.imag.clone(), rhs.imag.clone())),
            K::add(K::mul(self.real, rhs.imag), K::mul(rhs.real, self.imag)),
        )
    }

    /// Applies element-wise multiplication operation with a scalar.
    ///
    /// `y = x * s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to multiply, element-wise.
    pub fn mul_scalar(self, rhs: burn_std::Scalar) -> Self {
        let device = self.device();
        let shape = self.shape();
        let scalar_complex = rhs.elem::<Complex<f64>>();
        let scalar_tensor = Self::full(shape, scalar_complex, TensorCreationOptions::new(device));
        self.mul(scalar_tensor)
    }

    /// Switch sign of each element in the complex tensor.
    ///
    /// `y = -x`
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        SplitComplexTensor::new(K::neg(self.real), K::neg(self.imag))
    }

    

    /// Aggregate all elements in the complex tensor with the sum operation.
    pub fn sum(self) -> Self {
        SplitComplexTensor::new(K::sum(self.real), K::sum(self.imag))
    }

    /// Aggregate all elements along the given dimension in the complex tensor with the sum operation.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements.
    pub fn sum_dim(self, dim: usize) -> Self {
        SplitComplexTensor::new(K::sum_dim(self.real, dim), K::sum_dim(self.imag, dim))
    }

    

    

    /// Aggregate all elements in the complex tensor with the mean operation.
    pub fn mean(self) -> Self {
        SplitComplexTensor::new(K::mean(self.real), K::mean(self.imag))
    }

    /// Aggregate all elements along the given dimension in the complex tensor with the mean operation.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements.
    pub fn mean_dim(self, dim: usize) -> Self {
        SplitComplexTensor::new(K::mean_dim(self.real, dim), K::mean_dim(self.imag, dim))
    }

    /// Computes the cumulative sum of complex elements along the given dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to compute the cumulative sum.
    pub fn cumsum(self, dim: usize) -> Self {
        // cumsum is linear, so it works componentwise
        SplitComplexTensor::new(K::cumsum(self.real,dim), K::cumsum(self.imag,dim))
    }

    /// Create a random complex tensor of the given shape where each element is sampled from
    /// the given distribution.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor.
    /// * `distribution` - The distribution to sample from.
    /// * `options` - Options to control creation, including device and dtype.
    pub fn random<S: Into<Shape>>(
        shape: S,
        distribution: Distribution,
        options: impl Into<TensorCreationOptions>,
    ) -> Self {
        // Use the given dtype when provided, otherwise default device dtype
        let opt = options.into();
        let shape= shape.into();
        let dtype = complex_utils::complex_to_real_dtype(opt.resolve_dtype::<K>());
        Self::new(
            K::random(shape.clone(), distribution, &opt.device, dtype).into(),
            K::random(shape, distribution, &opt.device, dtype).into(),
        )
    }

    /// Converts the data of the current tensor.
    ///
    /// # Note
    ///
    /// For better performance, prefer using a [Transaction](burn_std::Transaction) when reading multiple
    /// tensors at once. This may improve laziness, especially if executed on a different
    /// thread in native environments.
    pub fn into_data(self) -> TensorData {
        self.try_into_data().expect(
            "Error while reading data: use `try_into_data` instead to catch the error at runtime",
        )
    }
    /// Converts the data of the current tensor and returns any error that might have occurred since the
    /// last time the device was synchronized.
    ///
    /// # Note
    ///
    /// For better performance, prefer using a [Transaction](burn_std::Transaction) when reading multiple
    /// tensors at once. This may improve laziness, especially if executed on a different
    /// thread in native environments.
    pub fn try_into_data(self) -> Result<TensorData, ExecutionError> {
        try_read_sync(self.into_data_async()).expect(
            "Failed to read tensor data synchronously.
        This can happen on platforms that don't support blocking futures like WASM.
        If possible, try using into_data_async instead.",
        )
    }

    /// Converts the data of the current tensor.
    ///
    /// # Note
    ///
    /// For better performance, prefer using a [Transaction](burn_std::Transaction) when reading multiple
    /// tensors at once. This may improve laziness, especially if executed on a different
    /// thread in native environments.
    pub fn to_data(&self) -> TensorData {
        self.clone().into_data()
    }

    /// Applies the matrix multiplication operation.
    ///
    /// `C = AB`
    pub fn matmul(self, rhs: Self) -> Self {
        // (A + iB)(C + iD) = (AC - BD) + i(AD + BC)
        SplitComplexTensor::new(
            K::sub(K::matmul(self.real.clone(), rhs.real.clone()), K::matmul(self.imag.clone(), rhs.imag.clone())),
            K::add(K::matmul(self.real, rhs.imag), K::matmul(self.imag, rhs.real)),
        )
    }
}

impl<const D: usize, K> SplitComplexTensor<D,K>
where K: Numeric+ BasicOps+FloatMathOps 
{
    /// Applies element-wise power operation with an integer tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The integer tensor to apply the power operation with.
    pub fn powi(self, other: Tensor<D, Int>) -> Self {
        self.powf(other.float())
    }

    /// Applies element-wise power operation with a floating-point tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The floating-point tensor to apply the power operation with.
    pub fn powf(self, other: Tensor<D, Float>) -> Self {
        // z^w = exp(w * ln(z)) where w is a real tensor
        let log_z = self.log();
        let w_log_z = SplitComplexTensor::new(K::mul(other.primitive.clone() , log_z.real), K::mul(other.primitive, log_z.imag));
        w_log_z.exp()
    }

    /// Applies element-wise power operation with an integer scalar.
    ///
    /// # Arguments
    ///
    /// * `other` - The scalar to apply the power operation with.
    pub fn powi_scalar<E: ElementConversion>(self, other: E) -> Self {
        self.powf_scalar(other)
    }

    /// Applies element-wise power operation with a floating-point scalar.
    ///
    /// # Arguments
    ///
    /// * `other` - The scalar to apply the power operation with.
    fn powf_scalar<E: ElementConversion>(self, other: E) -> Self {
        let other = Scalar::new(other.elem::<f64>(), &burn_std::complex_utils::complex_to_real_dtype(self.dtype()));
        let log_z = self.log();
        let w_log_z = SplitComplexTensor::new(
            K::mul_scalar(log_z.real, other.clone()),
            K::mul_scalar(log_z.imag, other),
        );
        w_log_z.exp()
    }

    /// Applies element-wise complex cosine.
    pub fn cos(self) -> Self {
        // cos(a + bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
        SplitComplexTensor::new(
            K::mul(K::cos(self.real.clone()), K::cosh(self.imag.clone())),
            K::neg(K::mul(K::sin(self.real) , K::sinh(self.imag)))
        )
    }

    /// Applies element-wise complex tangent.
    pub fn tan(self) -> Self {
        // tan(z) = sin(z) / cos(z)
        // Compute sin(a), cos(a), sinh(b), cosh(b) once and share between numerator/denominator.
        let sin_a = K::sin(self.real.clone());
        let cos_a = K::cos(self.real);
        let sinh_b = K::sinh(self.imag.clone());
        let cosh_b = K::cosh(self.imag);
        let sin_z = SplitComplexTensor::new(
            K::mul(sin_a.clone(), cosh_b.clone()),
            K::mul(cos_a.clone(), sinh_b.clone()),
        );
        let cos_z = SplitComplexTensor::new(K::mul(cos_a, cosh_b), K::neg(K::mul(sin_a, sinh_b)));
        sin_z.div(cos_z)
    }

    /// Applies element-wise complex arccosine.
    pub fn acos(self) -> Self {
        // acos(z) = -i * ln(z + i * sqrt(1 - z²))
        let device = self.device();
        let shape = self.shape();
        let fdtype = self.inner_dtype().into();
        let ones = SplitComplexTensor::new(
            K::ones(shape.clone(), &device, fdtype),
            K::zeros(shape, &device, fdtype),
        );
        // 1 - z²
        let z_sq = self.clone().mul(self.clone());
        let one_minus_z_sq = ones.sub(z_sq);
        // i * sqrt(1 - z²): multiply by i via (-imag, real)
        let sqrt_term = one_minus_z_sq.sqrt();
        let i_sqrt = SplitComplexTensor::new(K::neg(sqrt_term.imag), sqrt_term.real);
        // z + i*sqrt(1 - z²)
        let inner = self.add(i_sqrt);
        // -i * ln(inner): multiply by -i via (imag, -real)
        let log_inner = inner.log();
        SplitComplexTensor::new(log_inner.imag, K::neg(log_inner.real))
    }

    /// Computes the cumulative product of complex elements along the given dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to compute the cumulative product.
    pub fn cumprod(self, dim: usize) -> Self {
        // cumprod(z, dim) = exp(cumsum(log(z), dim))
        self.log().cumsum(dim).exp()
    }

    /// Aggregate all elements in the complex tensor with the product operation.
    pub fn prod(self) -> Self {
        // prod(z) = exp(sum(log(z)))
        self.log().sum().exp()
    }

    /// Applies element-wise complex hyperbolic arccosine.
    pub fn acosh(self) -> Self {
        // acosh(z) = ln(z + sqrt(z² - 1))
        let device = self.device();
        let shape = self.shape();
        let fdtype = self.inner_dtype().into();
        let ones = SplitComplexTensor::new(
            K::ones(shape.clone(), &device, fdtype),
            K::zeros(shape, &device, fdtype),
        );
        // z² - 1
        let z_sq = self.clone().mul(self.clone());
        let z_sq_minus_one = z_sq.sub(ones);
        // z + sqrt(z² - 1)
        let sqrt_term = z_sq_minus_one.sqrt();
        let inner = self.add(sqrt_term);
        inner.log()
    }

    /// Applies element-wise complex arcsine.
    pub fn asin(self) -> Self {
        // asin(z) = -i * ln(i*z + sqrt(1 - z²))
        let device = self.device();
        let shape = self.shape();
        let fdtype = self.inner_dtype().into();
        let ones = SplitComplexTensor::new(
            K::ones(shape.clone(), &device, fdtype),
            K::zeros(shape, &device, fdtype),
        );
        // z² and i*z — clone before partial-moving self
        let z_sq = self.clone().mul(self.clone());
        // i*z = (-imag, real)
        let i_z = SplitComplexTensor::new(K::neg(self.imag), self.real);
        // 1 - z²
        let one_minus_z_sq = ones.sub(z_sq);
        // i*z + sqrt(1 - z²)
        let sqrt_term = one_minus_z_sq.sqrt();
        let inner = i_z.add(sqrt_term);
        // -i * ln(inner): (imag, -real)
        let log_inner = inner.log();
        SplitComplexTensor::new(log_inner.imag, K::neg(log_inner.real))
    }

    /// Applies element-wise complex hyperbolic arcsine.
    pub fn asinh(self) -> Self {
        // asinh(z) = ln(z + sqrt(z² + 1))
        let device = self.device();
        let shape = self.shape();
        let fdtype = self.inner_dtype().into();
        let ones = SplitComplexTensor::new(
            K::ones(shape.clone(), &device, fdtype),
            K::zeros(shape, &device, fdtype),
        );
        // z² + 1
        let z_sq = self.clone().mul(self.clone());
        let z_sq_plus_one = z_sq.add(ones);
        // z + sqrt(z² + 1)
        let sqrt_term = z_sq_plus_one.sqrt();
        let inner = self.add(sqrt_term);
        inner.log()
    }

    /// Applies element-wise complex arctangent.
    pub fn atan(self) -> Self {
        // atan(z) = (-i/2) * ln((1 + i*z) / (1 - i*z))
        let device = self.device();
        let shape = self.shape();
        
        let ones = Self::ones(shape, TensorCreationOptions::new(device).with_dtype(real_to_complex_dtype(self.dtype()).into()));
        // i*z = (-imag, real)
        let i_z = SplitComplexTensor::new(K::neg(self.imag), self.real);
        // 1 + i*z and 1 - i*z
        // ln((1 + i*z) / (1 - i*z))
        let log_ratio = (
            (ones.clone() + i_z.clone()) /
            (ones - i_z)
        ).log();
        // (-i/2) * log_ratio: -i*(a+bi) = (b, -a), then /2
        SplitComplexTensor::new(
            K::div_scalar(log_ratio.imag, burn_std::Scalar::Float(2.0)),
            K::neg(K::div_scalar(
                log_ratio.real,
                burn_std::Scalar::Float(2.0),
            )),
        )
    }

    /// Applies element-wise complex hyperbolic arctangent.
    pub fn atanh(self) -> Self {
        // atanh(z) = (1/2) * ln((1 + z) / (1 - z))
        let device = self.device();
        let shape = self.shape();
        let fdtype = self.inner_dtype().into();
        let ones = Self::ones(shape, TensorCreationOptions::new(device).with_dtype(real_to_complex_dtype(fdtype).into()));
        let one_plus_z = ones.clone() + self.clone();
        let one_minus_z = ones - self;
        let log_ratio = (one_plus_z / one_minus_z).log();
        SplitComplexTensor::new(
            K::div_scalar(log_ratio.real, burn_std::Scalar::Float(2.0)),
            K::div_scalar(log_ratio.imag, burn_std::Scalar::Float(2.0)),
        )
    }

    /// Applies element-wise complex natural logarithm.
    ///
    /// For a complex number `z = r · exp(i · θ)`, computes `ln(r) + i · θ`.
    pub fn log(self) -> Self {
        // formula: ln(z) = ln|z| + i*arg(z)
        // where |z| = sqrt(real^2 + imag^2) and arg(z) = atan2(imag, real)

        // Compute norm: sqrt(real^2 + imag^2)
        let norm = self.clone().squared_norm().sqrt();

        // Compute arg: atan2(imag, real)
        let arg = atan2_impl(self.imag, self.real);

        SplitComplexTensor::<D,K>::new(norm.log().primitive, arg)
    }
    /// Applies element-wise complex square root.
    pub fn sqrt(self) -> Self {
        // sqrt(z) = from_polar(sqrt(|z|), arg(z) / 2)
        let abs = self.clone().magnitude();
        let sqrt_abs = abs.sqrt();
        let arg = atan2_impl(self.imag,self.real);
        let half_arg = K::div_scalar(arg, burn_std::Scalar::Float(2.0));
        Self::from_polar(Tensor::<D,K>::new(sqrt_abs.primitive), Tensor::new(half_arg))
    }

    /// Aggregate all elements along the given dimension in the complex tensor with the product operation.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements.
    pub fn prod_dim(self, dim: usize) -> Self {
        // prod_dim(z, dim) = exp(sum_dim(log(z), dim))
        self.log().sum_dim(dim).exp()
    }

    /// Returns the signs of the elements of the complex tensor.
    ///
    /// For a non-zero element `z`, returns `z / |z|`. For zero, returns zero.
    pub fn sign(self) -> Self {
        // sign(z) = z / |z| = from_polar(1, arg(z))
        let abs = self.clone().magnitude().primitive;
        SplitComplexTensor::new(K::div(self.real, abs.clone()), K::div(self.imag, abs))
    }
}

// ComplexOnlyOps
impl<const D: usize, K> SplitComplexTensor<D,K>
where
    K: Numeric + BasicOps+FloatMathOps,
{
    /// Returns the complex conjugate of each element.
    ///
    /// For a complex number `a + bi`, the conjugate is `a - bi`.
    pub fn conj(self) -> Self {
        // conj(a + bi) = a - bi
        SplitComplexTensor::new(self.real, K::neg(self.imag))
    }

    /// Returns the argument (phase angle) of each element, in radians.
    ///
    /// For a complex number `a + bi`, the phase is `atan2(b, a)`, ranging from `-π` to `π`.
    pub fn phase(self) -> Tensor<D, K> {
        // arg(a + bi) = atan2(b, a)
        Tensor::new(atan2_impl(self.imag, self.real))
    }

    /// Returns the magnitude (absolute value, modulus) of each element.
    ///
    /// For a complex number `a + bi`, the magnitude is `sqrt(a² + b²)`.
    pub fn magnitude(self) -> Tensor<D, Float> {
        //could use a hypot function for float kinds
        Tensor::new(Float::sqrt(K::add(K::mul(self.real.clone(), self.real), K::mul(self.imag.clone(), self.imag))))
        
    }

    


    /// Applies element-wise complex exponential.
    ///
    /// For a complex number `a + bi`, computes `exp(a) * (cos(b) + i·sin(b))`.
    pub fn exp(self) -> Self {
        // formula: e^(a + bi) = e^a * (cos(b) + i*sin(b)) = from_polar(e^a, b)
        //TODO: add the checks for corner cases +∞, -∞, and NaN
        //https://github.com/skewballfox/burn/blob/67d84b677b3d718cb25fbdc2535dbf04706b0863/crates/burn-complex/src/base/element.rs#L322-L323
        let exp_real = K::exp(self.real);
        let cos_imag = K::cos(self.imag.clone());
        let sin_imag = K::sin(self.imag);
        SplitComplexTensor::new(K::mul(exp_real.clone(), cos_imag), K::mul(exp_real, sin_imag))
    }

    /// Applies element-wise complex sine.
    pub fn sin(self) -> Self {
        // sin(a + bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
        SplitComplexTensor::new(
            K::mul(K::sin(self.real.clone()), K::cosh(self.imag.clone())),
            K::mul(K::cos(self.real), K::sinh(self.imag)),
        )
    }

    /// Create a complex tensor from separate real and imaginary data.
    ///
    /// # Arguments
    ///
    /// * `real` - The real part data.
    /// * `imag` - The imaginary part data.
    pub fn from_parts<T: Into<TensorData>>(real: T, imag: T, device: &Device) -> Self {
        let (real, imag) = (real.into(), imag.into());
        let dtype = real.dtype;
        assert_eq!(
            dtype,
            imag.dtype,
            "from_parts: real and imaginary data must have the same dtype, got {:?} and {:?}",
            dtype,
            imag.dtype
        );
        let real = K::from_data(real, device, dtype);
        let imag = K::from_data(imag, device, dtype);
        Self::new(real, imag)
    }

    /// Create a complex tensor from interleaved (real, imaginary) data.
    ///
    /// The input data should contain alternating real and imaginary values.
    ///
    /// # Arguments
    ///
    /// * `data` - Interleaved complex data.
    /// * `device` - The device to create the tensor on.
    pub fn from_interleaved_data(data: TensorData, device: &Device) -> Self {
        Self::from_split_data(
            burn_std::complex_utils::split_from_interleaved_data(data),
            device,
        )
    }

    /// Create a complex tensor from polar form.
    ///
    /// Constructs a complex tensor where each element `z = r · exp(i · θ)`,
    /// given magnitude `r` and phase angle `θ`.
    ///
    /// # Arguments
    ///
    /// * `magnitude` - The magnitude (modulus) of each element.
    /// * `phase` - The phase angle of each element, in radians.
    pub fn from_polar(magnitude: Tensor<D, K>, phase: Tensor<D, K>) -> Self {
        Self::new(
            (magnitude.clone() * phase.clone().cos().clone()).primitive,
            (magnitude * phase.sin()).primitive,
        )
    }

    
}

use crate::Bool;
use crate::BroadcastArgs;
use crate::Device;
use crate::Float;
use crate::Int;
use crate::ReshapeArgs;
use crate::Tensor;
use crate::TensorCreationOptions;
use crate::kind::Basic;
use crate::kind::Numeric;
use crate::ops::BasicOps;
use crate::ops::BridgeTensor;
use crate::ops::FloatMathOps;
use crate::powf_impl;


// SplitComplexTensor + SplitComplexTensor
impl<const D: usize, K: Numeric + BasicOps> core::ops::Add<Self> for SplitComplexTensor<D,K> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::add(self, rhs)
    }
}

// SplitComplexTensor + Tensor<D, Float> — adds real tensor to the real part
impl<const D: usize, K: Numeric + BasicOps> core::ops::Add<Tensor<D, K>> for SplitComplexTensor<D,K> {
    type Output = Self;

    fn add(self, rhs: Tensor<D, K>) -> Self::Output {
        SplitComplexTensor::new(K::add(self.real, rhs.primitive), self.imag)
    }
}

// SplitComplexTensor + scalar (concrete types to avoid coherence conflict with ElementConversion)
macro_rules! impl_complex_tensor_add_scalar {
    ($($t:ty),*) => {
        $(
            impl<const D: usize, K: Numeric + BasicOps> core::ops::Add<$t> for SplitComplexTensor<D,K> {
                type Output = Self;

                fn add(self, rhs: $t) -> Self::Output {
                    Self::add_scalar(self, burn_std::Scalar::Float(rhs as f64))
                }
            }
        )*
    }
}
impl_complex_tensor_add_scalar!(f32, f64, i32, i64, u32, u64);

impl<const D: usize, K: Numeric + BasicOps, E: Element> core::ops::Add<Complex<E>>
    for SplitComplexTensor<D,K>
{
    type Output = Self;

    fn add(self, rhs: Complex<E>) -> Self::Output {
        Self::add_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
    }
}
// Tensor - tensor
impl<const D: usize, K: Numeric> core::ops::Sub<Self> for SplitComplexTensor<D,K> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::sub(self, rhs)
    }
}

// SplitComplexTensor - Tensor<D, Float>
impl<const D: usize, K: Numeric> core::ops::Sub<Tensor<D, K>> for SplitComplexTensor<D,K> {
    type Output = Self;

    fn sub(self, rhs: Tensor<D, K>) -> Self::Output {
        let prim = rhs.primitive;
        SplitComplexTensor::new(K::sub(self.real, prim), self.imag)
    }
}

// SplitComplexTensor - scalar
macro_rules! impl_complex_tensor_sub_scalar {
    ($($t:ty),*) => {
        $(
            impl<const D: usize, K: Numeric> core::ops::Sub<$t> for SplitComplexTensor<D,K> {
                type Output = Self;

                fn sub(self, rhs: $t) -> Self::Output {
                    Self::sub_scalar(self, burn_std::Scalar::Float(rhs as f64))
                }
            }
        )*
    }
}
impl_complex_tensor_sub_scalar!(f32, f64, i32, i64, u32, u64);

impl<const D: usize, K: Numeric, E: Element> core::ops::Sub<Complex<E>>
    for SplitComplexTensor<D,K>
{
    type Output = Self;

    fn sub(self, rhs: Complex<E>) -> Self::Output {
        Self::sub_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
    }
}

// Tensor * tensor
impl<const D: usize, K: Numeric> core::ops::Mul<Self> for SplitComplexTensor<D,K> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::mul(self, rhs)
    }
}

// SplitComplexTensor * Tensor<D, Float>
impl<const D: usize, K: Numeric> core::ops::Mul<Tensor<D, K>> for SplitComplexTensor<D,K> {
    type Output = Self;

    fn mul(self, rhs: Tensor<D, K>) -> Self::Output {
        let prim = rhs.primitive;
        
        SplitComplexTensor::new(K::mul(self.real, prim.clone()), K::mul(self.imag, prim))
    }
}

// SplitComplexTensor * scalar
macro_rules! impl_complex_tensor_mul_scalar {
    ($($t:ty),*) => {
        $(
            impl<const D: usize, K: Numeric> core::ops::Mul<$t> for SplitComplexTensor<D,K> {
                type Output = Self;

                fn mul(self, rhs: $t) -> Self::Output {
                    Self::mul_scalar(self, burn_std::Scalar::Float(rhs as f64))
                }
            }
        )*
    }
}
impl_complex_tensor_mul_scalar!(f32, f64, i32, i64, u32, u64);

impl<const D: usize, K: Numeric, E: Element> core::ops::Mul<Complex<E>>
    for SplitComplexTensor<D,K>
{
    type Output = Self;

    fn mul(self, rhs: Complex<E>) -> Self::Output {
        Self::mul_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
    }
}

// Tensor / tensor
impl<const D: usize, K: Numeric> core::ops::Div<Self> for SplitComplexTensor<D,K> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self::div(self, rhs)
    }
}

// SplitComplexTensor / Tensor<D, Float>
impl<const D: usize, K: Numeric> core::ops::Div<Tensor<D, Float>> for SplitComplexTensor<D,K> {
    type Output = Self;

    fn div(self, rhs: Tensor<D, Float>) -> Self::Output {
        let prim = rhs.primitive;
        SplitComplexTensor::new(K::div(self.real, prim.clone()), K::div(self.imag, prim))
    }
}

// SplitComplexTensor / scalar
macro_rules! impl_complex_tensor_div_scalar {
    ($($t:ty),*) => {
        $(
            impl<const D: usize, K: Numeric> core::ops::Div<$t> for SplitComplexTensor<D,K> {
                type Output = Self;

                fn div(self, rhs: $t) -> Self::Output {
                    Self::div_scalar(self, burn_std::Scalar::Float(rhs as f64))
                }
            }
        )*
    }
}
impl_complex_tensor_div_scalar!(f32, f64, i32, i64, u32, u64);

impl<const D: usize, K: Numeric, E: Element> core::ops::Div<Complex<E>>
    for SplitComplexTensor<D,K>
{
    type Output = Self;

    fn div(self, rhs: Complex<E>) -> Self::Output {
        Self::div_scalar(self, Scalar::Complex(rhs.to_complex64()))
    }
}

// Tensor % tensor
impl<const D: usize, K: Numeric> core::ops::Rem<Self> for SplitComplexTensor<D,K> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self::remainder(self, rhs)
    }
}

// SplitComplexTensor % Tensor<D, Float>
impl<const D: usize, K: Numeric> core::ops::Rem<Tensor<D, Float>> for SplitComplexTensor<D,K> {
    type Output = Self;

    fn rem(self, rhs: Tensor<D, Float>) -> Self::Output {
        let rhs = rhs.primitive;
        SplitComplexTensor::new(K::remainder(self.real, rhs.clone()), K::remainder(self.imag, rhs))
    }
}

// SplitComplexTensor % scalar
macro_rules! impl_complex_tensor_rem_scalar {
    ($($t:ty),*) => {
        $(
            impl<const D: usize, K: Numeric> core::ops::Rem<$t> for SplitComplexTensor<D,K> {
                type Output = Self;

                fn rem(self, rhs: $t) -> Self::Output {
                    Self::remainder_scalar(self, burn_std::Scalar::Float(rhs as f64))
                }
            }
        )*
    }
}
impl_complex_tensor_rem_scalar!(f32, f64, i32, i64, u32, u64);

impl<const D: usize,K: Numeric, E: Element> core::ops::Rem<Complex<E>>
    for SplitComplexTensor<D,K>
{
    type Output = Self;

    fn rem(self, rhs: Complex<E>) -> Self::Output {
        Self::remainder_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
    }
}

impl<const D: usize, K: Numeric> core::ops::Neg for SplitComplexTensor<D,K> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::neg(self)
    }
}
