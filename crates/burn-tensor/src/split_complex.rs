use std::os::unix::raw::dev_t;

use alloc::vec::Vec;
use burn_backend::ops::ComplexTensorOps;
use burn_backend::ops::FloatTensorOps;
use burn_backend::try_read_sync;
use burn_backend::{
    Backend, BackendTypes, DTypeUsageSet, ExecutionError, TensorMetadata, tensor::Device,
};
use burn_backend::{CBT, ComplexTensor, ComplexTensorBackend, DefaultComplexOps, SplitLayout};
use burn_dispatch::{Dispatch, DispatchTensor};
use burn_std::DType;
use burn_std::{Complex, ComplexElement, ElementComparison, Scalar, TensorData, cast::ToElement};
use burn_std::{ComplexDType, FloatDType, IndexingUpdateOp, Shape, SplitTensorData};
use bytemuck::Pod;

#[derive(Debug, Clone)]
pub struct SplitComplexTensor<B: Backend, const D: usize> {
    _phantom: core::marker::PhantomData<B>,
    pub(crate) real: DispatchTensor,
    pub(crate) imag: DispatchTensor,
}

impl<B: Backend<FloatTensorPrimitive = T>, T: TensorMetadata, const D: usize>
    SplitComplexTensor<B, D>
{
    pub fn new(real: DispatchTensor, imag: DispatchTensor) -> Self {
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
            _phantom: core::marker::PhantomData,
            real,
            imag,
        }
    }

    pub fn device(&self) -> Device<B> {
        self.real.device().into()
    }

    pub fn shape(&self) -> Shape {
        self.real.shape().clone()
    }

    pub fn inner_dtype(&self) -> burn_std::DType {
        self.real.dtype()
    }

    pub fn from_parts_data(real: TensorData, imag: TensorData, device: &B::Device) -> Self {
        let real_tensor = DispatchTensor::from_data(real, device);
        let imag_tensor = DispatchTensor::from_data(imag, device);
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
        Self::new(real_tensor, imag_tensor)
    }

    pub fn inner_dtype(&self) -> burn_std::DType {
        self.real.dtype()
    }

    pub fn from_real_data(data: TensorData, device: &B::Device) -> Self {
        let shape = data.shape.clone();
        let dtype = data.dtype;
        Self {
            _phantom: core::marker::PhantomData,
            real: DispatchTensor::from_data(data, device),
            imag: DispatchTensor::zeros(shape, device, dtype.into()),
        }
    }

    pub fn from_imag_data(data: TensorData, device: &B::Device) -> Self {
        let shape = data.shape.clone();
        let dtype = data.dtype;
        Self {
            _phantom: core::marker::PhantomData,
            real: DispatchTensor::zeros(shape, device, dtype.into()),
            imag: DispatchTensor::from_data(data, device),
        }
    }

    pub fn from_split_data(data: SplitTensorData, device: &B::Device) -> Self {
        let SplitTensorData {
            real_bytes: real,
            imag_bytes: imag,
            shape,
            dtype,
        } = data;

        Self {
            _phantom: core::marker::PhantomData,
            real: DispatchTensor::from_data(
                TensorData::from_bytes(real, shape.clone(), dtype),
                device,
            ),
            imag: DispatchTensor::from_data(TensorData::from_bytes(imag, shape, dtype), device),
        }
    }
    pub fn real(self) -> DispatchTensor {
        self.real
    }
    pub fn imag(self) -> DispatchTensor {
        self.imag
    }
    pub fn real_ref(&self) -> &DispatchTensor {
        &self.real
    }
    pub fn imag_ref(&self) -> &DispatchTensor {
        &self.imag
    }
}

impl<B: Backend<FloatTensorPrimitive = T>, T: TensorMetadata + 'static, const D: usize>
    TensorMetadata for SplitComplexTensor<B, D>
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

#[derive(Debug, Clone)]
/// A newtype that wraps a real backend B and exposes a split-layout complex backend.
pub struct SplitBackend<B: Backend, const D: usize>(core::marker::PhantomData<B>);
impl<B: Backend, const D: usize> CBT for SplitBackend<B, D> {
    type ComplexTensorPrimitive = SplitComplexTensor<B, D>;

    type ComplexScalar = Complex<B::FloatElem>;
}
impl<B: Backend, const D: usize> BackendTypes for SplitBackend<B, D> {
    type Device = B::Device;

    type FloatTensorPrimitive = B::FloatTensorPrimitive;

    type FloatElem = B::FloatElem;

    type IntTensorPrimitive = B::IntTensorPrimitive;

    type IntElem = B::IntElem;

    type BoolTensorPrimitive = B::BoolTensorPrimitive;

    type BoolElem = B::BoolElem;

    type QuantizedTensorPrimitive = B::QuantizedTensorPrimitive;

    fn dtype_usage(device: &Self::Device, dtype: burn_std::DType) -> DTypeUsageSet {
        B::dtype_usage(device, dtype)
    }

    fn device_count(type_id: u16) -> usize {
        B::device_count(type_id)
    }

    type ComplexScalar = Complex<B::FloatElem>;

    type ComplexTensorPrimitive = SplitComplexTensor<B, D>;
}

impl<B: Backend, const D: usize> ComplexTensorBackend for SplitBackend<B, D>
where
    B::FloatElem: ElementComparison + Pod,
    B::FloatTensorPrimitive: TensorMetadata + 'static,
    B::ComplexScalar: Complex<B::FloatElem>,
{
    type InnerBackend = B;
    type Layout = SplitLayout<B>;

    fn complex_from_real_data(data: TensorData, device: &B::Device) -> ComplexTensor<Self> {
        // ComplexTensor<Self> = Complex<SplitComplexTensor<B::FloatTensorPrimitive>>
        // i.e. Complex { re: SplitComplexTensor { real, imag } }
        Self::ComplexTensorPrimitive::from_real_data(data, device)
    }

    fn complex_from_imag_data(data: TensorData, device: &B::Device) -> ComplexTensor<Self> {
        Self::ComplexTensorPrimitive::from_imag_data(data, device)
    }
    // Should these be a result
    fn complex_from_interleaved_data(data: TensorData, device: &B::Device) -> ComplexTensor<Self> {
        Self::ComplexTensorPrimitive::from_split_data(
            burn_std::complex_utils::split_from_interleaved_data(data),
            device,
        )
    }

    fn complex_from_parts_data(
        real_data: TensorData,
        imag_data: TensorData,
        device: &Self::Device,
    ) -> ComplexTensor<Self> {
        let real = DispatchTensor::from_data(real_data, device);
        let imag = DispatchTensor::from_data(imag_data, device);
        Self::ComplexTensorPrimitive::new(real, imag)
    }
}

impl<B, const D: usize> ComplexTensorOps<SplitBackend<B, D>> for SplitBackend<B, D>
where
    B: Backend,
    B::FloatElem: ElementComparison + Pod,
{
    fn to_complex(tensor: B::FloatTensorPrimitive) -> ComplexTensor<SplitBackend<B, D>> {
        let shape = tensor.shape().clone();
        let dtype = tensor.dtype().into();
        let device = &<Self as ComplexTensorBackend>::InnerBackend::float_device(&tensor);
        ComplexTensor::<SplitBackend<B, D>>::new(
            tensor.into(),
            DispatchTensor::zeros(shape, device, dtype),
        )
    }

    fn real(tensor: ComplexTensor<SplitBackend<B, D>>) -> B::FloatTensorPrimitive {
        tensor.real.into_primitive().to_float_tensor()
    }
    fn imag(tensor: ComplexTensor<SplitBackend<B, D>>) -> B::FloatTensorPrimitive {
        tensor.imag.into_primitive().to_float_tensor()
    }

    fn complex_not_equal_elem(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let (lhs_real, lhs_imag) = (lhs.real, lhs.imag);
        let rhs_real = rhs.real();
        let rhs_imag = rhs.imag();

        let real_cmp = B::float_not_equal_elem(
            lhs_real.into_primitive().to_float_tensor(),
            burn_std::Scalar::Float(rhs_real.to_f64()),
            out_dtype,
        );
        let imag_cmp = B::float_not_equal_elem(
            lhs_imag.into_primitive().to_float_tensor(),
            burn_std::Scalar::Float(rhs_imag.to_f64()),
            out_dtype,
        );
        B::bool_or(real_cmp, imag_cmp)
    }

    fn complex_equal_elem(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let (lhs_real, lhs_imag) = (lhs.real, lhs.imag);
        let rhs_real = rhs.real();
        let rhs_imag = rhs.imag();

        let real_cmp = B::float_equal_elem(
            lhs_real.into_primitive().to_float_tensor(),
            burn_std::Scalar::Float(rhs_real.to_f64()),
            out_dtype,
        );
        let imag_cmp = B::float_equal_elem(
            lhs_imag.into_primitive().to_float_tensor(),
            burn_std::Scalar::Float(rhs_imag.to_f64()),
            out_dtype,
        );
        B::bool_and(real_cmp, imag_cmp)
    }

    fn complex_equal(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let real_cmp = B::float_equal(
            lhs.real.into_primitive().to_float_tensor(),
            rhs.real.into_primitive().to_float_tensor(),
            out_dtype,
        );
        let imag_cmp = B::float_equal(
            lhs.imag.into_primitive().to_float_tensor(),
            rhs.imag.into_primitive().to_float_tensor(),
            out_dtype,
        );
        B::bool_and(real_cmp, imag_cmp)
    }

    fn complex_not_equal(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let real_cmp = B::float_not_equal(
            lhs.real.into_primitive().to_float_tensor(),
            rhs.real.into_primitive().to_float_tensor(),
            out_dtype,
        );
        let imag_cmp = B::float_not_equal(
            lhs.imag.into_primitive().to_float_tensor(),
            rhs.imag.into_primitive().to_float_tensor(),
            out_dtype,
        );
        B::bool_or(real_cmp, imag_cmp)
    }

    async fn complex_into_real_data(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> Result<TensorData, ExecutionError> {
        tensor
            .real
            .into_primitive()
            .to_float_tensor()
            .into_data()
            .await
    }

    async fn complex_into_imag_data(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> Result<TensorData, ExecutionError> {
        B::float_into_data(tensor.imag.into_primitive().to_float_tensor()).await
    }

    async fn complex_into_interleaved_data(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> Result<TensorData, ExecutionError> {
        let real_data = B::float_into_data(tensor.real.into_primitive().to_float_tensor()).await?;
        let imag_data = B::float_into_data(tensor.imag.into_primitive().to_float_tensor()).await?;
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
            real_bytes: real.into_data().await?.bytes,
            imag_bytes: imag.into_data().await?.bytes,
            shape,
            dtype,
        })
    }

    fn complex_device(tensor: &ComplexTensor<SplitBackend<B, D>>) -> B::Device {
        B::float_device(&tensor.real.into_primitive().to_float_tensor())
    }

    fn complex_add(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        ComplexTensor::<SplitBackend<B, D>>::new(lhs.real + rhs.real, lhs.imag + rhs.imag)
    }

    fn complex_sub(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        ComplexTensor::<SplitBackend<B, D>>::new(lhs.real - rhs.real, lhs.imag - rhs.imag)
    }

    fn complex_mul(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // (a + i b) * (c + i d) == (a*c - b*d) + i (a*d + b*c)
        ComplexTensor::<SplitBackend<B, D>>::new(
            (lhs.real * rhs.real) - (lhs.imag * rhs.imag),
            (lhs.real * rhs.imag) + (rhs.real * lhs.imag),
        )
    }

    fn complex_div(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // (a + i b) / (c + i d) == [(a + i b) * (c - i d)] / (c*c + d*d)
        //   == [(a*c + b*d) / (c*c + d*d)] + i [(b*c - a*d) / (c*c + d*d)]

        let norm_sqr = SplitBackend::<B, D>::complex_squared_norm(rhs.clone());

        ComplexTensor::<SplitBackend<B, D>>::new(
            ((lhs.real * rhs.real) + (lhs.imag * rhs.imag)) / norm_sqr,
            ((lhs.imag * rhs.real) - (lhs.real * rhs.imag)) / norm_sqr,
        )
    }
    fn abs(tensor: ComplexTensor<SplitBackend<B, D>>) -> B::FloatTensorPrimitive {
        //todo! https://github.com/tracel-ai/burn/issues/4836
        B::float_sqrt(SplitBackend::<B, D>::complex_squared_norm(tensor))
    }

    fn complex_from_parts(real: TensorData, imag: TensorData) -> ComplexTensor<SplitBackend<B, D>> {
        ComplexTensor::<SplitBackend<B, D>>::from_parts_data(real, imag, &B::default_device())
    }

    fn complex_exp(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        // formula: e^(a + bi) = e^a * (cos(b) + i*sin(b)) = from_polar(e^a, b)
        //TODO: add the checks for corner cases +∞, -∞, and NaN
        //https://github.com/skewballfox/burn/blob/67d84b677b3d718cb25fbdc2535dbf04706b0863/crates/burn-complex/src/base/element.rs#L322-L323
        let exp_real = tensor.real.exp();
        let cos_imag = tensor.imag.cos();
        let sin_imag = tensor.imag.sin();

        SplitComplexTensor::new(exp_real.clone() * cos_imag, exp_real * sin_imag)
    }

    fn complex_log(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        // formula: ln(z) = ln|z| + i*arg(z)
        // where |z| = sqrt(real^2 + imag^2) and arg(z) = atan2(imag, real)

        // Compute norm: sqrt(real^2 + imag^2)
        let real_sq = tensor.real * tensor.real;
        let imag_sq = tensor.imag * tensor.imag;
        let norm_sq = real_sq + imag_sq;
        let norm = norm_sq.sqrt();

        // Compute arg: atan2(imag, real)
        let arg = tensor.imag.atan2(tensor.real);

        SplitComplexTensor::new(norm.log(), arg)
    }

    fn complex_squared_norm(tensor: ComplexTensor<SplitBackend<B, D>>) -> B::FloatTensorPrimitive {
        let real_sq = tensor.real * tensor.real;
        let imag_sq = tensor.imag * tensor.imag;
        real_sq + imag_sq
    }

    fn complex_from_polar(
        magnitude: B::FloatTensorPrimitive,
        phase: B::FloatTensorPrimitive,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        ComplexTensor::<SplitBackend<B, D>>::new(
            (magnitude.clone() * B::float_cos(phase.clone())),
            (magnitude * B::float_sin(phase)),
        )
    }

    fn complex_gather(
        dim: usize,
        tensor: ComplexTensor<SplitBackend<B, D>>,
        indices: B::IntTensorPrimitive,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        ComplexTensor::<SplitBackend<B, D>>::new(
            DispatchTensor::float_gather(dim, tensor.real, indices.clone()),
            DispatchTensor::float_gather(dim, tensor.imag, indices),
        )
    }

    fn complex_scatter_add(
        dim: usize,
        tensor: ComplexTensor<SplitBackend<B, D>>,
        indices: B::IntTensorPrimitive,
        values: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        ComplexTensor::<SplitBackend<B, D>>::new(
            DispatchTensor::float_scatter_add(dim, tensor.real, indices.clone(), values.real),
            DispatchTensor::float_scatter_add(dim, tensor.imag, indices, values.imag),
        )
    }

    fn complex_random(
        shape: burn_std::Shape,
        distribution: burn_std::Distribution,
        device: &Device<B>,
        dtype: FloatDType,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        ComplexTensor::<SplitBackend<B, D>>::new(
            DispatchTensor::float_random(shape.clone(), distribution, device, dtype),
            DispatchTensor::float_random(shape, distribution, device, dtype),
        )
    }

    fn complex_to_device(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        device: &Device<B>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(tensor.real.to_device(device), tensor.imag.to_device(device))
    }

    fn complex_reshape(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        shape: burn_std::Shape,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            tensor.real.reshape(shape.clone()),
            tensor.imag.reshape(shape),
        )
    }

    fn complex_transpose(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(tensor.real.transpose(), tensor.imag.transpose())
    }

    fn complex_neg(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(-tensor.real, -tensor.imag)
    }

    fn conj(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        // conj(a + bi) = a - bi
        SplitComplexTensor::new(tensor.real, -tensor.imag)
    }

    fn complex_arg(tensor: ComplexTensor<SplitBackend<B, D>>) -> B::FloatTensorPrimitive {
        // arg(a + bi) = atan2(b, a)
        tensor.imag.atan2(tensor.real)
    }

    fn complex_powc(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // z^w = exp(w * ln(z))
        let log_lhs = SplitBackend::<B, D>::complex_log(lhs);
        let product = SplitBackend::<B, D>::complex_mul(rhs, log_lhs);
        SplitBackend::<B, D>::complex_exp(product)
    }

    fn complex_sqrt(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // sqrt(z) = from_polar(sqrt(|z|), arg(z) / 2)
        let abs = SplitBackend::<B, D>::abs(tensor.clone());
        let sqrt_abs = B::float_sqrt(abs);
        let arg = tensor.imag.atan2(tensor.real);
        let half_arg = B::float_div_scalar(arg, burn_std::Scalar::Float(2.0));
        SplitBackend::<B, D>::complex_from_polar(sqrt_abs, half_arg)
    }

    fn complex_sin(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        // sin(a + bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
        SplitComplexTensor::new(
            tensor.real.sin() * tensor.imag.cosh(),
            (tensor.real.cos() * tensor.imag.sinh()),
        )
    }

    fn complex_cos(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        // cos(a + bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
        SplitComplexTensor::new(
            tensor.real.cos() * tensor.imag.cosh(),
            -(tensor.real.sin() * tensor.imag.sinh()),
        )
    }

    fn complex_tan(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        // tan(z) = sin(z) / cos(z)
        // sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
        // cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
        // Compute sin(a), cos(a), sinh(b), cosh(b) once and share between numerator/denominator.
        let sin_a = tensor.real.sin();
        let cos_a = tensor.real.cos();
        let sinh_b = tensor.imag.sinh();
        let cosh_b = tensor.imag.cosh();
        let sin_z = SplitComplexTensor::new(
            (sin_a.clone() * cosh_b.clone()),
            (cos_a.clone() * sinh_b.clone()),
        );
        let cos_z = SplitComplexTensor::new((cos_a * cosh_b), -(sin_a * sinh_b));
        SplitBackend::<B, D>::complex_div(sin_z, cos_z)
    }

    fn complex_acos(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // acos(z) = -i * ln(z + i * sqrt(1 - z²))
        let device = tensor.device();
        let shape = tensor.shape();
        let fdtype = tensor.inner_dtype().into();
        let ones = SplitComplexTensor::new(
            DispatchTensor::ones(shape.clone(), &device, fdtype),
            DispatchTensor::zeros(shape, &device, fdtype),
        );
        // 1 - z²
        let z_sq = SplitBackend::<B, D>::complex_mul(tensor.clone(), tensor.clone());
        let one_minus_z_sq = SplitBackend::<B, D>::complex_sub(ones, z_sq);
        // i * sqrt(1 - z²): multiply by i via (-imag, real)
        let sqrt_term = SplitBackend::<B, D>::complex_sqrt(one_minus_z_sq);
        let i_sqrt = SplitComplexTensor::new(-sqrt_term.imag, sqrt_term.real);
        // z + i*sqrt(1 - z²)
        let inner = SplitBackend::<B, D>::complex_add(tensor, i_sqrt);
        // -i * ln(inner): multiply by -i via (imag, -real)
        let log_inner = SplitBackend::<B, D>::complex_log(inner);
        SplitComplexTensor::new(log_inner.imag, -log_inner.real)
    }

    fn complex_acosh(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // acosh(z) = ln(z + sqrt(z² - 1))
        let device = tensor.device();
        let shape = tensor.shape();
        let fdtype = tensor.inner_dtype().into();
        let ones = SplitComplexTensor::new(
            DispatchTensor::ones(shape.clone(), &device, fdtype),
            DispatchTensor::zeros(shape, &device, fdtype),
        );
        // z² - 1
        let z_sq = SplitBackend::<B, D>::complex_mul(tensor.clone(), tensor.clone());
        let z_sq_minus_one = SplitBackend::<B, D>::complex_sub(z_sq, ones);
        // z + sqrt(z² - 1)
        let sqrt_term = SplitBackend::<B, D>::complex_sqrt(z_sq_minus_one);
        let inner = SplitBackend::<B, D>::complex_add(tensor, sqrt_term);
        SplitBackend::<B, D>::complex_log(inner)
    }

    fn complex_asin(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // asin(z) = -i * ln(i*z + sqrt(1 - z²))
        let device = tensor.device();
        let shape = tensor.shape();
        let fdtype = tensor.inner_dtype().into();
        let ones = SplitComplexTensor::new(
            DispatchTensor::ones(shape.clone(), &device, fdtype),
            DispatchTensor::zeros(shape, &device, fdtype),
        );
        // z² and i*z before consuming tensor
        let z_sq = SplitBackend::<B, D>::complex_mul(tensor.clone(), tensor.clone());
        // i*z = (-imag, real)
        let i_z = SplitComplexTensor::new(DispatchTensor::neg(tensor.imag), tensor.real);
        // 1 - z²
        let one_minus_z_sq = SplitBackend::<B, D>::complex_sub(ones, z_sq);
        // i*z + sqrt(1 - z²)
        let sqrt_term = SplitBackend::<B, D>::complex_sqrt(one_minus_z_sq);
        let inner = SplitBackend::<B, D>::complex_add(i_z, sqrt_term);
        // -i * ln(inner): (imag, -real)
        let log_inner = SplitBackend::<B, D>::complex_log(inner);
        SplitComplexTensor::new(log_inner.imag, DispatchTensor::neg(log_inner.real))
    }

    fn complex_asinh(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // asinh(z) = ln(z + sqrt(z² + 1))
        let device = tensor.device();
        let shape = tensor.shape();
        let fdtype = tensor.inner_dtype().into();
        let ones = SplitComplexTensor::new(
            DispatchTensor::ones(shape.clone(), &device, fdtype),
            DispatchTensor::zeros(shape, &device, fdtype),
        );
        // z² + 1
        let z_sq = SplitBackend::<B, D>::complex_mul(tensor.clone(), tensor.clone());
        let z_sq_plus_one = SplitBackend::<B, D>::complex_add(z_sq, ones);
        // z + sqrt(z² + 1)
        let sqrt_term = SplitBackend::<B, D>::complex_sqrt(z_sq_plus_one);
        let inner = SplitBackend::<B, D>::complex_add(tensor, sqrt_term);
        SplitBackend::<B, D>::complex_log(inner)
    }

    fn complex_atan(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // atan(z) = (-i/2) * ln((1 + i*z) / (1 - i*z))
        let device = tensor.device();
        let shape = tensor.shape();
        let fdtype = tensor.inner_dtype().into();
        let ones = SplitComplexTensor::new(
            DispatchTensor::ones(shape.clone(), &device, fdtype),
            DispatchTensor::zeros(shape, &device, fdtype),
        );
        // i*z = (-imag, real)
        let i_z = SplitComplexTensor::new(DispatchTensor::neg(tensor.imag), tensor.real);
        // 1 + i*z and 1 - i*z
        let one_plus_i_z = SplitBackend::<B, D>::complex_add(ones.clone(), i_z.clone());
        let one_minus_i_z = SplitBackend::<B, D>::complex_sub(ones, i_z);
        // ln((1 + i*z) / (1 - i*z))
        let log_ratio = SplitBackend::<B, D>::complex_log(SplitBackend::<B, D>::complex_div(
            one_plus_i_z,
            one_minus_i_z,
        ));
        // (-i/2) * log_ratio: -i*(a+bi) = (b, -a), then /2
        SplitComplexTensor::new(
            DispatchTensor::div_scalar(log_ratio.imag, burn_std::Scalar::Float(2.0)),
            DispatchTensor::neg(DispatchTensor::div_scalar(
                log_ratio.real,
                burn_std::Scalar::Float(2.0),
            )),
        )
    }

    fn complex_atanh(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // atanh(z) = (1/2) * ln((1 + z) / (1 - z))
        let device = tensor.device();
        let shape = tensor.shape();
        let fdtype = tensor.inner_dtype().into();
        let ones = SplitComplexTensor::new(
            DispatchTensor::ones(shape.clone(), &device, fdtype),
            DispatchTensor::zeros(shape, &device, fdtype),
        );
        let one_plus_z = ones + tensor;
        let one_minus_z = ones - tensor;
        let log_ratio = (one_plus_z / one_minus_z).log();
        SplitComplexTensor::new(
            log_ratio.real / burn_std::Scalar::Float(2.0),
            log_ratio.imag / burn_std::Scalar::Float(2.0),
        )
    }

    fn complex_slice(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        slices: &[burn_std::Slice],
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(tensor.real.slice(slices), tensor.imag.slice(slices))
    }

    fn complex_slice_assign(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        ranges: &[burn_std::Slice],
        value: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            tensor.real.slice_assign(ranges, value.real),
            tensor.imag.slice_assign(ranges, value.imag),
        )
    }

    fn complex_swap_dims(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim1: usize,
        dim2: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            tensor.real.swap_dims(dim1, dim2),
            tensor.imag.swap_dims(dim1, dim2),
        )
    }

    fn complex_repeat_dim(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
        times: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            tensor.real.repeat_dim(dim, times),
            tensor.imag.repeat_dim(dim, times),
        )
    }

    fn complex_cat(
        tensors: Vec<ComplexTensor<SplitBackend<B, D>>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        let (reals, imags): (Vec<_>, Vec<_>) =
            tensors.into_iter().map(|t| (t.real, t.imag)).unzip();
        SplitComplexTensor::new(reals.cat(dim), imags.cat(dim))
    }

    fn complex_any(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let real_any = tensor.real.any(out_dtype);
        let imag_any = tensor.imag.any(out_dtype);
        B::bool_or(real_any, imag_any)
    }

    fn complex_any_dim(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let real_any = tensor.real.any_dim(dim, out_dtype);
        let imag_any = tensor.imag.any_dim(dim, out_dtype);
        B::bool_or(real_any, imag_any)
    }

    fn complex_all(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        // A complex element is nonzero if either real or imag is nonzero.
        // all(z != 0) = all(real != 0 || imag != 0)
        let real_nonzero = tensor
            .real
            .not_equal_elem(burn_std::Scalar::Float(0.0), out_dtype);
        let imag_nonzero = tensor
            .imag
            .not_equal_elem(burn_std::Scalar::Float(0.0), out_dtype);
        let elem_nonzero = B::bool_or(real_nonzero, imag_nonzero);
        B::bool_all(elem_nonzero)
    }

    fn complex_all_dim(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let real_nonzero = tensor
            .real
            .not_equal_elem(burn_std::Scalar::Float(0.0), out_dtype);
        let imag_nonzero = tensor
            .imag
            .not_equal_elem(burn_std::Scalar::Float(0.0), out_dtype);
        let elem_nonzero = B::bool_or(real_nonzero, imag_nonzero);
        B::bool_all_dim(elem_nonzero, dim)
    }

    fn complex_permute(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        axes: &[usize],
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(tensor.real.permute(axes), tensor.imag.permute(axes))
    }

    fn complex_expand(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        shape: burn_std::Shape,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(tensor.real.expand(shape.clone()), tensor.imag.expand(shape))
    }

    fn complex_flip(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        axes: &[usize],
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(tensor.real.flip(axes), tensor.imag.flip(axes))
    }

    fn complex_unfold(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            tensor.real.unfold(dim, size, step),
            tensor.imag.unfold(dim, size, step),
        )
    }

    fn complex_select(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
        indices: B::IntTensorPrimitive,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            tensor.real.select(dim, indices.clone()),
            tensor.imag.select(dim, indices),
        )
    }

    fn complex_sum(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(tensor.real.sum(), tensor.imag.sum())
    }

    fn complex_sum_dim(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(tensor.real.sum_dim(dim), tensor.imag.sum_dim(dim))
    }

    fn complex_prod(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // prod(z) = exp(sum(log(z)))
        let log_tensor = SplitBackend::<B, D>::complex_log(tensor);
        let sum_log = SplitBackend::<B, D>::complex_sum(log_tensor);
        SplitBackend::<B, D>::complex_exp(sum_log)
    }

    fn complex_prod_dim(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // prod_dim(z, dim) = exp(sum_dim(log(z), dim))
        let log_tensor = SplitBackend::<B, D>::complex_log(tensor);
        let sum_log = SplitBackend::<B, D>::complex_sum_dim(log_tensor, dim);
        SplitBackend::<B, D>::complex_exp(sum_log)
    }

    fn complex_mean(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(tensor.real.mean(), tensor.imag.mean())
    }

    fn complex_mean_dim(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(tensor.real.mean_dim(dim), tensor.imag.mean_dim(dim))
    }

    fn complex_remainder(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // Componentwise remainder (matching Complex<E> Rem impl)
        SplitComplexTensor::new(lhs.real.remainder(rhs.real), lhs.imag.remainder(rhs.imag))
    }

    fn complex_remainder_scalar(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: Scalar,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        use burn_std::cast::ToElement;
        SplitComplexTensor::new(
            lhs.real
                .remainder_scalar(burn_std::Scalar::Float(rhs.real().to_f64())),
            lhs.imag
                .remainder_scalar(burn_std::Scalar::Float(rhs.imag().to_f64())),
        )
    }

    fn complex_mask_where(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        mask: B::BoolTensorPrimitive,
        source: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            tensor.real.mask_where(mask.clone(), source.real),
            tensor.imag.mask_where(mask, source.imag),
        )
    }

    fn complex_mask_fill(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        mask: B::BoolTensorPrimitive,
        value: Scalar,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        use burn_std::cast::ToElement;
        let (real, imag) = match value {
            Scalar::Complex(c) => (c.real().to_f64(), c.imag().to_f64()),
            _ => (value.elem::<f64>(), 0.0),
        };
        SplitComplexTensor::new(
            tensor
                .real
                .mask_fill(mask.clone(), burn_std::Scalar::Float(real)),
            tensor.imag.mask_fill(mask, burn_std::Scalar::Float(imag)),
        )
    }

    fn complex_sign(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // sign(z) = z / |z| = from_polar(1, arg(z))
        let abs = SplitBackend::<B, D>::abs(tensor.clone());
        SplitComplexTensor::new(tensor.real.div(abs.clone()), tensor.imag.div(abs))
    }

    fn complex_matmul(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // (A + iB)(C + iD) = (AC - BD) + i(AD + BC)
        SplitComplexTensor::new(
            (lhs.real.matmul(rhs.real)) - (lhs.imag.matmul(rhs.imag)),
            (lhs.real.matmul(rhs.imag)) + (lhs.imag.matmul(rhs.real)),
        )
    }

    fn complex_cumsum(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // cumsum is linear, so it works componentwise
        SplitComplexTensor::new(tensor.real.cumsum(dim), tensor.imag.cumsum(dim))
    }

    fn complex_cumprod(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // cumprod(z, dim) = exp(cumsum(log(z), dim))
        let log_tensor = SplitBackend::<B, D>::complex_log(tensor);
        let cumsum_log = SplitBackend::<B, D>::complex_cumsum(log_tensor, dim);
        SplitBackend::<B, D>::complex_exp(cumsum_log)
    }

    fn complex_select_add(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
        indices: B::IntTensorPrimitive,
        values: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            tensor.real.select_add(dim, indices.clone(), values.real),
            tensor.imag.select_add(dim, indices, values.imag),
        )
    }

    fn complex_powc_scalar(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: Scalar,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // z^c = exp(c * ln(z)) where c = a + bi is a scalar
        // (a + bi) * (u + vi) = (au - bv) + (av + bu)i
        let a = burn_std::Scalar::Float(rhs.real().to_f64());
        let b = burn_std::Scalar::Float(rhs.imag().to_f64());
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
        // z^w = exp(w * ln(z)) where w is a real tensor
        let log_z = SplitBackend::<B, D>::complex_log(lhs);
        let w_log_z = SplitComplexTensor::new((rhs.clone() * log_z.real), (rhs * log_z.imag));
        SplitBackend::<B, D>::complex_exp(w_log_z)
    }

    fn complex_powf_scalar(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: Scalar,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // z^w = exp(w * ln(z)) where w is a real scalar
        let log_z = lhs.log();
        let w_log_z = SplitComplexTensor::new(
            log_z.real * burn_std::Scalar::Float(rhs.real().to_f64()),
            log_z.imag * burn_std::Scalar::Float(rhs.imag().to_f64()),
        );
        SplitBackend::<B, D>::complex_exp(w_log_z)
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
    fn zeros(shape: Shape, device: &Device<B>, _dtype: ComplexDType) -> ComplexTensor<B> {
        let real = B::InnerBackend::float_from_data(
            TensorData::zeros::<<B::InnerBackend as BackendTypes>::FloatElem, _>(&shape),
            device,
        );
        let imag = B::InnerBackend::float_from_data(
            TensorData::zeros::<<B::InnerBackend as BackendTypes>::FloatElem, _>(shape),
            device,
        );
        // ComplexTensor<B> = Complex<T> via SplitLayout
        <B as BackendTypes>::ComplexTensor::new(real, imag)
    }

    fn ones(shape: Shape, device: &Device<B>, _dtype: ComplexDType) -> ComplexTensor<B> {
        let real = TensorData::ones::<<B::InnerBackend as BackendTypes>::FloatElem, _>(&shape);
        let imag = TensorData::zeros::<<B::InnerBackend as BackendTypes>::FloatElem, _>(shape);
        SplitComplexTensor::<B, D>::from_parts_data(
            TensorData::ones(&shape),
            TensorData::zeros(&shape),
        )
    }

    fn full(
        shape: Shape,
        fill_value: B::ComplexScalar,
        device: &Device<B>,
    ) -> SplitComplexTensor<B, D> {
        SplitComplexTensor::<B, D>::from_parts_data(
            TensorData::full(&shape, fill_value.real()),
            TensorData::full(&shape, fill_value.imag()),
        )
    }

    async fn complex_into_data(
        tensor: ComplexTensor<B>,
    ) -> Result<Self::OutTensorData, ExecutionError> {
        B::complex_into_split_data(tensor).await
    }
}

impl<B, const D: usize> SplitComplexTensor<B, D>
where
    B: Backend,
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
        let dtype = opt.resolve_dtype::<Float>();
        //check!(TensorCheck::creation_ops::<D>("Empty", &shape));
        SplitBackend::<B, D>::complex_zeros(shape, &opt.device.into(), dtype.into())
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
            d
        }
        SplitBackend::<B, D>::complex_slice(self, &slices)
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
        let dtype = crate::utils::real_to_complex_dtype(options.resolve_dtype::<Float>());
        match dtype {
            DType::Complex32 | DType::Complex64 => {
                SplitBackend::<B, D>::complex_zeros(shape, device.into(), dtype.into())
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
        let device = &options.device;
        let dtype = crate::utils::real_to_complex_dtype(options.resolve_dtype::<Float>());
        match dtype {
            DType::Complex32 | DType::Complex64 => {
                SplitBackend::<B, D>::complex_ones(shape, device.into(), dtype.into())
            }
            _ => panic!("Unsupported complex dtype"),
        }
    }
}
//BasicOps
impl<B, const D: usize> SplitComplexTensor<B, D>
where
    B: Backend,
{
    /// Select complex tensor elements along the given dimension corresponding to the given indices.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to select from.
    /// * `indices` - The indices of the elements to select.
    pub fn select(self, dim: usize, indices: Tensor<1, Int>) -> Self {
        // Uses your existing `select` name.
        SplitBackend::<B, D>::complex_select(self, dim, indices.tensor())
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
                SplitBackend::<B, D>::complex_select_add(self, dim, indices.tensor(), values)
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
    ) -> SplitComplexTensor<B, D2> {
        // Convert reshape args to shape
        let shape = shape.into_shape::<D2>(self.shape());
        let (real, imag) = (self.real, self.imag);
        SplitComplexTensor::new(real.reshape(shape.clone()), imag.reshape(shape))
    }

    /// Transpose the complex tensor.
    ///
    /// For a 2D tensor, this is the standard matrix transpose. For `D > 2`, the transpose is
    /// applied on the last two dimensions.
    pub fn transpose(self) -> Self {
        SplitBackend::<B, D>::complex_transpose(self)
    }

    /// Swaps two dimensions of a complex tensor.
    ///
    /// # Arguments
    ///
    /// * `dim1` - The first dimension to swap.
    /// * `dim2` - The second dimension to swap.
    pub fn swap_dims(self, dim1: usize, dim2: usize) -> Self {
        SplitBackend::<B, D>::complex_swap_dims(self, dim1, dim2)
    }

    /// Returns the device of the current complex tensor.
    pub fn device(&self) -> B::Device {
        SplitBackend::<B, D>::complex_device(self)
    }

    /// Move the complex tensor to the given device.
    pub fn to_device(self, device: &B::Device) -> Self {
        SplitBackend::<B, D>::complex_to_device(self, device)
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
        SplitBackend::<B, D>::complex_into_interleaved_data(self).await
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
        SplitBackend::<B, D>::complex_from_interleaved_data(
            data.convert::<Scalar>(),
            &opt.device.into(),
        )
    }

    /// Repeat the complex tensor along the given dimension.
    ///
    /// # Arguments
    /// - `dim`: The dimension to repeat.
    /// - `times`: The number of times to repeat the tensor along the given dimension.
    pub fn repeat_dim(self, dim: usize, times: usize) -> Self {
        SplitBackend::<B, D>::complex_repeat_dim(self, dim, times)
    }
    /// Applies element-wise equal comparison.
    ///
    /// # Returns
    ///
    /// A boolean tensor that is `true` where the two complex elements are equal and `false` elsewhere.
    pub fn equal(self, rhs: Self) -> B::BoolTensorPrimitive {
        let out_dtype = self.real.dtype();
        SplitBackend::<B, D>::complex_equal(self, rhs, out_dtype.into())
    }

    /// Applies element-wise non-equality comparison.
    ///
    /// # Returns
    ///
    /// A boolean tensor that is `true` where the two complex elements are not equal and `false` elsewhere.
    pub fn not_equal(self, rhs: Self) -> B::BoolTensorPrimitive {
        let out_dtype = self.real.dtype();
        SplitBackend::<B, D>::complex_not_equal(self, rhs, out_dtype.into())
    }

    /// Concatenates all complex tensors into a new one along the given dimension.
    ///
    /// # Panics
    ///
    /// - If `dim` is higher than the rank.
    /// - If `tensors` is an empty vector.
    /// - If all tensors don't have the same shape (the dimension `dim` is ignored).
    pub fn cat(tensors: Vec<Self>, dim: usize) -> Self {
        SplitBackend::<B, D>::complex_cat(tensors, dim)
    }

    /// Tests if any element in the complex tensor evaluates to non-zero (i.e., true).
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element, `true` if any element is non-zero, `false` otherwise.
    pub fn any(self) -> B::BoolTensorPrimitive {
        let out_dtype = self.device().bool_dtype;
        SplitBackend::<B, D>::complex_any(self, out_dtype)
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
    pub fn any_dim(self, dim: usize) -> B::BoolTensorPrimitive {
        let out_dtype = self.device().bool_dtype;
        SplitBackend::<B, D>::complex_any_dim(self, dim, out_dtype)
    }

    /// Tests if all elements in the complex tensor evaluate to non-zero (i.e., true).
    ///
    /// # Returns
    ///
    /// A boolean tensor with a single element, `true` if all elements are non-zero, `false` otherwise.
    pub fn all(self) -> B::BoolTensorPrimitive {
        let out_dtype = self.device().bool_dtype;
        SplitBackend::<B, D>::complex_all(self, out_dtype)
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
    pub fn all_dim(self, dim: usize) -> B::BoolTensorPrimitive {
        let out_dtype = self.device().bool_dtype;
        SplitBackend::<B, D>::complex_all_dim(self, dim, out_dtype)
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
            SplitBackend::<B, D>::complex_permute(self, &fixed_axes)
        }
    }

    // pub fn expand(self, shape: Shape) -> Self {
    //     SplitBackend::<B, D>::complex_expand(self, shape)
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
    ) -> SplitComplexTensor<B, D2> {
        let shape = shape.into_shape(&self.shape());
        // check!(TensorCheck::expand::<D, D2>(
        //     "expand",
        //     &self.shape(),
        //     &shape,
        // ));
        let (real, imag) = (self.real, self.imag);
        SplitComplexTensor::<B, D2>::new(real.expand(shape.clone()), imag.expand(shape))
    }

    // pub fn flip(self, axes: &[usize]) -> Self {
    //     SplitBackend::<B, D>::complex_flip(self, axes)
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

        SplitComplexTensor::<B, D>::new(real.flip(&transformed_axes), imag.flip(&transformed_axes))
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
    ) -> SplitComplexTensor<B, D2> {
        let dim = dim.expect_dim_index(D);
        let (real, imag) = (self.real, self.imag);
        SplitComplexTensor::new(real.unfold(dim, size, step), imag.unfold(dim, size, step))
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

        let values_shape = SplitBackend::<B, D>::complex_shape(&values);
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

        SplitBackend::<B, D>::complex_slice_assign(self, &slices, values)
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
        SplitBackend::<B, D>::complex_mask_where(self, mask.tensor(), source)
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
        SplitBackend::<B, D>::complex_mask_fill(self, mask.tensor(), value.elem())
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
        SplitBackend::<B, D>::complex_gather(dim, self, indices.tensor())
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
                SplitBackend::<B, D>::complex_scatter_add(dim, self, indices.tensor(), values)
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
        let out_dtype = self.device().bool_dtype;
        Tensor::<B, D, Bool>::new(SplitBackend::<B, D>::complex_equal_elem(
            self,
            other.elem(),
            out_dtype,
        ))
    }

    /// Applies element-wise non-equality comparison with a scalar and returns a boolean tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The element to compare each complex element against.
    pub fn not_equal_elem<E: Element>(self, other: E) -> Tensor<D, Bool> {
        let out_dtype = self.device().bool_dtype;
        Tensor::<B, D, Bool>::new(SplitBackend::<B, D>::complex_not_equal_elem(
            self,
            other.elem(),
            out_dtype,
        ))
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
        values: SplitComplexTensor<B, DV>,
        update: IndexingUpdateOp,
    ) -> Self {
        // check!(TensorCheck::scatter_nd::<D, M, DV>(
        //     &self.shape(),
        //     &indices.shape(),
        //     &values.shape()
        // ));
        let indices = indices.tensor();
        let SplitComplexTensor::<B, D> { real, imag, .. } = self;
        let SplitComplexTensor::<B, DV> {
            real: real_values,
            imag: imag_values,
            ..
        } = values;
        SplitComplexTensor::new(
            real.scatter_nd(indices.clone(), real_values, update),
            imag.scatter_nd(indices, imag_values, update),
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
    ) -> SplitComplexTensor<B, DV> {
        let indices = indices.tensor();
        let SplitComplexTensor::<B, D> { real, imag, .. } = self;
        //check!(TensorCheck::gather_nd::<D, M, DV>(&indices.shape()));
        SplitComplexTensor::new(real.gather_nd(indices.clone()), imag.gather_nd(indices))
    }
}
//impl<B, F> Numeric<B> for SplitComplexTensor<F>
impl<B: Backend, const D: usize, F> SplitComplexTensor<B, D, F>
where
    B: BackendTypes<FloatTensorPrimitive = F>,
    F: TensorMetadata + 'static,
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
        SplitBackend::<B, D>::complex_add(self, rhs)
    }

    /// Applies element-wise addition operation with a scalar.
    ///
    /// `y = x + s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to add, element-wise.
    pub fn add_scalar(self, rhs: burn_std::Scalar) -> Self {
        let device = SplitBackend::<B, D>::complex_device(&self);
        let shape = SplitBackend::<B, D>::complex_shape(&self);
        let scalar_complex = rhs.elem();
        let scalar_tensor = SplitBackend::<B, D>::complex_full(shape, scalar_complex, &device);
        SplitBackend::<B, D>::complex_add(self, scalar_tensor)
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
        SplitBackend::<B, D>::complex_sub(self, rhs)
    }

    /// Applies element-wise subtraction operation with a scalar.
    ///
    /// `y = x - s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to subtract, element-wise.
    pub fn sub_scalar(self, rhs: burn_std::Scalar) -> Self {
        let device = SplitBackend::<B, D>::complex_device(&self);
        let shape = SplitBackend::<B, D>::complex_shape(&self);
        let scalar_complex = rhs.elem();
        let scalar_tensor = SplitBackend::<B, D>::complex_full(shape, scalar_complex, &device);
        SplitBackend::<B, D>::complex_sub(self, scalar_tensor)
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
        SplitBackend::<B, D>::complex_div(self, rhs)
    }

    /// Applies element-wise division operation with a scalar.
    ///
    /// `y = x / s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to divide by, element-wise.
    pub fn div_scalar(self, rhs: burn_std::Scalar) -> Self {
        let device = SplitBackend::<B, D>::complex_device(&self);
        let shape = SplitBackend::<B, D>::complex_shape(&self);
        let scalar_complex = rhs.elem();
        let scalar_tensor = SplitBackend::<B, D>::complex_full(shape, scalar_complex, &device);
        SplitBackend::<B, D>::complex_div(self, scalar_tensor)
    }

    /// Applies element-wise the remainder operation.
    ///
    /// `y = x2 % x1`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex tensor to compute the remainder with.
    pub fn remainder(self, rhs: Self) -> Self {
        SplitBackend::<B, D>::complex_remainder(self, rhs)
    }

    /// Applies element-wise the remainder operation with a scalar.
    ///
    /// `y = x % s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to compute the remainder with, element-wise.
    pub fn remainder_scalar(self, rhs: burn_std::Scalar) -> Self {
        SplitBackend::<B, D>::complex_remainder_scalar(self, rhs.elem())
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
        SplitBackend::<B, D>::complex_mul(self, rhs)
    }

    /// Applies element-wise multiplication operation with a scalar.
    ///
    /// `y = x * s`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The complex scalar to multiply, element-wise.
    pub fn mul_scalar(self, rhs: burn_std::Scalar) -> Self {
        let device = SplitBackend::<B, D>::complex_device(&self);
        let shape = SplitBackend::<B, D>::complex_shape(&self);
        let scalar_complex = rhs.elem();
        let scalar_tensor = SplitBackend::<B, D>::complex_full(shape, scalar_complex, &device);
        SplitBackend::<B, D>::complex_mul(self, scalar_tensor)
    }

    /// Switch sign of each element in the complex tensor.
    ///
    /// `y = -x`
    #[allow(clippy::should_implement_trait)]
    pub fn neg(self) -> Self {
        SplitBackend::<B, D>::complex_neg(self)
    }

    /// Returns the signs of the elements of the complex tensor.
    ///
    /// For a non-zero element `z`, returns `z / |z|`. For zero, returns zero.
    pub fn sign(self) -> Self {
        SplitBackend::<B, D>::complex_sign(self)
    }

    /// Aggregate all elements in the complex tensor with the sum operation.
    pub fn sum(self) -> Self {
        SplitBackend::<B, D>::complex_sum(self)
    }

    /// Aggregate all elements along the given dimension in the complex tensor with the sum operation.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements.
    pub fn sum_dim(self, dim: usize) -> Self {
        SplitBackend::<B, D>::complex_sum_dim(self, dim)
    }

    /// Aggregate all elements in the complex tensor with the product operation.
    pub fn prod(self) -> Self {
        SplitBackend::<B, D>::complex_prod(self)
    }

    /// Aggregate all elements along the given dimension in the complex tensor with the product operation.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements.
    pub fn prod_dim(self, dim: usize) -> Self {
        SplitBackend::<B, D>::complex_prod_dim(self, dim)
    }

    /// Aggregate all elements in the complex tensor with the mean operation.
    pub fn mean(self) -> Self {
        SplitBackend::<B, D>::complex_mean(self)
    }

    /// Aggregate all elements along the given dimension in the complex tensor with the mean operation.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to aggregate the elements.
    pub fn mean_dim(self, dim: usize) -> Self {
        SplitBackend::<B, D>::complex_mean_dim(self, dim)
    }

    /// Computes the cumulative sum of complex elements along the given dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to compute the cumulative sum.
    pub fn cumsum(self, dim: usize) -> Self {
        SplitBackend::<B, D>::complex_cumsum(self, dim)
    }

    /// Computes the cumulative product of complex elements along the given dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension or axis along which to compute the cumulative product.
    pub fn cumprod(self, dim: usize) -> Self {
        SplitBackend::<B, D>::complex_cumprod(self, dim)
    }

    /// Applies element-wise power operation with an integer tensor.
    ///
    /// # Arguments
    ///
    /// * `other` - The integer tensor to apply the power operation with.
    pub fn powi(self, other: Tensor<D, Int>) -> Self {
        SplitBackend::<B, D>::complex_powi(self, other.tensor())
    }

    /// Applies element-wise power operation with an integer scalar.
    ///
    /// # Arguments
    ///
    /// * `other` - The scalar to apply the power operation with.
    pub fn powi_scalar<E: ElementConversion>(self, other: E) -> Self {
        let other = Scalar::new(other, &self.dtype());
        SplitBackend::<B, D>::complex_powi_scalar(self, other)
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
        let dtype = opt.resolve_dtype::<Float>();
        SplitBackend::<B, D>::complex_random(
            shape.into(),
            distribution,
            &opt.device.into(),
            dtype.into(),
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
        SplitBackend::<B, D>::complex_matmul(self, rhs)
    }
}

// ComplexOnlyOps
impl<B, const D: usize, F> SplitComplexTensor<B, D, F>
where
    B: Backend,
    B: BackendTypes<FloatTensorPrimitive = F>,
    F: TensorMetadata + 'static,
{
    /// Returns the complex conjugate of each element.
    ///
    /// For a complex number `a + bi`, the conjugate is `a - bi`.
    pub fn conj(self) -> Self {
        SplitBackend::<B, D>::conj(self)
    }

    /// Returns the argument (phase angle) of each element, in radians.
    ///
    /// For a complex number `a + bi`, the phase is `atan2(b, a)`, ranging from `-π` to `π`.
    pub fn phase(self) -> F {
        SplitBackend::<B, D>::complex_arg(self)
    }

    /// Returns the magnitude (absolute value, modulus) of each element.
    ///
    /// For a complex number `a + bi`, the magnitude is `sqrt(a² + b²)`.
    pub fn magnitude(self) -> F {
        SplitBackend::<B, D>::abs(self)
    }

    /// Applies element-wise complex exponential.
    ///
    /// For a complex number `a + bi`, computes `exp(a) * (cos(b) + i·sin(b))`.
    pub fn exp(self) -> Self {
        SplitBackend::<B, D>::complex_exp(self)
    }

    /// Applies element-wise complex sine.
    pub fn sin(self) -> Self {
        SplitBackend::<B, D>::complex_sin(self)
    }

    /// Create a complex tensor from separate real and imaginary data.
    ///
    /// # Arguments
    ///
    /// * `real` - The real part data.
    /// * `imag` - The imaginary part data.
    pub fn from_parts<T: Into<TensorData>>(real: T, imag: T) -> Self {
        SplitBackend::<B, D>::complex_from_parts(real.into(), imag.into())
    }

    /// Create a complex tensor from interleaved (real, imaginary) data.
    ///
    /// The input data should contain alternating real and imaginary values.
    ///
    /// # Arguments
    ///
    /// * `data` - Interleaved complex data.
    /// * `device` - The device to create the tensor on.
    pub fn from_interleaved_data(data: TensorData, device: &B::Device) -> Self {
        SplitBackend::<B, D>::complex_from_interleaved_data(data, device)
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
    pub fn from_polar(magnitude: F, phase: F) -> Self {
        SplitBackend::<B, D>::complex_from_polar(magnitude, phase)
    }

    /// Applies element-wise complex cosine.
    pub fn cos(self) -> Self {
        SplitBackend::<B, D>::complex_cos(self)
    }

    /// Applies element-wise complex tangent.
    pub fn tan(self) -> Self {
        SplitBackend::<B, D>::complex_tan(self)
    }

    /// Applies element-wise complex arccosine.
    pub fn acos(self) -> Self {
        SplitBackend::<B, D>::complex_acos(self)
    }

    /// Applies element-wise complex hyperbolic arccosine.
    pub fn acosh(self) -> Self {
        SplitBackend::<B, D>::complex_acosh(self)
    }

    /// Applies element-wise complex arcsine.
    pub fn asin(self) -> Self {
        SplitBackend::<B, D>::complex_asin(self)
    }

    /// Applies element-wise complex hyperbolic arcsine.
    pub fn asinh(self) -> Self {
        SplitBackend::<B, D>::complex_asinh(self)
    }

    /// Applies element-wise complex arctangent.
    pub fn atan(self) -> Self {
        SplitBackend::<B, D>::complex_atan(self)
    }

    /// Applies element-wise complex hyperbolic arctangent.
    pub fn atanh(self) -> Self {
        SplitBackend::<B, D>::complex_atanh(self)
    }

    /// Applies element-wise complex natural logarithm.
    ///
    /// For a complex number `z = r · exp(i · θ)`, computes `ln(r) + i · θ`.
    pub fn log(self) -> Self {
        SplitBackend::<B, D>::complex_log(self)
    }
    /// Applies element-wise complex square root.
    pub fn sqrt(self) -> Self {
        SplitBackend::<B, D>::complex_sqrt(self)
    }
}

use crate::split::SplitComplexTensor;

// SplitComplexTensor + SplitComplexTensor
impl<B: Backend, const D: usize> core::ops::Add<Self> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::add(self, rhs)
    }
}

// SplitComplexTensor + Tensor<D, Float> — adds real tensor to the real part
impl<B: Backend, const D: usize> core::ops::Add<Tensor<D>> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn add(self, rhs: Tensor<D>) -> Self::Output {
        let prim = rhs.tensor();
        let (real, imag) = self.into_parts();
        SplitComplexTensor::new(real + prim, imag)
    }
}

// SplitComplexTensor + scalar (concrete types to avoid coherence conflict with ElementConversion)
macro_rules! impl_complex_tensor_add_scalar {
    ($($t:ty),*) => {
        $(
            impl<B: Backend, const D: usize> core::ops::Add<$t> for SplitComplexTensor<B, D> {
                type Output = Self;

                fn add(self, rhs: $t) -> Self::Output {
                    Self::add_scalar(self, burn_std::Scalar::Float(rhs as f64))
                }
            }
        )*
    }
}
impl_complex_tensor_add_scalar!(f32, f64, i32, i64, u32, u64);

impl<B: Backend, const D: usize, E: Element> core::ops::Add<Complex<E>>
    for SplitComplexTensor<B, D>
{
    type Output = Self;

    fn add(self, rhs: Complex<E>) -> Self::Output {
        Self::add_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
    }
}
// Tensor - tensor
impl<B: Backend, const D: usize> core::ops::Sub<Self> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::sub(self, rhs)
    }
}

// SplitComplexTensor - Tensor<D, Float>
impl<B: Backend, const D: usize> core::ops::Sub<Tensor<D, Float>> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn sub(self, rhs: Tensor<D, Float>) -> Self::Output {
        let prim = rhs.tensor();

        let (real, imag) = self.into_parts();
        SplitComplexTensor::new(real - prim, imag)
    }
}

// SplitComplexTensor - scalar
macro_rules! impl_complex_tensor_sub_scalar {
    ($($t:ty),*) => {
        $(
            impl<B: Backend, const D: usize> core::ops::Sub<$t> for SplitComplexTensor<B, D> {
                type Output = Self;

                fn sub(self, rhs: $t) -> Self::Output {
                    Self::sub_scalar(self, burn_std::Scalar::Float(rhs as f64))
                }
            }
        )*
    }
}
impl_complex_tensor_sub_scalar!(f32, f64, i32, i64, u32, u64);

impl<B: Backend, const D: usize, E: Element> core::ops::Sub<Complex<E>>
    for SplitComplexTensor<B, D>
{
    type Output = Self;

    fn sub(self, rhs: Complex<E>) -> Self::Output {
        Self::sub_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
    }
}

// Tensor * tensor
impl<B: Backend, const D: usize> core::ops::Mul<Self> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::mul(self, rhs)
    }
}

// SplitComplexTensor * Tensor<D, Float>
impl<B: Backend, const D: usize> core::ops::Mul<Tensor<D, Float>> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn mul(self, rhs: Tensor<D, Float>) -> Self::Output {
        let prim = rhs.tensor();
        let (real, imag) = self.into_parts();
        SplitComplexTensor::new(real * prim, imag * prim)
    }
}

// SplitComplexTensor * scalar
macro_rules! impl_complex_tensor_mul_scalar {
    ($($t:ty),*) => {
        $(
            impl<B: Backend, const D: usize> core::ops::Mul<$t> for SplitComplexTensor<B, D> {
                type Output = Self;

                fn mul(self, rhs: $t) -> Self::Output {
                    Self::mul_scalar(self, burn_std::Scalar::Float(rhs as f64))
                }
            }
        )*
    }
}
impl_complex_tensor_mul_scalar!(f32, f64, i32, i64, u32, u64);

impl<B: Backend, const D: usize, E: Element> core::ops::Mul<Complex<E>>
    for SplitComplexTensor<B, D>
{
    type Output = Self;

    fn mul(self, rhs: Complex<E>) -> Self::Output {
        Self::mul_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
    }
}

// Tensor / tensor
impl<B: Backend, const D: usize> core::ops::Div<Self> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self::div(self, rhs)
    }
}

// SplitComplexTensor / Tensor<D, Float>
impl<B: Backend, const D: usize> core::ops::Div<Tensor<D, Float>> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn div(self, rhs: Tensor<D, Float>) -> Self::Output {
        let prim = rhs.tensor();
        let (real, imag) = self.into_parts();
        SplitComplexTensor::new(real / prim, imag / prim)
    }
}

// SplitComplexTensor / scalar
macro_rules! impl_complex_tensor_div_scalar {
    ($($t:ty),*) => {
        $(
            impl<B: Backend, const D: usize> core::ops::Div<$t> for SplitComplexTensor<B, D> {
                type Output = Self;

                fn div(self, rhs: $t) -> Self::Output {
                    Self::div_scalar(self, burn_std::Scalar::Float(rhs as f64))
                }
            }
        )*
    }
}
impl_complex_tensor_div_scalar!(f32, f64, i32, i64, u32, u64);

impl<B: Backend, const D: usize, E: Element> core::ops::Div<Complex<E>>
    for SplitComplexTensor<B, D>
{
    type Output = Self;

    fn div(self, rhs: Complex<E>) -> Self::Output {
        Self::div_scalar(self, Scalar::Complex(rhs.to_complex64()))
    }
}

// Tensor % tensor
impl<B: Backend, const D: usize> core::ops::Rem<Self> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self::remainder(self, rhs)
    }
}

// SplitComplexTensor % Tensor<D, Float>
impl<B: Backend, const D: usize> core::ops::Rem<Tensor<D, Float>> for SplitComplexTensor<B, D> {
    type Output = Self;

    fn rem(self, rhs: Tensor<D, Float>) -> Self::Output {
        let prim = rhs.tensor();
        let (real, imag) = self.into_parts();
        SplitComplexTensor::new(real % prim, imag % prim)
    }
}

// SplitComplexTensor % scalar
macro_rules! impl_complex_tensor_rem_scalar {
    ($($t:ty),*) => {
        $(
            impl<B: Backend, const D: usize> core::ops::Rem<$t> for SplitComplexTensor<B, D> {
                type Output = Self;

                fn rem(self, rhs: $t) -> Self::Output {
                    Self::remainder_scalar(self, burn_std::Scalar::Float(rhs as f64))
                }
            }
        )*
    }
}
impl_complex_tensor_rem_scalar!(f32, f64, i32, i64, u32, u64);

impl<B: Backend, const D: usize, E: Element> core::ops::Rem<Complex<E>>
    for SplitComplexTensor<B, D>
{
    type Output = Self;

    fn rem(self, rhs: Complex<E>) -> Self::Output {
        Self::remainder_scalar(self, burn_std::Scalar::Complex(rhs.to_complex64()))
    }
}

impl<B: Backend, const D: usize> core::ops::Neg for SplitComplexTensor<B, D> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::neg(self)
    }
}
