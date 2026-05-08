mod ops;
use crate::{
    base::{
        CBT, ComplexTensor, ComplexTensorBackend, ComplexTensorOps, DefaultComplexOps, Layout,
        SplitLayout, SplitTensorData,
    },
    utils::{self, real_to_complex_dtype, split_from_interleaved_data},
};

use alloc::vec::Vec;
use burn_std::{ComplexDType, FloatDType, Shape};
use burn_tensor::{
    Complex, ComplexElement, Device, ElementComparison, Scalar, TensorData, TensorMetadata,
    backend::{Backend, BackendTypes, ExecutionError},
    cast::ToElement,
    ops::FloatTensorOps,
};

use bytemuck::Pod;

impl Layout for SplitLayout {}

#[derive(Debug, Clone)]
pub struct SplitComplexTensor<
    B: Backend,
    const D: usize,
    T: TensorMetadata = <B as BackendTypes>::FloatTensorPrimitive,
> {
    _phantom: core::marker::PhantomData<B>,
    pub(crate) real: T,
    pub(crate) imag: T,
}

impl<B: Backend<FloatTensorPrimitive = T>, T: TensorMetadata, const D: usize>
    SplitComplexTensor<B, D, T>
{
    pub fn new(real: T, imag: T) -> Self {
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

    pub fn inner_dtype(&self) -> burn_std::DType {
        self.real.dtype()
    }

    pub fn from_real_data(data: TensorData, device: &B::Device) -> Self {
        let real = data.clone();

        Self {
            _phantom: core::marker::PhantomData,
            real: B::float_from_data(real, device),
            imag: B::float_zeros(data.shape, device, data.dtype.into()),
        }
    }

    pub fn from_imag_data(data: TensorData, device: &B::Device) -> Self {
        let imag = data.clone();

        Self {
            _phantom: core::marker::PhantomData,
            real: B::float_zeros(data.shape, device, data.dtype.into()),
            imag: B::float_from_data(imag, device),
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
            real: B::float_from_data(TensorData::from_bytes(real, shape.clone(), dtype), device),
            imag: B::float_from_data(TensorData::from_bytes(imag, shape, dtype), device),
        }
    }
    pub fn real(self) -> T {
        self.real
    }
    pub fn imag(self) -> T {
        self.imag
    }
    pub fn real_ref(&self) -> &T {
        &self.real
    }
    pub fn imag_ref(&self) -> &T {
        &self.imag
    }
}

impl<B: Backend<FloatTensorPrimitive = T>, T: TensorMetadata + 'static, const D: usize>
    TensorMetadata for SplitComplexTensor<B, D, T>
{
    fn shape(&self) -> burn_tensor::Shape {
        self.real.shape()
    }

    fn rank(&self) -> usize {
        self.shape().num_dims()
    }

    fn dtype(&self) -> burn_std::DType {
        utils::real_to_complex_dtype(self.inner_dtype())
    }
}

/// A newtype that wraps a real backend B and exposes a split-layout complex backend.
pub struct SplitBackend<B: Backend, const D: usize>(core::marker::PhantomData<B>);
impl<B: Backend, const D: usize> CBT for SplitBackend<B, D> {
    type ComplexTensorPrimitive = SplitComplexTensor<B, D, B::FloatTensorPrimitive>;
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

    fn dtype_usage(
        device: &Self::Device,
        dtype: burn_std::DType,
    ) -> burn_tensor::backend::DTypeUsageSet {
        B::dtype_usage(device, dtype)
    }

    fn device_count(type_id: u16) -> usize {
        B::device_count(type_id)
    }
    //type FloatTensorPrimitive = SplitComplexTensor<B::FloatTensorPrimitive>;
}

impl<B: Backend, const D: usize> ComplexTensorBackend for SplitBackend<B, D>
where
    B::FloatElem: ElementComparison + Pod,
    B::FloatTensorPrimitive: TensorMetadata + 'static,
{
    type InnerBackend = B;
    type ComplexScalar = Complex<B::FloatElem>;
    type Layout = SplitLayout; //<B::FloatTensorPrimitive>;

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
        Self::ComplexTensorPrimitive::from_split_data(split_from_interleaved_data(data), device)
    }

    fn complex_from_parts_data(
        real_data: TensorData,
        imag_data: TensorData,
        device: &Self::Device,
    ) -> ComplexTensor<Self> {
        let real = B::float_from_data(real_data, device);
        let imag = B::float_from_data(imag_data, device);
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
        ComplexTensor::<SplitBackend<B, D>>::new(tensor, B::float_zeros(shape, device, dtype))
    }

    fn real(tensor: ComplexTensor<SplitBackend<B, D>>) -> B::FloatTensorPrimitive {
        tensor.real
    }
    fn imag(tensor: ComplexTensor<SplitBackend<B, D>>) -> B::FloatTensorPrimitive {
        tensor.imag
    }

    fn complex_not_equal_elem(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: <SplitBackend<B, D> as ComplexTensorBackend>::ComplexScalar,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let (lhs_real, lhs_imag) = (lhs.real, lhs.imag);
        let rhs_real = rhs.real();
        let rhs_imag = rhs.imag();

        let real_cmp = B::float_not_equal_elem(
            lhs_real,
            burn_tensor::Scalar::Float(rhs_real.to_f64()),
            out_dtype,
        );
        let imag_cmp = B::float_not_equal_elem(
            lhs_imag,
            burn_tensor::Scalar::Float(rhs_imag.to_f64()),
            out_dtype,
        );
        B::bool_or(real_cmp, imag_cmp)
    }

    fn complex_equal_elem(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: <SplitBackend<B, D> as ComplexTensorBackend>::ComplexScalar,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let (lhs_real, lhs_imag) = (lhs.real, lhs.imag);
        let rhs_real = rhs.real();
        let rhs_imag = rhs.imag();

        let real_cmp = B::float_equal_elem(
            lhs_real,
            burn_tensor::Scalar::Float(rhs_real.to_f64()),
            out_dtype,
        );
        let imag_cmp = B::float_equal_elem(
            lhs_imag,
            burn_tensor::Scalar::Float(rhs_imag.to_f64()),
            out_dtype,
        );
        B::bool_and(real_cmp, imag_cmp)
    }

    fn complex_equal(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let real_cmp = B::float_equal(lhs.real, rhs.real, out_dtype);
        let imag_cmp = B::float_equal(lhs.imag, rhs.imag, out_dtype);
        B::bool_and(real_cmp, imag_cmp)
    }

    fn complex_not_equal(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let real_cmp = B::float_not_equal(lhs.real, rhs.real, out_dtype);
        let imag_cmp = B::float_not_equal(lhs.imag, rhs.imag, out_dtype);
        B::bool_or(real_cmp, imag_cmp)
    }

    async fn complex_into_real_data(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> Result<TensorData, burn_tensor::backend::ExecutionError> {
        B::float_into_data(tensor.real).await
    }

    async fn complex_into_imag_data(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> Result<TensorData, burn_tensor::backend::ExecutionError> {
        B::float_into_data(tensor.imag).await
    }

    async fn complex_into_interleaved_data(
        tensor: ComplexTensor<SplitBackend<B, D>>,
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
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> Result<SplitTensorData, burn_tensor::backend::ExecutionError> {
        let real_data = B::float_into_data(tensor.real).await?;
        let imag_data = B::float_into_data(tensor.imag).await?;
        Ok(SplitTensorData {
            real_bytes: real_data.bytes,
            imag_bytes: imag_data.bytes,
            shape: real_data.shape,
            dtype: real_data.dtype,
        })
    }

    fn complex_device(tensor: &ComplexTensor<SplitBackend<B, D>>) -> B::Device {
        B::float_device(&tensor.real)
    }

    fn complex_add(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        ComplexTensor::<SplitBackend<B, D>>::new(
            B::float_add(lhs.real, rhs.real),
            B::float_add(lhs.imag, rhs.imag),
        )
    }

    fn complex_sub(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        ComplexTensor::<SplitBackend<B, D>>::new(
            B::float_sub(lhs.real, rhs.real),
            B::float_sub(lhs.imag, rhs.imag),
        )
    }

    fn complex_mul(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        ComplexTensor::<SplitBackend<B, D>>::new(
            B::float_sub(
                B::float_mul(lhs.real.clone(), rhs.real.clone()),
                B::float_mul(lhs.imag.clone(), rhs.imag.clone()),
            ),
            B::float_add(
                B::float_mul(lhs.real, rhs.imag),
                B::float_mul(rhs.real, lhs.imag),
            ),
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
            B::float_div(
                B::float_add(
                    B::float_mul(lhs.real.clone(), rhs.real.clone()),
                    B::float_mul(lhs.imag.clone(), rhs.imag.clone()),
                ),
                norm_sqr.clone(),
            ),
            B::float_div(
                B::float_sub(
                    B::float_mul(lhs.imag.clone(), rhs.real.clone()),
                    B::float_mul(lhs.real.clone(), rhs.imag.clone()),
                ),
                norm_sqr.clone(),
            ),
        )
    }
    fn abs(tensor: ComplexTensor<SplitBackend<B, D>>) -> B::FloatTensorPrimitive {
        //todo! https://github.com/tracel-ai/burn/issues/4836
        // |z| = sqrt(real^2 + imag^2)
        let real_sq = B::float_mul(tensor.real.clone(), tensor.real.clone());
        let imag_sq = B::float_mul(tensor.imag.clone(), tensor.imag.clone());
        let norm_sq = B::float_add(real_sq, imag_sq);
        B::float_sqrt(norm_sq)
    }

    fn complex_from_parts(real: TensorData, imag: TensorData) -> ComplexTensor<SplitBackend<B, D>> {
        ComplexTensor::<SplitBackend<B, D>>::new(
            B::float_from_data(real, &Default::default()),
            B::float_from_data(imag, &Default::default()),
        )
    }

    fn complex_exp(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        // formula: e^(a + bi) = e^a * (cos(b) + i*sin(b)) = from_polar(e^a, b)
        //TODO: add the checks for corner cases +∞, -∞, and NaN
        //https://github.com/skewballfox/burn/blob/67d84b677b3d718cb25fbdc2535dbf04706b0863/crates/burn-complex/src/base/element.rs#L322-L323
        let exp_real = B::float_exp(tensor.real.clone());
        let cos_imag = B::float_cos(tensor.imag.clone());
        let sin_imag = B::float_sin(tensor.imag);

        SplitComplexTensor::new(
            B::float_mul(exp_real.clone(), cos_imag),
            B::float_mul(exp_real, sin_imag),
        )
    }

    fn complex_log(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        // formula: ln(z) = ln|z| + i*arg(z)
        // where |z| = sqrt(real^2 + imag^2) and arg(z) = atan2(imag, real)

        // Compute norm: sqrt(real^2 + imag^2)
        let real_sq = B::float_mul(tensor.real.clone(), tensor.real.clone());
        let imag_sq = B::float_mul(tensor.imag.clone(), tensor.imag.clone());
        let norm_sq = B::float_add(real_sq, imag_sq);
        let norm = B::float_sqrt(norm_sq);

        // Compute arg: atan2(imag, real)
        let arg = B::float_atan2(tensor.imag, tensor.real);

        SplitComplexTensor::new(B::float_log(norm), arg)
    }

    fn complex_squared_norm(tensor: ComplexTensor<SplitBackend<B, D>>) -> B::FloatTensorPrimitive {
        let real_sq = B::float_mul(tensor.real.clone(), tensor.real.clone());
        let imag_sq = B::float_mul(tensor.imag.clone(), tensor.imag.clone());
        B::float_add(real_sq, imag_sq)
    }

    fn complex_from_polar(
        magnitude: B::FloatTensorPrimitive,
        phase: B::FloatTensorPrimitive,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        ComplexTensor::<SplitBackend<B, D>>::new(
            B::float_mul(magnitude.clone(), B::float_cos(phase.clone())),
            B::float_mul(magnitude, B::float_sin(phase)),
        )
    }

    fn complex_gather(
        dim: usize,
        tensor: ComplexTensor<SplitBackend<B, D>>,
        indices: B::IntTensorPrimitive,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        ComplexTensor::<SplitBackend<B, D>>::new(
            B::float_gather(dim, tensor.real, indices.clone()),
            B::float_gather(dim, tensor.imag, indices),
        )
    }

    fn complex_scatter_add(
        dim: usize,
        tensor: ComplexTensor<SplitBackend<B, D>>,
        indices: B::IntTensorPrimitive,
        values: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        ComplexTensor::<SplitBackend<B, D>>::new(
            B::float_scatter_add(dim, tensor.real, indices.clone(), values.real),
            B::float_scatter_add(dim, tensor.imag, indices, values.imag),
        )
    }

    fn complex_random(
        shape: burn_std::Shape,
        distribution: burn_tensor::Distribution,
        device: &Device<B>,
        dtype: FloatDType,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        ComplexTensor::<SplitBackend<B, D>>::new(
            B::float_random(shape.clone(), distribution, device, dtype),
            B::float_random(shape, distribution, device, dtype),
        )
    }

    fn complex_to_device(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        device: &Device<B>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            B::float_to_device(tensor.real, device),
            B::float_to_device(tensor.imag, device),
        )
    }

    fn complex_reshape(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        shape: burn_std::Shape,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            B::float_reshape(tensor.real, shape.clone()),
            B::float_reshape(tensor.imag, shape),
        )
    }

    fn complex_transpose(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            B::float_transpose(tensor.real),
            B::float_transpose(tensor.imag),
        )
    }

    fn complex_neg(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(B::float_neg(tensor.real), B::float_neg(tensor.imag))
    }

    fn conj(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        // conj(a + bi) = a - bi
        SplitComplexTensor::new(tensor.real, B::float_neg(tensor.imag))
    }

    fn complex_arg(tensor: ComplexTensor<SplitBackend<B, D>>) -> B::FloatTensorPrimitive {
        // arg(a + bi) = atan2(b, a)
        B::float_atan2(tensor.imag, tensor.real)
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
        let arg = B::float_atan2(tensor.imag, tensor.real);
        let half_arg = B::float_div_scalar(arg, burn_tensor::Scalar::Float(2.0));
        SplitBackend::<B, D>::complex_from_polar(sqrt_abs, half_arg)
    }

    fn complex_sin(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        // sin(a + bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
        SplitComplexTensor::new(
            B::float_mul(
                B::float_sin(tensor.real.clone()),
                B::float_cosh(tensor.imag.clone()),
            ),
            B::float_mul(B::float_cos(tensor.real), B::float_sinh(tensor.imag)),
        )
    }

    fn complex_cos(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        // cos(a + bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
        SplitComplexTensor::new(
            B::float_mul(
                B::float_cos(tensor.real.clone()),
                B::float_cosh(tensor.imag.clone()),
            ),
            B::float_neg(B::float_mul(
                B::float_sin(tensor.real),
                B::float_sinh(tensor.imag),
            )),
        )
    }

    fn complex_tan(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        // tan(z) = sin(z) / cos(z)
        let sin = SplitBackend::<B, D>::complex_sin(tensor.clone());
        let cos = SplitBackend::<B, D>::complex_cos(tensor);
        SplitBackend::<B, D>::complex_div(sin, cos)
    }

    fn complex_acos(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // acos(z) = -i * ln(z + i * sqrt(1 - z²))
        let device = B::float_device(&tensor.real);
        let shape = tensor.real.shape().clone();
        let fdtype = tensor.real.dtype().into();
        let ones = SplitComplexTensor::new(
            B::float_ones(shape.clone(), &device, fdtype),
            B::float_zeros(shape, &device, fdtype),
        );
        // 1 - z²
        let z_sq = SplitBackend::<B, D>::complex_mul(tensor.clone(), tensor.clone());
        let one_minus_z_sq = SplitBackend::<B, D>::complex_sub(ones, z_sq);
        // i * sqrt(1 - z²): multiply by i via (-imag, real)
        let sqrt_term = SplitBackend::<B, D>::complex_sqrt(one_minus_z_sq);
        let i_sqrt = SplitComplexTensor::new(B::float_neg(sqrt_term.imag), sqrt_term.real);
        // z + i*sqrt(1 - z²)
        let inner = SplitBackend::<B, D>::complex_add(tensor, i_sqrt);
        // -i * ln(inner): multiply by -i via (imag, -real)
        let log_inner = SplitBackend::<B, D>::complex_log(inner);
        SplitComplexTensor::new(log_inner.imag, B::float_neg(log_inner.real))
    }

    fn complex_acosh(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // acosh(z) = ln(z + sqrt(z² - 1))
        let device = B::float_device(&tensor.real);
        let shape = tensor.real.shape().clone();
        let fdtype = tensor.real.dtype().into();
        let ones = SplitComplexTensor::new(
            B::float_ones(shape.clone(), &device, fdtype),
            B::float_zeros(shape, &device, fdtype),
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
        let device = B::float_device(&tensor.real);
        let shape = tensor.real.shape().clone();
        let fdtype = tensor.real.dtype().into();
        let ones = SplitComplexTensor::new(
            B::float_ones(shape.clone(), &device, fdtype),
            B::float_zeros(shape, &device, fdtype),
        );
        // z² and i*z before consuming tensor
        let z_sq = SplitBackend::<B, D>::complex_mul(tensor.clone(), tensor.clone());
        // i*z = (-imag, real)
        let i_z = SplitComplexTensor::new(B::float_neg(tensor.imag), tensor.real);
        // 1 - z²
        let one_minus_z_sq = SplitBackend::<B, D>::complex_sub(ones, z_sq);
        // i*z + sqrt(1 - z²)
        let sqrt_term = SplitBackend::<B, D>::complex_sqrt(one_minus_z_sq);
        let inner = SplitBackend::<B, D>::complex_add(i_z, sqrt_term);
        // -i * ln(inner): (imag, -real)
        let log_inner = SplitBackend::<B, D>::complex_log(inner);
        SplitComplexTensor::new(log_inner.imag, B::float_neg(log_inner.real))
    }

    fn complex_asinh(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // asinh(z) = ln(z + sqrt(z² + 1))
        let device = B::float_device(&tensor.real);
        let shape = tensor.real.shape().clone();
        let fdtype = tensor.real.dtype().into();
        let ones = SplitComplexTensor::new(
            B::float_ones(shape.clone(), &device, fdtype),
            B::float_zeros(shape, &device, fdtype),
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
        let device = B::float_device(&tensor.real);
        let shape = tensor.real.shape().clone();
        let fdtype = tensor.real.dtype().into();
        let ones = SplitComplexTensor::new(
            B::float_ones(shape.clone(), &device, fdtype),
            B::float_zeros(shape, &device, fdtype),
        );
        // i*z = (-imag, real)
        let i_z = SplitComplexTensor::new(B::float_neg(tensor.imag), tensor.real);
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
            B::float_div_scalar(log_ratio.imag, burn_tensor::Scalar::Float(2.0)),
            B::float_neg(B::float_div_scalar(
                log_ratio.real,
                burn_tensor::Scalar::Float(2.0),
            )),
        )
    }

    fn complex_atanh(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // atanh(z) = (1/2) * ln((1 + z) / (1 - z))
        let device = B::float_device(&tensor.real);
        let shape = tensor.real.shape().clone();
        let fdtype = tensor.real.dtype().into();
        let ones = SplitComplexTensor::new(
            B::float_ones(shape.clone(), &device, fdtype),
            B::float_zeros(shape, &device, fdtype),
        );
        let one_plus_z = SplitBackend::<B, D>::complex_add(ones.clone(), tensor.clone());
        let one_minus_z = SplitBackend::<B, D>::complex_sub(ones, tensor);
        let log_ratio = SplitBackend::<B, D>::complex_log(SplitBackend::<B, D>::complex_div(
            one_plus_z,
            one_minus_z,
        ));
        SplitComplexTensor::new(
            B::float_div_scalar(log_ratio.real, burn_tensor::Scalar::Float(2.0)),
            B::float_div_scalar(log_ratio.imag, burn_tensor::Scalar::Float(2.0)),
        )
    }

    fn complex_slice(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        slices: &[burn_tensor::Slice],
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            B::float_slice(tensor.real, slices),
            B::float_slice(tensor.imag, slices),
        )
    }

    fn complex_slice_assign(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        ranges: &[burn_tensor::Slice],
        value: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            B::float_slice_assign(tensor.real, ranges, value.real),
            B::float_slice_assign(tensor.imag, ranges, value.imag),
        )
    }

    fn complex_swap_dims(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim1: usize,
        dim2: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            B::float_swap_dims(tensor.real, dim1, dim2),
            B::float_swap_dims(tensor.imag, dim1, dim2),
        )
    }

    fn complex_repeat_dim(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
        times: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            B::float_repeat_dim(tensor.real, dim, times),
            B::float_repeat_dim(tensor.imag, dim, times),
        )
    }

    fn complex_cat(
        tensors: Vec<ComplexTensor<SplitBackend<B, D>>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        let (reals, imags): (Vec<_>, Vec<_>) =
            tensors.into_iter().map(|t| (t.real, t.imag)).unzip();
        SplitComplexTensor::new(B::float_cat(reals, dim), B::float_cat(imags, dim))
    }

    fn complex_any(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let real_any = B::float_any(tensor.real, out_dtype);
        let imag_any = B::float_any(tensor.imag, out_dtype);
        B::bool_or(real_any, imag_any)
    }

    fn complex_any_dim(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let real_any = B::float_any_dim(tensor.real, dim, out_dtype);
        let imag_any = B::float_any_dim(tensor.imag, dim, out_dtype);
        B::bool_or(real_any, imag_any)
    }

    fn complex_all(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        // A complex element is nonzero if either real or imag is nonzero.
        // all(z != 0) = all(real != 0 || imag != 0)
        let real_nonzero =
            B::float_not_equal_elem(tensor.real, burn_tensor::Scalar::Float(0.0), out_dtype);
        let imag_nonzero =
            B::float_not_equal_elem(tensor.imag, burn_tensor::Scalar::Float(0.0), out_dtype);
        let elem_nonzero = B::bool_or(real_nonzero, imag_nonzero);
        B::bool_all(elem_nonzero)
    }

    fn complex_all_dim(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> B::BoolTensorPrimitive {
        let real_nonzero =
            B::float_not_equal_elem(tensor.real, burn_tensor::Scalar::Float(0.0), out_dtype);
        let imag_nonzero =
            B::float_not_equal_elem(tensor.imag, burn_tensor::Scalar::Float(0.0), out_dtype);
        let elem_nonzero = B::bool_or(real_nonzero, imag_nonzero);
        B::bool_all_dim(elem_nonzero, dim)
    }

    fn complex_permute(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        axes: &[usize],
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            B::float_permute(tensor.real, axes),
            B::float_permute(tensor.imag, axes),
        )
    }

    fn complex_expand(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        shape: burn_std::Shape,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            B::float_expand(tensor.real, shape.clone()),
            B::float_expand(tensor.imag, shape),
        )
    }

    fn complex_flip(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        axes: &[usize],
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            B::float_flip(tensor.real, axes),
            B::float_flip(tensor.imag, axes),
        )
    }

    fn complex_unfold(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            B::float_unfold(tensor.real, dim, size, step),
            B::float_unfold(tensor.imag, dim, size, step),
        )
    }

    fn complex_select(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
        indices: B::IntTensorPrimitive,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            B::float_select(tensor.real, dim, indices.clone()),
            B::float_select(tensor.imag, dim, indices),
        )
    }

    fn complex_sum(tensor: ComplexTensor<SplitBackend<B, D>>) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(B::float_sum(tensor.real), B::float_sum(tensor.imag))
    }

    fn complex_sum_dim(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            B::float_sum_dim(tensor.real, dim),
            B::float_sum_dim(tensor.imag, dim),
        )
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
        SplitComplexTensor::new(B::float_mean(tensor.real), B::float_mean(tensor.imag))
    }

    fn complex_mean_dim(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            B::float_mean_dim(tensor.real, dim),
            B::float_mean_dim(tensor.imag, dim),
        )
    }

    fn complex_remainder(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // Componentwise remainder (matching Complex<E> Rem impl)
        SplitComplexTensor::new(
            B::float_remainder(lhs.real, rhs.real),
            B::float_remainder(lhs.imag, rhs.imag),
        )
    }

    fn complex_remainder_scalar(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: <SplitBackend<B, D> as ComplexTensorBackend>::ComplexScalar,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        use burn_tensor::cast::ToElement;
        SplitComplexTensor::new(
            B::float_remainder_scalar(lhs.real, burn_tensor::Scalar::Float(rhs.real().to_f64())),
            B::float_remainder_scalar(lhs.imag, burn_tensor::Scalar::Float(rhs.imag().to_f64())),
        )
    }

    fn complex_mask_where(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        mask: B::BoolTensorPrimitive,
        source: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        SplitComplexTensor::new(
            B::float_mask_where(tensor.real, mask.clone(), source.real),
            B::float_mask_where(tensor.imag, mask, source.imag),
        )
    }

    fn complex_mask_fill(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        mask: B::BoolTensorPrimitive,
        value: <SplitBackend<B, D> as ComplexTensorBackend>::ComplexScalar,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        use burn_tensor::cast::ToElement;
        SplitComplexTensor::new(
            B::float_mask_fill(
                tensor.real,
                mask.clone(),
                burn_tensor::Scalar::Float(value.real().to_f64()),
            ),
            B::float_mask_fill(
                tensor.imag,
                mask,
                burn_tensor::Scalar::Float(value.imag().to_f64()),
            ),
        )
    }

    fn complex_sign(
        tensor: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // sign(z) = z / |z| = from_polar(1, arg(z))
        let abs = SplitBackend::<B, D>::abs(tensor.clone());
        SplitComplexTensor::new(
            B::float_div(tensor.real, abs.clone()),
            B::float_div(tensor.imag, abs),
        )
    }

    fn complex_matmul(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: ComplexTensor<SplitBackend<B, D>>,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // (A + iB)(C + iD) = (AC - BD) + i(AD + BC)
        SplitComplexTensor::new(
            B::float_sub(
                B::float_matmul(lhs.real.clone(), rhs.real.clone()),
                B::float_matmul(lhs.imag.clone(), rhs.imag.clone()),
            ),
            B::float_add(
                B::float_matmul(lhs.real, rhs.imag),
                B::float_matmul(lhs.imag, rhs.real),
            ),
        )
    }

    fn complex_cumsum(
        tensor: ComplexTensor<SplitBackend<B, D>>,
        dim: usize,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // cumsum is linear, so it works componentwise
        SplitComplexTensor::new(
            B::float_cumsum(tensor.real, dim),
            B::float_cumsum(tensor.imag, dim),
        )
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
            B::float_select_add(tensor.real, dim, indices.clone(), values.real),
            B::float_select_add(tensor.imag, dim, indices, values.imag),
        )
    }

    fn complex_powc_scalar(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: <SplitBackend<B, D> as ComplexTensorBackend>::ComplexScalar,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // z^c = exp(c * ln(z)) where c = a + bi is a scalar
        // (a + bi) * (u + vi) = (au - bv) + (av + bu)i
        let a = rhs.real().to_f64();
        let b = rhs.imag().to_f64();
        let ln_z = SplitBackend::<B, D>::complex_log(lhs);
        let c_ln_z = SplitComplexTensor::new(
            B::float_sub(
                B::float_mul_scalar(ln_z.real.clone(), burn_tensor::Scalar::Float(a)),
                B::float_mul_scalar(ln_z.imag.clone(), burn_tensor::Scalar::Float(b)),
            ),
            B::float_add(
                B::float_mul_scalar(ln_z.real, burn_tensor::Scalar::Float(b)),
                B::float_mul_scalar(ln_z.imag, burn_tensor::Scalar::Float(a)),
            ),
        );
        SplitBackend::<B, D>::complex_exp(c_ln_z)
    }

    fn complex_powf(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: B::FloatTensorPrimitive,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // z^w = exp(w * ln(z)) where w is a real tensor
        let log_z = SplitBackend::<B, D>::complex_log(lhs);
        let w_log_z = SplitComplexTensor::new(
            B::float_mul(rhs.clone(), log_z.real),
            B::float_mul(rhs, log_z.imag),
        );
        SplitBackend::<B, D>::complex_exp(w_log_z)
    }

    fn complex_powf_scalar(
        lhs: ComplexTensor<SplitBackend<B, D>>,
        rhs: Scalar,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // z^w = exp(w * ln(z)) where w is a real scalar
        let log_z = SplitBackend::<B, D>::complex_log(lhs);
        let w_log_z = SplitComplexTensor::new(
            B::float_mul_scalar(log_z.real, rhs),
            B::float_mul_scalar(log_z.imag, rhs),
        );
        SplitBackend::<B, D>::complex_exp(w_log_z)
    }

    fn complex_scatter_nd(
        _tensor: ComplexTensor<SplitBackend<B, D>>,
        _indices: B::IntTensorPrimitive,
        _value: ComplexTensor<SplitBackend<B, D>>,
        _reduction: burn_tensor::IndexingUpdateOp,
    ) -> ComplexTensor<SplitBackend<B, D>> {
        // can't implement scatter_nd via trait as SplitTensor isn't generic over rank
        // In order to be api compatible with Tensor<B,D,K> anything that requires more than 1 generic const rank
        // needs to be implemented directly on the SplitTensor type.
        unreachable!("inlined into scatter_nd for SplitTensor")
    }
}

impl<B, const D: usize, F> DefaultComplexOps<B> for SplitLayout
where
    B: ComplexTensorBackend<Layout = SplitLayout>,
    B: BackendTypes<FloatTensorPrimitive = F>,
    B: CBT<ComplexTensorPrimitive = SplitComplexTensor<B::InnerBackend, D, F>>,
    F: TensorMetadata + 'static,
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
        SplitComplexTensor::new(real, imag)
    }

    fn ones(shape: Shape, device: &Device<B>, _dtype: ComplexDType) -> ComplexTensor<B> {
        let real = B::InnerBackend::float_from_data(
            TensorData::ones::<<B::InnerBackend as BackendTypes>::FloatElem, _>(&shape),
            device,
        );
        let imag = B::InnerBackend::float_from_data(
            TensorData::zeros::<<B::InnerBackend as BackendTypes>::FloatElem, _>(shape),
            device,
        );
        SplitComplexTensor::new(real, imag)
    }

    fn full(shape: Shape, fill_value: B::ComplexScalar, device: &Device<B>) -> ComplexTensor<B> {
        SplitComplexTensor::new(
            B::InnerBackend::float_from_data(TensorData::full(&shape, fill_value.real()), device),
            B::InnerBackend::float_from_data(TensorData::full(shape, fill_value.imag()), device),
        )
    }

    async fn complex_into_data(
        tensor: ComplexTensor<B>,
    ) -> Result<Self::OutTensorData, ExecutionError> {
        B::complex_into_split_data(tensor).await
    }
}
