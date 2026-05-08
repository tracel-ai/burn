use crate::base::{CBT, ComplexTensor, ComplexTensorBackend};
use alloc::vec::Vec;
use burn_std::{DType, FloatDType, Shape, Slice};
use burn_tensor::{
    BasicOps, Device, Distribution, FloatMathOps, IndexingUpdateOp, Int, Numeric, Scalar, Tensor,
    TensorData, TensorKind, TensorMetadata,
    backend::{Backend, BackendTypes, ExecutionError},
    get_device_settings,
};

/// A type-level representation of the kind of a complex tensor.
#[derive(Clone, Debug)]
pub struct ComplexKind;

#[allow(unused_variables)]
impl<C: ComplexTensorBackend> BasicOps<C> for ComplexKind {
    type Elem = C::ComplexScalar;

    fn empty(shape: Shape, device: &C::Device, dtype: DType) -> Self::Primitive {
        // should I check then pass the dtype?
        C::complex_zeros(shape, device, dtype.into())
    }

    fn reshape(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        C::complex_reshape(tensor, shape)
    }

    fn transpose(tensor: Self::Primitive) -> Self::Primitive {
        C::complex_transpose(tensor)
    }

    fn swap_dims(tensor: Self::Primitive, dim1: usize, dim2: usize) -> Self::Primitive {
        C::complex_swap_dims(tensor, dim1, dim2)
    }

    fn slice(tensor: Self::Primitive, ranges: &[Slice]) -> Self::Primitive {
        //TensorPrimitive::Complex(B::complex_slice(tensor, ranges))
        C::complex_slice(tensor, ranges)
    }

    fn device(tensor: &Self::Primitive) -> Device<C> {
        C::complex_device(tensor)
    }

    fn to_device(tensor: Self::Primitive, device: &C::Device) -> Self::Primitive {
        C::complex_to_device(tensor, device)
    }

    async fn into_data_async(tensor: Self::Primitive) -> Result<TensorData, ExecutionError> {
        C::complex_into_interleaved_data(tensor).await
    }

    fn from_data(data: TensorData, device: &C::Device, dtype: DType) -> Self::Primitive {
        C::complex_from_interleaved_data(data.convert::<C::ComplexScalar>(), device)
    }

    fn repeat_dim(tensor: Self::Primitive, dim: usize, times: usize) -> Self::Primitive {
        C::complex_repeat_dim(tensor, dim, times)
    }
    fn equal(lhs: Self::Primitive, rhs: Self::Primitive) -> C::BoolTensorPrimitive {
        let out_dtype = get_device_settings::<C>(&C::complex_device(&lhs)).bool_dtype;
        C::complex_equal(lhs, rhs, out_dtype)
    }

    fn not_equal(lhs: Self::Primitive, rhs: Self::Primitive) -> C::BoolTensorPrimitive {
        let out_dtype = get_device_settings::<C>(&C::complex_device(&lhs)).bool_dtype;
        C::complex_not_equal(lhs, rhs, out_dtype)
    }

    fn cat(tensors: Vec<Self::Primitive>, dim: usize) -> Self::Primitive {
        C::complex_cat(tensors, dim)
    }

    fn any(tensor: Self::Primitive) -> C::BoolTensorPrimitive {
        let out_dtype = get_device_settings::<C>(&C::complex_device(&tensor)).bool_dtype;
        C::complex_any(tensor, out_dtype)
    }

    fn any_dim(tensor: Self::Primitive, dim: usize) -> C::BoolTensorPrimitive {
        let out_dtype = get_device_settings::<C>(&C::complex_device(&tensor)).bool_dtype;
        C::complex_any_dim(tensor, dim, out_dtype)
    }

    fn all(tensor: Self::Primitive) -> C::BoolTensorPrimitive {
        let out_dtype = get_device_settings::<C>(&C::complex_device(&tensor)).bool_dtype;
        C::complex_all(tensor, out_dtype)
    }

    fn all_dim(tensor: Self::Primitive, dim: usize) -> C::BoolTensorPrimitive {
        let out_dtype = get_device_settings::<C>(&C::complex_device(&tensor)).bool_dtype;
        C::complex_all_dim(tensor, dim, out_dtype)
    }

    fn permute(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        C::complex_permute(tensor, axes)
    }

    fn expand(tensor: Self::Primitive, shape: Shape) -> Self::Primitive {
        C::complex_expand(tensor, shape)
    }

    fn flip(tensor: Self::Primitive, axes: &[usize]) -> Self::Primitive {
        C::complex_flip(tensor, axes)
    }

    fn unfold(tensor: Self::Primitive, dim: usize, size: usize, step: usize) -> Self::Primitive {
        C::complex_unfold(tensor, dim, size, step)
    }

    fn slice_assign(
        tensor: Self::Primitive,
        ranges: &[Slice],
        value: Self::Primitive,
    ) -> Self::Primitive {
        C::complex_slice_assign(tensor, ranges, value)
    }

    fn select(
        tensor: Self::Primitive,
        dim: usize,
        indices: <C as BackendTypes>::IntTensorPrimitive,
    ) -> Self::Primitive {
        // Uses your existing `select` name.
        C::complex_select(tensor, dim, indices)
    }

    fn select_assign(
        tensor: Self::Primitive,
        dim: usize,
        indices: <C as BackendTypes>::IntTensorPrimitive,
        values: Self::Primitive,
        update: IndexingUpdateOp,
    ) -> Self::Primitive {
        match update {
            IndexingUpdateOp::Add => C::complex_select_add(tensor, dim, indices, values),
            _ => unimplemented!(),
        }
    }

    fn zeros(shape: Shape, device: &<C as BackendTypes>::Device, dtype: DType) -> Self::Primitive {
        match dtype {
            DType::Complex32 | DType::Complex64 => C::complex_zeros(shape, device, dtype.into()),
            _ => panic!("Unsupported complex dtype"),
        }
    }

    fn ones(shape: Shape, device: &<C as BackendTypes>::Device, dtype: DType) -> Self::Primitive {
        match dtype {
            DType::Complex32 | DType::Complex64 => C::complex_ones(shape, device, dtype.into()),
            _ => panic!("Unsupported complex dtype"),
        }
    }

    fn mask_where(
        tensor: Self::Primitive,
        mask: C::BoolTensorPrimitive,
        source: Self::Primitive,
    ) -> Self::Primitive {
        C::complex_mask_where(tensor, mask, source)
    }

    fn mask_fill(
        tensor: Self::Primitive,
        mask: C::BoolTensorPrimitive,
        value: burn_tensor::Scalar,
    ) -> Self::Primitive {
        C::complex_mask_fill(tensor, mask, value.elem())
    }

    fn gather(
        dim: usize,
        tensor: Self::Primitive,
        indices: C::IntTensorPrimitive,
    ) -> Self::Primitive {
        C::complex_gather(dim, tensor, indices)
    }

    fn scatter(
        dim: usize,
        tensor: Self::Primitive,
        indices: C::IntTensorPrimitive,
        values: Self::Primitive,
        update: burn_tensor::IndexingUpdateOp,
    ) -> Self::Primitive {
        match update {
            IndexingUpdateOp::Add => C::complex_scatter_add(dim, tensor, indices, values),
            _ => unimplemented!(),
        }
    }

    fn equal_elem(
        lhs: Self::Primitive,
        rhs: burn_tensor::Scalar,
    ) -> <C as BackendTypes>::BoolTensorPrimitive {
        let out_dtype = get_device_settings::<C>(&C::complex_device(&lhs)).bool_dtype;
        C::complex_equal_elem(lhs, rhs.elem(), out_dtype)
    }

    fn not_equal_elem(
        lhs: Self::Primitive,
        rhs: burn_tensor::Scalar,
    ) -> <C as BackendTypes>::BoolTensorPrimitive {
        let out_dtype = get_device_settings::<C>(&C::complex_device(&lhs)).bool_dtype;
        C::complex_not_equal_elem(lhs, rhs.elem(), out_dtype)
    }

    fn full(
        shape: Shape,
        fill_value: burn_tensor::Scalar,
        device: &<C as BackendTypes>::Device,
        dtype: DType,
    ) -> Self::Primitive {
        // Enforce complex dtype for clarity (mirrors from_data_dtype below).
        if !dtype.is_complex() {
            panic!("Expected complex dtype, got {dtype:?}");
        }
        // `elem()` should yield something convertible to `B::ComplexElem`.
        C::complex_full(shape, fill_value.elem(), device)
    }

    fn scatter_nd(
        data: Self::Primitive,
        indices: burn_tensor::ops::IntTensor<C>,
        values: Self::Primitive,
        reduction: IndexingUpdateOp,
    ) -> Self::Primitive {
        C::complex_scatter_nd(data, indices, values, reduction)
    }

    fn gather_nd(
        data: Self::Primitive,
        indices: burn_tensor::ops::IntTensor<C>,
    ) -> Self::Primitive {
        C::complex_gather_nd(data, indices)
    }
}

/// Operations that are specific to complex tensors and have no analogue for real tensors.
pub trait ComplexOnlyOps<C: ComplexTensorBackend> {
    /// Computes the complex conjugate of each element, negating the imaginary part.
    ///
    /// # Arguments
    ///
    /// * `self` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A new tensor where each element `a + bi` is replaced by `a - bi`.
    fn conj(self) -> C::ComplexTensorPrimitive;

    /// Computes the phase angle (argument) of each complex element, in radians.
    ///
    /// # Arguments
    ///
    /// * `self` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A real-valued tensor containing the angle `atan2(im, re)` for each element,
    /// in the range `(-π, π]`.
    fn phase(self) -> C::FloatTensorPrimitive;

    /// Extracts the real part of each complex element.
    ///
    /// # Arguments
    ///
    /// * `self` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A real-valued tensor containing the real component of each element.
    fn real(self) -> C::FloatTensorPrimitive;

    /// Extracts the imaginary part of each complex element.
    ///
    /// # Arguments
    ///
    /// * `self` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A real-valued tensor containing the imaginary component of each element.
    fn imag(self) -> C::FloatTensorPrimitive;

    /// Computes the magnitude (absolute value) of each complex element.
    ///
    /// # Arguments
    ///
    /// * `self` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A real-valued tensor containing `sqrt(re² + im²)` for each element.
    fn magnitude(self) -> C::FloatTensorPrimitive;

    /// Creates a complex tensor by combining separate real and imaginary part tensors.
    ///
    /// # Arguments
    ///
    /// * `real` - The real parts, as anything that can be converted into `TensorData`.
    /// * `imag` - The imaginary parts, as anything that can be converted into `TensorData`.
    ///
    /// # Returns
    ///
    /// A complex tensor whose shape matches the input data.
    fn from_parts<T>(real: T, imag: T) -> Self
    where
        T: Into<TensorData>;

    /// Creates a complex tensor from interleaved `[re₀, im₀, re₁, im₁, …]` data.
    ///
    /// # Arguments
    ///
    /// * `data` - Flat tensor data with real and imaginary values interleaved.
    /// * `device` - The device on which the tensor will be allocated.
    ///
    /// # Returns
    ///
    /// A complex tensor whose element count is half the length of the flat data.
    fn from_interleaved_data(data: TensorData, device: &C::Device) -> Self;

    /// Creates a complex tensor from polar form, converting `(r, θ)` pairs to `r·cos θ + i·r·sin θ`.
    ///
    /// # Arguments
    ///
    /// * `magnitude` - A real-valued tensor of radii `r`.
    /// * `phase` - A real-valued tensor of angles `θ` in radians.
    ///
    /// # Returns
    ///
    /// A complex tensor with the same shape as the inputs.
    fn from_polar(magnitude: C::FloatTensorPrimitive, phase: C::FloatTensorPrimitive) -> Self;
}

impl<C: ComplexTensorBackend + Backend, const D: usize> ComplexOnlyOps<C>
    for Tensor<C, D, ComplexKind>
{
    fn conj(self) -> C::ComplexTensorPrimitive {
        C::conj(self.into_primitive())
    }
    fn phase(self) -> C::FloatTensorPrimitive {
        C::phase(self.into_primitive())
    }

    fn from_interleaved_data(
        data: TensorData,
        device: &C::Device,
    ) -> burn_tensor::Tensor<C, D, ComplexKind> {
        Tensor::from_primitive(C::complex_from_interleaved_data(data, device))
    }

    fn real(self) -> C::FloatTensorPrimitive {
        C::real(self.into_primitive())
    }

    fn imag(self) -> C::FloatTensorPrimitive {
        C::imag(self.into_primitive())
    }

    fn magnitude(self) -> C::FloatTensorPrimitive {
        C::abs(self.into_primitive())
    }

    fn from_parts<T>(real: T, imag: T) -> Self
    where
        T: Into<TensorData>,
    {
        Self::new(C::complex_from_parts(real.into(), imag.into()))
    }
    fn from_polar(magnitude: C::FloatTensorPrimitive, phase: C::FloatTensorPrimitive) -> Self {
        Self::new(C::complex_from_polar(magnitude, phase))
    }
}

#[allow(unused_variables)]
impl<C: ComplexTensorBackend> Numeric<C> for ComplexKind
where
    C: CBT + core::fmt::Debug + Clone,
{
    type IntTensor = Int;
    fn add(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        C::complex_add(lhs, rhs)
    }

    fn sub(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        C::complex_sub(lhs, rhs)
    }

    fn sub_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        let device = C::complex_device(&lhs);
        let shape = C::complex_shape(&lhs);
        let scalar_complex: C::ComplexScalar = rhs.elem();
        let scalar_tensor = C::complex_full(shape, scalar_complex, &device);
        C::complex_sub(lhs, scalar_tensor)
    }

    fn mul(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        C::complex_mul(lhs, rhs)
    }

    fn mul_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        let device = C::complex_device(&lhs);
        let shape = C::complex_shape(&lhs);
        let scalar_complex: C::ComplexScalar = rhs.elem();
        let scalar_tensor = C::complex_full(shape, scalar_complex, &device);
        C::complex_mul(lhs, scalar_tensor)
    }

    fn div(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        C::complex_div(lhs, rhs)
    }

    fn div_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        let device = C::complex_device(&lhs);
        let shape = C::complex_shape(&lhs);
        let scalar_complex: C::ComplexScalar = rhs.elem();
        let scalar_tensor = C::complex_full(shape, scalar_complex, &device);
        C::complex_div(lhs, scalar_tensor)
    }

    fn random(
        shape: Shape,
        distribution: Distribution,
        device: &C::Device,
        dtype: DType,
    ) -> Self::Primitive {
        C::complex_random(
            shape,
            distribution,
            device,
            FloatDType::from(crate::utils::complex_to_real_dtype(dtype)),
        )
    }

    fn remainder(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        // not mathematically defined; mimic float backend remainder
        C::complex_remainder(lhs, rhs)
    }

    fn remainder_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        C::complex_remainder_scalar(lhs, rhs.elem())
    }

    fn sum(tensor: Self::Primitive) -> Self::Primitive {
        C::complex_sum(tensor)
    }

    fn sum_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        C::complex_sum_dim(tensor, dim)
    }

    fn prod(tensor: Self::Primitive) -> Self::Primitive {
        C::complex_prod(tensor)
    }

    fn prod_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        C::complex_prod_dim(tensor, dim)
    }

    fn mean(tensor: Self::Primitive) -> Self::Primitive {
        C::complex_mean(tensor)
    }

    fn mean_dim(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        C::complex_mean_dim(tensor, dim)
    }

    fn powi(lhs: Self::Primitive, rhs: C::IntTensorPrimitive) -> Self::Primitive {
        C::complex_powi(lhs, rhs)
    }

    fn powi_scalar(lhs: Self::Primitive, rhs: Scalar) -> Self::Primitive {
        C::complex_powi_scalar(lhs, rhs)
    }

    fn matmul(lhs: Self::Primitive, rhs: Self::Primitive) -> Self::Primitive {
        C::complex_matmul(lhs, rhs)
    }

    fn cumsum(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        C::complex_cumsum(tensor, dim)
    }

    fn cumprod(tensor: Self::Primitive, dim: usize) -> Self::Primitive {
        C::complex_cumprod(tensor, dim)
    }

    fn neg(tensor: Self::Primitive) -> Self::Primitive {
        C::complex_neg(tensor)
    }

    fn sign(tensor: Self::Primitive) -> Self::Primitive {
        C::complex_sign(tensor)
    }

    fn add_scalar(lhs: Self::Primitive, rhs: burn_tensor::Scalar) -> Self::Primitive {
        let device = C::complex_device(&lhs);
        let shape = C::complex_shape(&lhs);
        let scalar_complex: C::ComplexScalar = rhs.elem();
        let scalar_tensor = C::complex_full(shape, scalar_complex, &device);
        C::complex_add(lhs, scalar_tensor)
    }
}

impl<C: ComplexTensorBackend> FloatMathOps<C> for ComplexKind
where
    C: CBT + core::fmt::Debug + Clone,
    ComplexTensor<C>: Clone,
{
    fn exp(tensor: Self::Primitive) -> C::ComplexTensorPrimitive {
        C::complex_exp(tensor)
    }

    fn sin(tensor: Self::Primitive) -> C::ComplexTensorPrimitive {
        C::complex_sin(tensor)
    }

    fn cos(tensor: Self::Primitive) -> C::ComplexTensorPrimitive {
        C::complex_cos(tensor)
    }

    fn sqrt(tensor: Self::Primitive) -> C::ComplexTensorPrimitive {
        C::complex_sqrt(tensor)
    }

    fn log(tensor: Self::Primitive) -> C::ComplexTensorPrimitive {
        C::complex_log(tensor)
    }

    fn square(tensor: Self::Primitive) -> Self::Primitive {
        C::complex_powi_scalar(tensor, 2.into())
    }

    fn log1p(tensor: Self::Primitive) -> Self::Primitive {
        let dtype = tensor.dtype();
        // log1p(z) = log(1 + z)
        let device = C::complex_device(&tensor);
        let shape = C::complex_shape(&tensor);
        let ones = C::complex_ones(shape, &device, dtype.into());
        C::complex_log(C::complex_add(tensor, ones))
    }

    fn tan(tensor: Self::Primitive) -> Self::Primitive {
        C::complex_tan(tensor)
    }

    fn cosh(tensor: Self::Primitive) -> Self::Primitive {
        // cosh(z) = (exp(z) + exp(-z)) / 2
        let device = C::complex_device(&tensor);
        let shape = C::complex_shape(&tensor);
        let two: C::ComplexScalar = Scalar::from(2.0_f32).elem();
        let two_tensor = C::complex_full(shape, two, &device);
        let exp_z = C::complex_exp(tensor.clone());
        let exp_neg_z = C::complex_exp(C::complex_neg(tensor));
        C::complex_div(C::complex_add(exp_z, exp_neg_z), two_tensor)
    }

    fn sinh(tensor: Self::Primitive) -> Self::Primitive {
        // sinh(z) = (exp(z) - exp(-z)) / 2
        let device = C::complex_device(&tensor);
        let shape = C::complex_shape(&tensor);
        let two: C::ComplexScalar = Scalar::from(2.0_f32).elem();
        let two_tensor = C::complex_full(shape, two, &device);
        let exp_z = C::complex_exp(tensor.clone());
        let exp_neg_z = C::complex_exp(C::complex_neg(tensor));
        C::complex_div(C::complex_sub(exp_z, exp_neg_z), two_tensor)
    }

    fn tanh(tensor: Self::Primitive) -> Self::Primitive {
        let dtype = tensor.dtype().into();
        // tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
        let device = C::complex_device(&tensor);
        let shape = C::complex_shape(&tensor);
        let ones = C::complex_ones(shape, &device, dtype);
        let two_z = C::complex_add(tensor.clone(), tensor);
        let e2z = C::complex_exp(two_z);
        let numerator = C::complex_sub(e2z.clone(), ones.clone());
        let denominator = C::complex_add(e2z, ones);
        C::complex_div(numerator, denominator)
    }

    fn acos(tensor: Self::Primitive) -> Self::Primitive {
        C::complex_acos(tensor)
    }

    fn acosh(tensor: Self::Primitive) -> Self::Primitive {
        C::complex_acosh(tensor)
    }

    fn asin(tensor: Self::Primitive) -> Self::Primitive {
        C::complex_asin(tensor)
    }

    fn asinh(tensor: Self::Primitive) -> Self::Primitive {
        C::complex_asinh(tensor)
    }

    fn atan(tensor: Self::Primitive) -> Self::Primitive {
        C::complex_atan(tensor)
    }

    fn atanh(tensor: Self::Primitive) -> Self::Primitive {
        C::complex_atanh(tensor)
    }
}

impl<B: ComplexTensorBackend> TensorKind<B> for ComplexKind {
    type Primitive = ComplexTensor<B>;
    fn name() -> &'static str {
        "Complex"
    }
}
