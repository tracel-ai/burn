use alloc::vec::Vec;
use burn_backend::ops::ComplexTensorOps;
use burn_backend::{Backend, ComplexTensorBackend, Distribution, Scalar, TensorData};
use burn_dispatch::Dispatch;
use burn_std::{DType, ExecutionError, IndexingUpdateOp, Shape, Slice};

use crate::bridge::{BasicOps, Numeric};
use crate::ops::{BridgeTensor, ComplexKind, FloatMathOps};
use crate::{Device, Tensor};

impl BasicOps for ComplexKind {
    fn empty(shape: Shape, device: &Device, dtype: DType) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_zeros(
            shape,
            device.as_dispatch(),
            dtype.into(),
        ))
    }

    fn zeros(shape: Shape, device: &Device, dtype: DType) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_zeros(
            shape,
            device.as_dispatch(),
            dtype.into(),
        ))
    }

    fn ones(shape: Shape, device: &Device, dtype: DType) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_ones(
            shape,
            device.as_dispatch(),
            dtype.into(),
        ))
    }

    fn full(shape: Shape, fill_value: Scalar, device: &Device, dtype: DType) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_full(
            shape,
            fill_value,
            device.as_dispatch(),
        ))
    }

    fn reshape(tensor: BridgeTensor, shape: Shape) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_reshape(tensor.into_complex(), shape))
    }

    fn transpose(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_transpose(tensor.into_complex()))
    }

    fn swap_dims(tensor: BridgeTensor, dim1: usize, dim2: usize) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_swap_dims(
            tensor.into_complex(),
            dim1,
            dim2,
        ))
    }

    fn slice(tensor: BridgeTensor, ranges: &[Slice]) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_slice(tensor.into_complex(), ranges))
    }

    fn device(tensor: &BridgeTensor) -> Device {
        Device::new(Dispatch::complex_device(tensor.as_complex()))
    }

    fn to_device(tensor: BridgeTensor, device: &Device) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_to_device(
            tensor.into_complex(),
            device.as_dispatch(),
        ))
    }

    async fn into_data_async(tensor: BridgeTensor) -> Result<TensorData, ExecutionError> {
        Dispatch::complex_into_interleaved_data(tensor.into_complex()).await
    }

    fn from_data(data: TensorData, device: &Device, dtype: DType) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_from_interleaved_data(
            data,
            device.as_dispatch(),
        ))
    }

    fn repeat_dim(tensor: BridgeTensor, dim: usize, times: usize) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_repeat_dim(
            tensor.into_complex(),
            dim,
            times,
        ))
    }
    fn equal(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let lhs = lhs.into_complex();
        let out_dtype = Dispatch::complex_device(&lhs).settings().bool_dtype;
        BridgeTensor::bool(Dispatch::complex_equal(lhs, rhs.into_complex(), out_dtype))
    }

    fn not_equal(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        let lhs = lhs.into_complex();
        let out_dtype = Dispatch::complex_device(&lhs).settings().bool_dtype;
        BridgeTensor::bool(Dispatch::complex_not_equal(
            lhs,
            rhs.into_complex(),
            out_dtype,
        ))
    }

    fn equal_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let lhs = lhs.into_complex();
        let out_dtype = Dispatch::complex_device(&lhs).settings().bool_dtype;
        BridgeTensor::bool(Dispatch::complex_equal_elem(lhs, rhs.elem(), out_dtype))
    }

    fn not_equal_elem(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let lhs = lhs.into_complex();
        let out_dtype = Dispatch::complex_device(&lhs).settings().bool_dtype;
        BridgeTensor::bool(Dispatch::complex_not_equal_elem(lhs, rhs.elem(), out_dtype))
    }

    fn cat(tensors: Vec<BridgeTensor>, dim: usize) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_cat(
            BridgeTensor::into_dispatch_vec(tensors),
            dim,
        ))
    }

    fn any(tensor: BridgeTensor) -> BridgeTensor {
        let tensor = tensor.into_complex();
        let out_dtype = Dispatch::complex_device(&tensor).settings().bool_dtype;
        BridgeTensor::bool(Dispatch::complex_any(tensor, out_dtype))
    }

    fn any_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        let tensor = tensor.into_complex();
        let out_dtype = Dispatch::complex_device(&tensor).settings().bool_dtype;
        BridgeTensor::bool(Dispatch::complex_any_dim(tensor, dim, out_dtype))
    }

    fn all(tensor: BridgeTensor) -> BridgeTensor {
        let tensor = tensor.into_complex();
        let out_dtype = Dispatch::complex_device(&tensor).settings().bool_dtype;
        BridgeTensor::bool(Dispatch::complex_all(tensor, out_dtype))
    }

    fn all_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        let tensor = tensor.into_complex();
        let out_dtype = Dispatch::complex_device(&tensor).settings().bool_dtype;
        BridgeTensor::bool(Dispatch::complex_all_dim(tensor, dim, out_dtype))
    }

    fn permute(tensor: BridgeTensor, axes: &[usize]) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_permute(tensor.into_complex(), axes))
    }

    fn expand(tensor: BridgeTensor, shape: Shape) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_expand(tensor.into_complex(), shape))
    }

    fn flip(tensor: BridgeTensor, axes: &[usize]) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_flip(tensor.into_complex(), axes))
    }

    fn unfold(tensor: BridgeTensor, dim: usize, size: usize, step: usize) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_unfold(
            tensor.into_complex(),
            dim,
            size,
            step,
        ))
    }

    fn slice_assign(tensor: BridgeTensor, ranges: &[Slice], value: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_slice_assign(
            tensor.into_complex(),
            ranges,
            value.into_complex(),
        ))
    }

    fn select(tensor: BridgeTensor, dim: usize, indices: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_select(
            tensor.into_complex(),
            dim,
            indices.into(),
        ))
    }

    fn select_assign(
        tensor: BridgeTensor,
        dim: usize,
        indices: BridgeTensor,
        values: BridgeTensor,
        update: IndexingUpdateOp,
    ) -> BridgeTensor {
        match update {
            IndexingUpdateOp::Add => BridgeTensor::complex(Dispatch::complex_select_add(
                tensor.into_complex(),
                dim,
                indices.into(),
                values.into_complex(),
            )),
            other => unimplemented!("Unsupported update op {other:?}"),
        }
    }

    fn mask_where(tensor: BridgeTensor, mask: BridgeTensor, source: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_mask_where(
            tensor.into_complex(),
            mask.into(),
            source.into_complex(),
        ))
    }

    fn mask_fill(tensor: BridgeTensor, mask: BridgeTensor, value: Scalar) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_mask_fill(
            tensor.into_complex(),
            mask.into(),
            value,
        ))
    }

    fn gather(dim: usize, tensor: BridgeTensor, indices: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_gather(
            dim,
            tensor.into_complex(),
            indices.into(),
        ))
    }

    fn scatter(
        dim: usize,
        tensor: BridgeTensor,
        indices: BridgeTensor,
        values: BridgeTensor,
        update: IndexingUpdateOp,
    ) -> BridgeTensor {
        match update {
            IndexingUpdateOp::Add => BridgeTensor::complex(Dispatch::complex_scatter_add(
                dim,
                tensor.into_complex(),
                indices.into(),
                values.into_complex(),
            )),
            other => unimplemented!("Unsupported update op {other:?}"),
        }
    }

    fn scatter_nd(
        data: BridgeTensor,
        indices: BridgeTensor,
        values: BridgeTensor,
        reduction: IndexingUpdateOp,
    ) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_scatter_nd(
            data.into_complex(),
            indices.into(),
            values.into_complex(),
            reduction,
        ))
    }

    fn gather_nd(data: BridgeTensor, indices: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_gather_nd(
            data.into_complex(),
            indices.into(),
        ))
    }
}

/// Operations that are specific to complex tensors and have no analogue for real tensors.
pub trait ComplexOnlyOps {
    /// Computes the complex conjugate of each element, negating the imaginary part.
    ///
    /// # Arguments
    ///
    /// * `self` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A new tensor where each element `a + bi` is replaced by `a - bi`.
    fn conj(tensor: BridgeTensor) -> BridgeTensor;

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
    fn phase(tensor: BridgeTensor) -> BridgeTensor;

    /// Extracts the real part of each complex element.
    ///
    /// # Arguments
    ///
    /// * `self` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A real-valued tensor containing the real component of each element.
    fn real(tensor: BridgeTensor) -> BridgeTensor;

    /// Extracts the imaginary part of each complex element.
    ///
    /// # Arguments
    ///
    /// * `self` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A real-valued tensor containing the imaginary component of each element.
    fn imag(tensor: BridgeTensor) -> BridgeTensor;

    /// Computes the magnitude (absolute value) of each complex element.
    ///
    /// # Arguments
    ///
    /// * `self` - The complex tensor.
    ///
    /// # Returns
    ///
    /// A real-valued tensor containing `sqrt(re² + im²)` for each element.
    fn magnitude(tensor: BridgeTensor) -> BridgeTensor;

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
    fn from_interleaved_data(data: TensorData, device: &Device) -> Self;

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
    fn from_polar(magnitude: BridgeTensor, phase: BridgeTensor) -> Self;
}

impl ComplexOnlyOps for ComplexKind {
    fn conj(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(burn_dispatch::DispatchTensor::from(Dispatch::complex_conj(
            tensor,
        )))
    }
    fn phase(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(burn_dispatch::DispatchTensor::from(Dispatch::complex_phase(
            self.into_primitive(),
        )))
    }

    fn from_interleaved_data(data: TensorData, device: &Device) -> BridgeTensor {
        BridgeTensor::complex(burn_dispatch::DispatchTensor::from(
            Dispatch::complex_complex_from_interleaved_data(data, device.into()),
        ))
    }

    fn real(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(burn_dispatch::DispatchTensor::complex_real(
            self.into_primitive(),
        ))
    }

    fn imag(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(burn_dispatch::DispatchTensor::complex_imag(
            self.into_primitive(),
        ))
    }

    fn magnitude(tensor: BridgeTensor) -> BridgeTensor {
        Dispatch::complex_abs(self.into_primitive())
    }

    fn from_parts<T>(real: T, imag: T) -> Self
    where
        T: Into<TensorData>,
    {
        Dispatch::complex_from_parts(real.into(), imag.into())
    }
    fn from_polar(magnitude: BridgeTensor, phase: BridgeTensor) -> Self {
        Dispatch::complex_from_polar(magnitude, phase)
    }
}

impl Numeric for ComplexKind {
    fn add(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_add(
            lhs.into_complex(),
            rhs.into_complex(),
        ))
    }

    fn add_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let lhs = lhs.into_complex();
        let rhs_tensor = Dispatch::complex_full(lhs.shape(), rhs, &Dispatch::complex_device(&lhs));
        BridgeTensor::complex(Dispatch::complex_add(lhs, rhs_tensor))
    }

    fn sub(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_sub(
            lhs.into_complex(),
            rhs.into_complex(),
        ))
    }

    fn sub_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let lhs = lhs.into_complex();
        let rhs_tensor = Dispatch::complex_full(lhs.shape(), rhs, &Dispatch::complex_device(&lhs));
        BridgeTensor::complex(Dispatch::complex_sub(lhs, rhs_tensor))
    }

    fn mul(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_mul(
            lhs.into_complex(),
            rhs.into_complex(),
        ))
    }

    fn mul_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let lhs = lhs.into_complex();
        let rhs_tensor = Dispatch::complex_full(lhs.shape(), rhs, &Dispatch::complex_device(&lhs));
        BridgeTensor::complex(Dispatch::complex_mul(lhs, rhs_tensor))
    }

    fn div(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_div(
            lhs.into_complex(),
            rhs.into_complex(),
        ))
    }

    fn div_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        let lhs = lhs.into_complex();
        let rhs_tensor = Dispatch::complex_full(lhs.shape(), rhs, &Dispatch::complex_device(&lhs));
        BridgeTensor::complex(Dispatch::complex_div(lhs, rhs_tensor))
    }

    fn remainder(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_remainder(
            lhs.into_complex(),
            rhs.into_complex(),
        ))
    }

    fn remainder_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_remainder_scalar(
            lhs.into_complex(),
            rhs.elem(),
        ))
    }

    fn neg(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_neg(tensor.into_complex()))
    }

    fn sum(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_sum(tensor.into_complex()))
    }

    fn sum_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_sum_dim(tensor.into_complex(), dim))
    }

    fn prod(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_prod(tensor.into_complex()))
    }

    fn prod_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_prod_dim(tensor.into_complex(), dim))
    }

    fn mean(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_mean(tensor.into_complex()))
    }

    fn mean_dim(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_mean_dim(tensor.into_complex(), dim))
    }

    fn powi(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_powc(
            lhs.into_complex(),
            rhs.into_complex(),
        ))
    }

    fn powi_scalar(lhs: BridgeTensor, rhs: Scalar) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_powf_scalar(lhs.into_complex(), rhs))
    }

    fn matmul(lhs: BridgeTensor, rhs: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_matmul(
            lhs.into_complex(),
            rhs.into_complex(),
        ))
    }

    fn cumsum(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_cumsum(tensor.into_complex(), dim))
    }

    fn cumprod(tensor: BridgeTensor, dim: usize) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_cumprod(tensor.into_complex(), dim))
    }

    fn sign(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_sign(tensor.into_complex()))
    }

    fn random(
        shape: Shape,
        distribution: Distribution,
        device: &Device,
        dtype: DType,
    ) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_random(
            shape,
            distribution,
            device.as_dispatch(),
            dtype.into(),
        ))
    }
}

impl FloatMathOps for ComplexKind {
    fn square(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_powf_scalar(
            tensor.into_complex(),
            2.into(),
        ))
    }

    fn exp(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_exp(tensor.into_complex()))
    }

    fn log(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_log(tensor.into_complex()))
    }

    fn log1p(tensor: BridgeTensor) -> BridgeTensor {
        let tensor = tensor.into_complex();
        let dtype = tensor.dtype().into();
        let device = Dispatch::complex_device(&tensor);
        let shape = tensor.shape();
        let ones = Dispatch::complex_ones(shape, &device, dtype);
        BridgeTensor::complex(Dispatch::complex_log(Dispatch::complex_add(tensor, ones)))
    }

    fn sqrt(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_sqrt(tensor.into_complex()))
    }

    fn cos(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_cos(tensor.into_complex()))
    }

    fn sin(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_sin(tensor.into_complex()))
    }

    fn tan(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_tan(tensor.into_complex()))
    }

    fn cosh(tensor: BridgeTensor) -> BridgeTensor {
        let tensor = tensor.into_complex();
        let device = Dispatch::complex_device(&tensor);
        let shape = tensor.shape();
        let two = Dispatch::complex_full(shape, Scalar::from(2.0_f32), &device);
        let exp_z = Dispatch::complex_exp(tensor.clone());
        let exp_neg_z = Dispatch::complex_exp(Dispatch::complex_neg(tensor));
        BridgeTensor::complex(Dispatch::complex_div(
            Dispatch::complex_add(exp_z, exp_neg_z),
            two,
        ))
    }

    fn sinh(tensor: BridgeTensor) -> BridgeTensor {
        let tensor = tensor.into_complex();
        let device = Dispatch::complex_device(&tensor);
        let shape = tensor.shape();
        let two = Dispatch::complex_full(shape, Scalar::from(2.0_f32), &device);
        let exp_z = Dispatch::complex_exp(tensor.clone());
        let exp_neg_z = Dispatch::complex_exp(Dispatch::complex_neg(tensor));
        BridgeTensor::complex(Dispatch::complex_div(
            Dispatch::complex_sub(exp_z, exp_neg_z),
            two,
        ))
    }

    fn tanh(tensor: BridgeTensor) -> BridgeTensor {
        let tensor = tensor.into_complex();
        let device = Dispatch::complex_device(&tensor);
        let shape = tensor.shape();
        let dtype = tensor.dtype().into();
        let ones = Dispatch::complex_ones(shape, &device, dtype);
        let two_z = Dispatch::complex_add(tensor.clone(), tensor);
        let e2z = Dispatch::complex_exp(two_z);
        let numerator = Dispatch::complex_sub(e2z.clone(), ones.clone());
        let denominator = Dispatch::complex_add(e2z, ones);
        BridgeTensor::complex(Dispatch::complex_div(numerator, denominator))
    }

    fn acos(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_acos(tensor.into_complex()))
    }

    fn acosh(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_acosh(tensor.into_complex()))
    }

    fn asin(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_asin(tensor.into_complex()))
    }

    fn asinh(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_asinh(tensor.into_complex()))
    }

    fn atan(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_atan(tensor.into_complex()))
    }

    fn atanh(tensor: BridgeTensor) -> BridgeTensor {
        BridgeTensor::complex(Dispatch::complex_atanh(tensor.into_complex()))
    }
}
