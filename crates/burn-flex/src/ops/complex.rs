use alloc::vec::Vec;
use burn_backend::tensor::{BoolTensor, Device, FloatTensor, IntTensor};
use burn_backend::{Complex, Distribution};
use burn_backend::{Element, TensorData};
use burn_complex::base::{CBT, SplitTensorData};
use burn_complex::utils::interleaved_data_to_imag_data;
use burn_complex::{
    base::{ComplexTensor, ComplexTensorBackend, ComplexTensorOps, InterleavedLayout},
    utils::{
        interleave_from_split_data, interleaved_data_from_imag_data,
        interleaved_data_from_real_data, interleaved_data_to_real_data,
        split_from_interleaved_data,
    },
};
use burn_std::{BoolDType, DType, FloatDType, Slice};
use num_traits::ToPrimitive;
use num_traits::Zero;

use crate::ops::binary::make_tensor;
use crate::ops::binary::{
    binary_op_typed_convert, can_use_binary_inplace, scalar_op_typed_convert,
};
use crate::ops::comparison::compare_elem_typed;
use crate::ops::{
    binary::scalar_op_typed_rhs,
    comparison::{bool_scalar, compare_typed, iter_elements, reduce_bool_dim},
};

use crate::strided_index::StridedIter;
use crate::{Flex, FlexDevice, FlexTensor, ops::binary::scalar_op_typed};
impl CBT for Flex {
    type ComplexTensorPrimitive = FlexTensor;
}
impl ComplexTensorBackend for Flex {
    type InnerBackend = Flex;

    type ComplexScalar = Complex<f32>;

    type Layout = InterleavedLayout;

    fn complex_from_real_data(data: TensorData, _device: &Device<Self>) -> ComplexTensor<Self> {
        let interleaved_data = interleaved_data_from_real_data(data);

        FlexTensor::from_data(interleaved_data)
    }

    fn complex_from_imag_data(data: TensorData, _device: &Device<Self>) -> ComplexTensor<Self> {
        let interleaved_data = interleaved_data_from_imag_data(data);

        FlexTensor::from_data(interleaved_data)
    }

    fn complex_from_interleaved_data(
        data: TensorData,
        _device: &Self::Device,
    ) -> ComplexTensor<Self> {
        FlexTensor::from_data(data)
    }

    fn complex_from_parts_data(
        real_data: TensorData,
        imag_data: TensorData,
        _device: &Self::Device,
    ) -> ComplexTensor<Self> {
        let interleaved_data = interleave_from_split_data(real_data, imag_data);
        FlexTensor::from_data(interleaved_data)
    }
}

impl ComplexTensorOps<Flex> for Flex {
    async fn complex_into_real_data(
        tensor: ComplexTensor<Flex>,
    ) -> Result<TensorData, burn_backend::ExecutionError> {
        Ok(interleaved_data_to_real_data(tensor.into_data()))
    }

    async fn complex_into_imag_data(
        tensor: ComplexTensor<Flex>,
    ) -> Result<TensorData, burn_backend::ExecutionError> {
        Ok(interleaved_data_to_imag_data(tensor.into_data()))
    }

    async fn complex_into_interleaved_data(
        tensor: ComplexTensor<Flex>,
    ) -> Result<TensorData, burn_backend::ExecutionError> {
        Ok(tensor.into_data())
    }

    async fn complex_into_split_data(
        tensor: ComplexTensor<Flex>,
    ) -> Result<SplitTensorData, burn_backend::ExecutionError> {
        Ok(split_from_interleaved_data(tensor.into_data()))
    }

    fn to_complex(tensor: FloatTensor<Flex>) -> ComplexTensor<Flex> {
        let interleaved_data = interleaved_data_from_real_data(tensor.into_data());
        FlexTensor::from_data(interleaved_data)
    }

    fn complex_squared_norm(tensor: ComplexTensor<Flex>) -> FloatTensor<Flex> {
        crate::c2r_unary_op!(tensor, |a| a.norm_sqr())
    }

    fn complex_device(_tensor: &ComplexTensor<Flex>) -> Device<Flex> {
        Default::default()
    }

    fn complex_to_device(
        tensor: ComplexTensor<Flex>,
        _device: &Device<Flex>,
    ) -> ComplexTensor<Flex> {
        tensor
    }

    fn complex_add(lhs: ComplexTensor<Flex>, rhs: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_binary_op!(lhs, rhs, |a, b| a + b)
    }

    fn complex_sub(lhs: ComplexTensor<Flex>, rhs: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_binary_op!(lhs, rhs, |a, b| a - b)
    }

    fn complex_mul(lhs: ComplexTensor<Flex>, rhs: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_binary_op!(lhs, rhs, |a, b| a * b)
    }

    fn complex_div(lhs: ComplexTensor<Flex>, rhs: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_binary_op!(lhs, rhs, |a, b| a / b)
    }

    fn real(tensor: ComplexTensor<Flex>) -> FloatTensor<Flex> {
        crate::c2r_unary_op!(tensor, |a| a.real)
    }

    fn imag(tensor: ComplexTensor<Flex>) -> FloatTensor<Flex> {
        crate::c2r_unary_op!(tensor, |a| a.imag)
    }

    fn abs(tensor: ComplexTensor<Flex>) -> FloatTensor<Flex> {
        crate::c2r_unary_op!(tensor, |a| a.norm())
    }

    fn complex_from_parts(real: TensorData, imag: TensorData) -> ComplexTensor<Flex> {
        <Flex as ComplexTensorBackend>::complex_from_parts_data(real, imag, &FlexDevice)
    }

    fn complex_from_polar(
        magnitude: FloatTensor<Flex>,
        phase: FloatTensor<Flex>,
    ) -> ComplexTensor<Flex> {
        // Convert polar to cartesian: magnitude * e^(i*phase) = magnitude * (cos(phase) + i*sin(phase))
        let real_part = crate::ops::binary::binary_op(
            magnitude.clone(),
            crate::ops::unary::cos(phase.clone()),
            |m, cos_p| m * cos_p,
            |m, cos_p| m * cos_p,
            None,
        );
        let imag_part = crate::ops::binary::binary_op(
            magnitude,
            crate::ops::unary::sin(phase),
            |m, sin_p| m * sin_p,
            |m, sin_p| m * sin_p,
            None,
        );
        Self::complex_from_parts(real_part.into_data(), imag_part.into_data())
    }

    fn complex_exp(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_unary_op!(tensor, |a| a.exp())
    }

    fn complex_log(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_unary_op!(tensor, |a| a.ln())
    }

    fn complex_not_equal_elem(
        lhs: ComplexTensor<Flex>,
        rhs: <Flex as ComplexTensorBackend>::ComplexScalar,
        out_dtype: BoolDType,
    ) -> BoolTensor<Flex> {
        let f32_cmp = |a, b| a != b;
        let f64_cmp = |a, b| a != b;
        //let _simd_hint = Some(CompareOp::Ne);

        let dtype = lhs.dtype();

        match dtype {
            DType::Complex32 => compare_elem_typed(lhs, rhs, out_dtype, f32_cmp),
            DType::Complex64 => compare_elem_typed(lhs, rhs, out_dtype, f64_cmp),
            _ => panic!("compare: unsupported dtype {:?}", dtype),
        }
    }

    fn complex_equal_elem(
        lhs: ComplexTensor<Flex>,
        rhs: <Flex as ComplexTensorBackend>::ComplexScalar,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        let f32_cmp = |a, b| a == b;
        let f64_cmp = |a, b| a == b;
        //let _simd_hint = Some(CompareOp::Eq);

        let dtype = lhs.dtype();

        match dtype {
            DType::Complex32 => compare_elem_typed(lhs, rhs, out_dtype, f32_cmp),
            DType::Complex64 => compare_elem_typed(lhs, rhs, out_dtype, f64_cmp),
            _ => panic!("compare: unsupported dtype {:?}", dtype),
        }
    }

    fn complex_equal(
        lhs: ComplexTensor<Flex>,
        rhs: ComplexTensor<Flex>,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        let f32_cmp = |a, b| a == b;
        let f64_cmp = |a, b| a == b;
        //let _simd_hint = Some(CompareOp::Eq);
        debug_assert_eq!(lhs.dtype(), rhs.dtype(), "compare: dtype mismatch");

        // Broadcast to same shape if needed
        let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);

        let dtype = lhs.dtype();

        match dtype {
            DType::Complex32 => compare_typed::<Complex<f32>, _>(lhs, &rhs, out_dtype, f32_cmp),
            DType::Complex64 => compare_typed::<Complex<f64>, _>(lhs, &rhs, out_dtype, f64_cmp),
            _ => panic!("compare: unsupported dtype {:?}", dtype),
        }
    }

    fn complex_not_equal(
        lhs: ComplexTensor<Flex>,
        rhs: ComplexTensor<Flex>,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        let lhs = lhs;
        let rhs = rhs;
        let f32_cmp = |a, b| a != b;
        let f64_cmp = |a, b| a != b;
        //let _simd_hint = Some(CompareOp::Ne);
        debug_assert_eq!(lhs.dtype(), rhs.dtype(), "compare: dtype mismatch");

        // Broadcast to same shape if needed
        let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);

        let dtype = lhs.dtype();

        match dtype {
            DType::Complex32 => compare_typed::<Complex<f32>, _>(lhs, &rhs, out_dtype, f32_cmp),
            DType::Complex64 => compare_typed::<Complex<f64>, _>(lhs, &rhs, out_dtype, f64_cmp),
            _ => panic!("compare: unsupported dtype {:?}", dtype),
        }
    }

    fn complex_gather(
        dim: usize,
        tensor: ComplexTensor<Flex>,
        indices: IntTensor<Flex>,
    ) -> ComplexTensor<Flex> {
        match tensor.dtype() {
            DType::Complex32 => {
                crate::ops::gather_scatter::gather::<Complex<f32>>(tensor, dim, indices)
            }
            DType::Complex64 => {
                crate::ops::gather_scatter::gather::<Complex<f64>>(tensor, dim, indices)
            }
            _ => panic!("complex_gather: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn complex_scatter_add(
        dim: usize,
        tensor: ComplexTensor<Flex>,
        indices: IntTensor<Flex>,
        values: ComplexTensor<Flex>,
    ) -> ComplexTensor<Flex> {
        match tensor.dtype() {
            DType::Complex32 => crate::ops::gather_scatter::scatter_add::<Complex<f32>>(
                tensor, dim, indices, values,
            ),
            DType::Complex64 => crate::ops::gather_scatter::scatter_add::<Complex<f64>>(
                tensor, dim, indices, values,
            ),
            _ => panic!(
                "complex_scatter_add: unsupported dtype {:?}",
                tensor.dtype()
            ),
        }
    }

    fn complex_random(
        shape: burn_std::Shape,
        distribution: Distribution,
        _device: &Device<Flex>,
        dtype: FloatDType,
    ) -> ComplexTensor<Flex> {
        let mut seed = crate::backend::SEED.lock().unwrap();
        let mut rng = seed.take().unwrap_or_else(crate::backend::get_seeded_rng);
        let data = match dtype {
            FloatDType::F32 => {
                TensorData::random::<Complex<f32>, _, _>(shape, distribution, &mut rng)
            }
            FloatDType::F64 => {
                TensorData::random::<Complex<f64>, _, _>(shape, distribution, &mut rng)
            }
            _ => panic!("select: unsupported dtype {:?}", dtype),
        };
        *seed = Some(rng);
        FlexTensor::from_data(data)
    }

    fn complex_reshape(tensor: ComplexTensor<Flex>, shape: burn_std::Shape) -> ComplexTensor<Flex> {
        tensor.reshape(shape)
    }

    fn complex_transpose(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        let ndims = tensor.layout().num_dims();
        if ndims < 2 {
            return tensor;
        }
        tensor.transpose(ndims - 2, ndims - 1)
    }

    fn complex_neg(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_unary_op!(tensor, |a| -a)
    }

    fn conj(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_unary_op!(tensor, |a| a.conj())
    }

    fn complex_arg(tensor: ComplexTensor<Flex>) -> FloatTensor<Flex> {
        crate::c2r_unary_op!(tensor, |a| a.arg())
    }

    fn complex_powc(lhs: ComplexTensor<Flex>, rhs: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::ops::complex::c2c_binary_op(
            lhs,
            rhs,
            |a: Complex<f32>, b: Complex<f32>| a.powc(b),
            |a: Complex<f64>, b: Complex<f64>| a.powc(b),
        )
    }

    fn complex_sqrt(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_unary_op!(tensor, |a| a.sqrt())
    }

    fn complex_sin(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_unary_op!(tensor, |a| a.sin())
    }

    fn complex_cos(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_unary_op!(tensor, |a| a.cos())
    }

    fn complex_tan(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_unary_op!(tensor, |a| a.tan())
    }

    fn complex_select(
        tensor: ComplexTensor<Flex>,
        dim: usize,
        indices: IntTensor<Flex>,
    ) -> ComplexTensor<Flex> {
        match tensor.dtype() {
            DType::Complex32 => {
                crate::ops::gather_scatter::select::<Complex<f32>>(tensor, dim, indices)
            }
            DType::Complex64 => {
                crate::ops::gather_scatter::select::<Complex<f64>>(tensor, dim, indices)
            }
            _ => panic!("select: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn complex_select_add(
        tensor: ComplexTensor<Flex>,
        dim: usize,
        indices: IntTensor<Flex>,
        values: ComplexTensor<Flex>,
    ) -> ComplexTensor<Flex> {
        match tensor.dtype() {
            DType::Complex32 => {
                crate::ops::gather_scatter::select_add::<Complex<f32>>(tensor, dim, indices, values)
            }
            DType::Complex64 => {
                crate::ops::gather_scatter::select_add::<Complex<f64>>(tensor, dim, indices, values)
            }
            _ => panic!("complex_select_add: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn complex_slice(tensor: ComplexTensor<Flex>, slices: &[Slice]) -> ComplexTensor<Flex> {
        crate::ops::slice::slice(tensor, slices)
    }

    fn complex_slice_assign(
        tensor: ComplexTensor<Flex>,
        ranges: &[Slice],
        value: ComplexTensor<Flex>,
    ) -> ComplexTensor<Flex> {
        crate::ops::slice::slice_assign(tensor, ranges, value)
    }

    fn complex_swap_dims(
        tensor: ComplexTensor<Flex>,
        dim1: usize,
        dim2: usize,
    ) -> ComplexTensor<Flex> {
        tensor.transpose(dim1, dim2)
    }

    fn complex_repeat_dim(
        tensor: ComplexTensor<Flex>,
        dim: usize,
        times: usize,
    ) -> ComplexTensor<Flex> {
        crate::ops::repeat_dim::repeat_dim(tensor, dim, times)
    }

    fn complex_cat(tensors: Vec<ComplexTensor<Flex>>, dim: usize) -> ComplexTensor<Flex> {
        crate::ops::cat::cat(tensors, dim)
    }

    fn complex_any(tensor: ComplexTensor<Flex>, out_dtype: BoolDType) -> BoolTensor<Flex> {
        let has_any = match tensor.dtype() {
            DType::Complex32 => {
                iter_elements::<Complex<f32>>(&tensor).any(|x| x.real != 0.0 || x.imag != 0.0)
            }
            DType::Complex64 => {
                iter_elements::<Complex<f64>>(&tensor).any(|x| x.real != 0.0 || x.imag != 0.0)
            }
            _ => panic!("any_float: unsupported dtype {:?}", tensor.dtype()),
        };
        bool_scalar(has_any, out_dtype)
    }

    fn complex_any_dim(
        tensor: ComplexTensor<Flex>,
        dim: usize,
        out_dtype: BoolDType,
    ) -> BoolTensor<Flex> {
        reduce_bool_dim(&tensor, dim, false, |a, b| a || b, out_dtype)
    }

    fn complex_all(tensor: ComplexTensor<Flex>, out_dtype: BoolDType) -> BoolTensor<Flex> {
        let all = match tensor.dtype() {
            DType::Complex32 => {
                iter_elements::<Complex<f32>>(&tensor).all(|x| x.real != 0.0 || x.imag != 0.0)
            }
            DType::Complex64 => {
                iter_elements::<Complex<f64>>(&tensor).all(|x| x.real != 0.0 || x.imag != 0.0)
            }

            _ => panic!("all_float: unsupported dtype {:?}", tensor.dtype()),
        };
        bool_scalar(all, out_dtype)
    }

    fn complex_all_dim(
        tensor: ComplexTensor<Flex>,
        dim: usize,
        out_dtype: BoolDType,
    ) -> BoolTensor<Flex> {
        reduce_bool_dim(&tensor, dim, true, |a, b| a && b, out_dtype)
    }

    fn complex_permute(tensor: ComplexTensor<Flex>, axes: &[usize]) -> ComplexTensor<Flex> {
        tensor.permute(axes)
    }

    fn complex_expand(tensor: ComplexTensor<Flex>, shape: burn_std::Shape) -> ComplexTensor<Flex> {
        crate::ops::expand::expand(tensor, shape)
    }

    fn complex_flip(tensor: ComplexTensor<Flex>, axes: &[usize]) -> ComplexTensor<Flex> {
        crate::ops::flip::flip(tensor, axes)
    }

    fn complex_unfold(
        tensor: ComplexTensor<Flex>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> ComplexTensor<Flex> {
        crate::ops::unfold::unfold(tensor, dim, size, step)
    }

    fn complex_sum(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::ops::reduce::sum(tensor)
    }

    fn complex_sum_dim(tensor: ComplexTensor<Flex>, dim: usize) -> ComplexTensor<Flex> {
        crate::ops::reduce::sum_dim(tensor, dim)
    }

    fn complex_prod(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::ops::reduce::prod(tensor)
    }

    fn complex_prod_dim(tensor: ComplexTensor<Flex>, dim: usize) -> ComplexTensor<Flex> {
        crate::ops::reduce::prod_dim(tensor, dim)
    }

    fn complex_mean(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::ops::reduce::mean(tensor)
    }

    fn complex_mean_dim(tensor: ComplexTensor<Flex>, dim: usize) -> ComplexTensor<Flex> {
        crate::ops::reduce::mean_dim(tensor, dim)
    }

    fn complex_remainder(
        lhs: ComplexTensor<Flex>,
        rhs: ComplexTensor<Flex>,
    ) -> ComplexTensor<Flex> {
        crate::c2c_binary_op!(lhs, rhs, |a, b| a % b)
    }

    fn complex_remainder_scalar(
        lhs: ComplexTensor<Flex>,
        rhs: <Flex as ComplexTensorBackend>::ComplexScalar,
    ) -> ComplexTensor<Flex> {
        let dtype = lhs.dtype();
        match dtype {
            DType::Complex32 => scalar_op_typed::<Complex<f32>, _>(lhs, rhs, |a, b| a % b),
            DType::Complex64 => {
                let rhs64 = Complex::<f64> {
                    real: rhs.real as f64,
                    imag: rhs.imag as f64,
                };
                scalar_op_typed::<Complex<f64>, _>(lhs, rhs64, |a, b| a % b)
            }
            _ => panic!("complex_remainder_scalar: unsupported dtype {:?}", dtype),
        }
    }

    fn complex_mask_where(
        tensor: ComplexTensor<Flex>,
        mask: BoolTensor<Flex>,
        source: ComplexTensor<Flex>,
    ) -> ComplexTensor<Flex> {
        match tensor.dtype() {
            DType::Complex32 => crate::ops::mask::mask_where::<Complex<f32>>(tensor, mask, source),
            DType::Complex64 => crate::ops::mask::mask_where::<Complex<f64>>(tensor, mask, source),
            _ => panic!("complex_mask_where: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn complex_mask_fill(
        tensor: ComplexTensor<Flex>,
        mask: BoolTensor<Flex>,
        value: <Flex as ComplexTensorBackend>::ComplexScalar,
    ) -> ComplexTensor<Flex> {
        match tensor.dtype() {
            DType::Complex32 => crate::ops::mask::mask_fill::<Complex<f32>>(tensor, mask, value),
            DType::Complex64 => {
                let value64 = Complex::<f64> {
                    real: value.real as f64,
                    imag: value.imag as f64,
                };
                crate::ops::mask::mask_fill::<Complex<f64>>(tensor, mask, value64)
            }
            _ => panic!("complex_mask_fill: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn complex_sign(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_unary_op!(tensor, |a| {
            if a == Complex::zero() {
                Complex::zero()
            } else {
                let norm = a.norm();
                Complex {
                    real: a.real / norm,
                    imag: a.imag / norm,
                }
            }
        })
    }

    fn complex_matmul(lhs: ComplexTensor<Flex>, rhs: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::ops::matmul::matmul(lhs, rhs)
    }

    fn complex_cumsum(tensor: ComplexTensor<Flex>, dim: usize) -> ComplexTensor<Flex> {
        match tensor.dtype() {
            DType::Complex32 => crate::ops::cumulative::cumsum::<Complex<f32>>(tensor, dim),
            DType::Complex64 => crate::ops::cumulative::cumsum::<Complex<f64>>(tensor, dim),
            _ => panic!("complex_cumsum: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn complex_cumprod(tensor: ComplexTensor<Flex>, dim: usize) -> ComplexTensor<Flex> {
        match tensor.dtype() {
            DType::Complex32 => crate::ops::cumulative::cumprod::<Complex<f32>>(tensor, dim),
            DType::Complex64 => crate::ops::cumulative::cumprod::<Complex<f64>>(tensor, dim),
            _ => panic!("complex_cumprod: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn complex_powc_scalar(
        lhs: ComplexTensor<Flex>,
        rhs: <Flex as ComplexTensorBackend>::ComplexScalar,
    ) -> ComplexTensor<Flex> {
        match lhs.dtype() {
            DType::Complex32 => scalar_op_typed::<Complex<f32>, _>(lhs, rhs, |a, b| a.powc(b)),
            DType::Complex64 => {
                let rhs64 = Complex::<f64> {
                    real: rhs.real as f64,
                    imag: rhs.imag as f64,
                };
                scalar_op_typed::<Complex<f64>, _>(lhs, rhs64, |a, b| a.powc(b))
            }
            _ => panic!("complex_powc_scalar: unsupported dtype {:?}", lhs.dtype()),
        }
    }

    fn complex_powf(lhs: ComplexTensor<Flex>, rhs: FloatTensor<Flex>) -> ComplexTensor<Flex> {
        match lhs.dtype() {
            DType::Complex32 => {
                let rhs = if rhs.dtype() == DType::F32 {
                    rhs
                } else {
                    FlexTensor::from_data(rhs.into_data().convert_dtype(DType::F32))
                };
                binary_op_typed_rhs::<Complex<f32>, f32, _>(lhs, &rhs, |a, b| a.powf(b))
            }
            DType::Complex64 => {
                let rhs = if rhs.dtype() == DType::F64 {
                    rhs
                } else {
                    FlexTensor::from_data(rhs.into_data().convert_dtype(DType::F64))
                };
                binary_op_typed_rhs::<Complex<f64>, f64, _>(lhs, &rhs, |a, b| a.powf(b))
            }
            _ => panic!("complex_powf: unsupported dtype {:?}", lhs.dtype()),
        }
    }

    fn complex_powf_scalar(
        lhs: ComplexTensor<Flex>,
        rhs: burn_backend::element::Scalar,
    ) -> ComplexTensor<Flex> {
        match lhs.dtype() {
            DType::Complex32 => {
                let rhs_f32 = rhs
                    .to_f32()
                    .expect("complex_powf_scalar: rhs must be a float scalar");
                scalar_op_typed_rhs::<Complex<f32>, f32, _>(lhs, rhs_f32, |a, b| a.powf(b))
            }
            DType::Complex64 => {
                let rhs_f64 = rhs
                    .to_f64()
                    .expect("complex_powf_scalar: rhs must be a float scalar");
                scalar_op_typed_rhs::<Complex<f64>, f64, _>(lhs, rhs_f64, |a, b| a.powf(b))
            }
            _ => panic!("complex_powf_scalar: unsupported dtype {:?}", lhs.dtype()),
        }
    }

    fn complex_scatter_nd(
        tensor: ComplexTensor<Flex>,
        indices: IntTensor<Flex>,
        value: ComplexTensor<Flex>,
        reduction: burn_backend::tensor::IndexingUpdateOp,
    ) -> ComplexTensor<Flex> {
        match tensor.dtype() {
            DType::Complex32 => crate::ops::gather_scatter::scatter_nd_no_ord::<Complex<f32>>(
                tensor, indices, value, reduction,
            ),
            DType::Complex64 => crate::ops::gather_scatter::scatter_nd_no_ord::<Complex<f64>>(
                tensor, indices, value, reduction,
            ),
            _ => panic!("complex_scatter_nd: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn complex_acos(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_unary_op!(tensor, |a| a.acos())
    }

    fn complex_acosh(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_unary_op!(tensor, |a| a.acosh())
    }

    fn complex_asin(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_unary_op!(tensor, |a| a.asin())
    }

    fn complex_asinh(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_unary_op!(tensor, |a| a.asinh())
    }

    fn complex_atan(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_unary_op!(tensor, |a| a.atan())
    }

    fn complex_atanh(tensor: ComplexTensor<Flex>) -> ComplexTensor<Flex> {
        crate::c2c_unary_op!(tensor, |a| a.atanh())
    }

    fn complex_gather_nd(
        data: ComplexTensor<Flex>,
        indices: IntTensor<Flex>,
    ) -> ComplexTensor<Flex> {
        match data.dtype() {
            DType::Complex32 => {
                crate::ops::gather_scatter::gather_nd::<Complex<f32>>(data, indices)
            }
            DType::Complex64 => {
                crate::ops::gather_scatter::gather_nd::<Complex<f64>>(data, indices)
            }
            _ => panic!("complex_gather_nd: unsupported dtype {:?}", data.dtype()),
        }
    }
}

/// Check if any element is non-zero (complex tensors).
pub fn any_complex(tensor: FlexTensor, out_dtype: BoolDType) -> FlexTensor {
    let has_any = match tensor.dtype() {
        DType::Complex32 => {
            iter_elements::<Complex<f32>>(&tensor).any(|x| x != Complex::<f32>::new(0.0, 0.0))
        }
        DType::Complex64 => {
            iter_elements::<Complex<f64>>(&tensor).any(|x| x != Complex::<f64>::new(0.0, 0.0))
        }
        _ => panic!("any_complex: unsupported dtype {:?}", tensor.dtype()),
    };
    bool_scalar(has_any, out_dtype)
}

#[macro_export]
macro_rules! c2c_binary_op {
    ($lhs:expr, $rhs:expr, $op:expr) => {
        $crate::ops::complex::c2c_binary_op(
            $lhs,
            $rhs,
            |a: Complex<f32>, b: Complex<f32>| $op(a, b),
            |a: Complex<f64>, b: Complex<f64>| $op(a, b),
        )
    };
}

#[macro_export]
macro_rules! c2c_scalar_op {
    ($tensor:expr, $scalar:expr, $op:expr) => {
        $crate::ops::complex::c2c_scalar_op(
            $tensor,
            $scalar,
            |a: Complex<f32>, b: Complex<f32>| $op(a, b),
            |a: Complex<f64>, b: Complex<f64>| $op(a, b),
        )
    };
}

#[macro_export]
macro_rules! c2r_binary_op {
    ($lhs:expr, $rhs:expr, $op:expr) => {
        $crate::ops::complex::c2r_binary_op(
            $lhs,
            $rhs,
            |a: Complex<f32>, b: Complex<f32>| $op(a, b),
            |a: Complex<f64>, b: Complex<f64>| $op(a, b),
        )
    };
}

#[macro_export]
macro_rules! c2r_unary_op {
    ($tensor:expr, $op:expr) => {
        $crate::ops::complex::c2r_unary_op($tensor, $op, $op)
    };
}

#[macro_export]
macro_rules! c2c_unary_op {
    ($tensor:expr, $op:expr) => {
        $crate::ops::complex::c2c_unary_op($tensor, $op, $op)
    };
}

#[macro_export]
macro_rules! c2r_scalar_op {
    ($tensor:expr, $scalar:expr, $op:expr) => {
        $crate::ops::complex::c2r_scalar_op(
            $tensor,
            $scalar,
            |a: Complex<f32>, b: Complex<f32>| $op(a, b),
            |a: Complex<f64>, b: Complex<f64>| $op(a, b),
        )
    };
}

pub fn c2c_binary_op<F32Op, F64Op>(
    lhs: FlexTensor,
    rhs: FlexTensor,
    f32_op: F32Op,
    f64_op: F64Op,
) -> FlexTensor
where
    F32Op: Fn(Complex<f32>, Complex<f32>) -> Complex<f32> + Copy,
    F64Op: Fn(Complex<f64>, Complex<f64>) -> Complex<f64> + Copy,
{
    use crate::ops::binary::binary_op_typed;

    debug_assert_eq!(
        lhs.dtype(),
        rhs.dtype(),
        "complex_binary_op: dtype mismatch"
    );

    let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);

    match lhs.dtype() {
        DType::Complex32 => binary_op_typed::<Complex<f32>, _>(lhs, &rhs, f32_op),
        DType::Complex64 => binary_op_typed::<Complex<f64>, _>(lhs, &rhs, f64_op),
        _ => panic!("complex_binary_op: unsupported dtype {:?}", lhs.dtype()),
    }
}

pub fn c2c_scalar_op<F32Op, F64Op>(
    tensor: FlexTensor,
    scalar: Complex<f64>,
    f32_op: F32Op,
    f64_op: F64Op,
) -> FlexTensor
where
    F32Op: Fn(Complex<f32>, Complex<f32>) -> Complex<f32> + Copy,
    F64Op: Fn(Complex<f64>, Complex<f64>) -> Complex<f64> + Copy,
{
    match tensor.dtype() {
        DType::Complex32 => {
            let s = Complex {
                real: scalar.real as f32,
                imag: scalar.imag as f32,
            };
            scalar_op_typed::<Complex<f32>, _>(tensor, s, f32_op)
        }
        DType::Complex64 => scalar_op_typed::<Complex<f64>, _>(tensor, scalar, f64_op),
        _ => panic!("complex_scalar_op: unsupported dtype {:?}", tensor.dtype()),
    }
}

pub fn c2r_binary_op<F32Op, F64Op>(
    lhs: FlexTensor,
    rhs: FlexTensor,
    f32_op: F32Op,
    f64_op: F64Op,
) -> FlexTensor
where
    F32Op: Fn(Complex<f32>, Complex<f32>) -> f32 + Copy,
    F64Op: Fn(Complex<f64>, Complex<f64>) -> f64 + Copy,
{
    debug_assert_eq!(
        lhs.dtype(),
        rhs.dtype(),
        "complex_binary_op: dtype mismatch"
    );

    let (lhs, rhs) = crate::ops::expand::broadcast_binary(lhs, rhs);

    match lhs.dtype() {
        DType::Complex32 => binary_op_typed_convert::<Complex<f32>, f32, _>(lhs, &rhs, f32_op),
        DType::Complex64 => binary_op_typed_convert::<Complex<f64>, f64, _>(lhs, &rhs, f64_op),
        _ => panic!("complex_binary_op: unsupported dtype {:?}", lhs.dtype()),
    }
}

pub fn c2r_scalar_op<F32Op, F64Op>(
    tensor: FlexTensor,
    scalar: Complex<f64>,
    f32_op: F32Op,
    f64_op: F64Op,
) -> FlexTensor
where
    F32Op: Fn(Complex<f32>, Complex<f32>) -> f32 + Copy,
    F64Op: Fn(Complex<f64>, Complex<f64>) -> f64 + Copy,
{
    match tensor.dtype() {
        DType::Complex32 => {
            let s = Complex {
                real: scalar.real as f32,
                imag: scalar.imag as f32,
            };
            scalar_op_typed_convert(tensor, s, f32_op)
        }
        DType::Complex64 => scalar_op_typed_convert(tensor, scalar, f64_op),
        _ => panic!("complex_scalar_op: unsupported dtype {:?}", tensor.dtype()),
    }
}

pub fn c2r_unary_op<F32Op, F64Op>(tensor: FlexTensor, f32_op: F32Op, f64_op: F64Op) -> FlexTensor
where
    F32Op: Fn(Complex<f32>) -> f32,
    F64Op: Fn(Complex<f64>) -> f64,
{
    use crate::ops::unary::unary_op_typed_convert;

    match tensor.dtype() {
        DType::Complex32 => unary_op_typed_convert(tensor, f32_op),
        DType::Complex64 => unary_op_typed_convert(tensor, f64_op),
        _ => panic!("c2r_unary_op: unsupported dtype {:?}", tensor.dtype()),
    }
}

pub fn c2c_unary_op<F32Op, F64Op>(tensor: FlexTensor, f32_op: F32Op, f64_op: F64Op) -> FlexTensor
where
    F32Op: Fn(Complex<f32>) -> Complex<f32>,
    F64Op: Fn(Complex<f64>) -> Complex<f64>,
{
    use crate::ops::unary::unary_op_typed;

    match tensor.dtype() {
        DType::Complex32 => unary_op_typed(tensor, f32_op),
        DType::Complex64 => unary_op_typed(tensor, f64_op),
        _ => panic!("c2c_unary_op: unsupported dtype {:?}", tensor.dtype()),
    }
}

/// Variant of binary_op that allows a different type for the rhs. Right now only needed for one function, and kind of situation unlikely to be
/// common, if it's useful outside of complex_powf, I'll move it back to binary
fn binary_op_typed_rhs<E1, E2, Op>(mut lhs: FlexTensor, rhs: &FlexTensor, op: Op) -> FlexTensor
where
    E1: Element + bytemuck::Pod,
    E2: Element + bytemuck::Pod,
    Op: Fn(E1, E2) -> E1,
{
    let rhs_storage: &[E2] = rhs.storage();

    // In-place fast path: lhs unique, contiguous at offset 0, rhs contiguous
    if let Some((l_end, r_start, r_end)) = can_use_binary_inplace(&lhs, rhs) {
        let lhs_storage: &mut [E1] = lhs.storage_mut();
        let r_slice = &rhs_storage[r_start..r_end];
        for (l, &r) in lhs_storage[..l_end].iter_mut().zip(r_slice) {
            *l = op(*l, r);
        }
        return lhs;
    }

    // Use the convert version for non-in-place operations
    binary_op_typed_convert_rhs(lhs, rhs, op)
}

/// Generic binary operation that converts between element types E -> O.
pub(crate) fn binary_op_typed_convert_rhs<E1, E2, O, Op>(
    lhs: FlexTensor,
    rhs: &FlexTensor,
    op: Op,
) -> FlexTensor
where
    E1: Element + bytemuck::Pod,
    E2: Element + bytemuck::Pod,
    O: Element + bytemuck::Pod,
    Op: Fn(E1, E2) -> O,
{
    let shape = lhs.layout().shape().clone();
    let lhs_storage: &[E1] = lhs.storage();
    let rhs_storage: &[E2] = rhs.storage();
    let lhs_layout = lhs.layout();
    let rhs_layout = rhs.layout();
    // Try contiguous path first
    let result = {
        match (
            lhs_layout.contiguous_offsets(),
            rhs_layout.contiguous_offsets(),
        ) {
            // Both contiguous
            (Some((l_start, l_end)), Some((r_start, r_end))) => {
                let l_slice = &lhs_storage[l_start..l_end];
                let r_slice = &rhs_storage[r_start..r_end];

                l_slice
                    .iter()
                    .zip(r_slice)
                    .map(|(&a, &b)| op(a, b))
                    .collect()
            }
            _ => {
                // Fast path for 2D non-contiguous (common for transpose)
                if lhs_layout.num_dims() == 2 {
                    {
                        let (rows, cols, l_row_stride, l_col_stride) =
                            lhs_layout.as_2d_strides().unwrap();
                        let (_, _, r_row_stride, r_col_stride) =
                            rhs_layout.as_2d_strides().unwrap();
                        let l_offset = lhs_layout.start_offset() as isize;
                        let r_offset = rhs_layout.start_offset() as isize;

                        let mut result = Vec::with_capacity(rows * cols);

                        for row in 0..rows {
                            let l_row_start = l_offset + row as isize * l_row_stride;
                            let r_row_start = r_offset + row as isize * r_row_stride;
                            for col in 0..cols {
                                let l_idx = (l_row_start + col as isize * l_col_stride) as usize;
                                let r_idx = (r_row_start + col as isize * r_col_stride) as usize;
                                result.push(op(lhs_storage[l_idx], rhs_storage[r_idx]));
                            }
                        }

                        result
                    }
                } else {
                    // General fallback
                    let lhs_iter = StridedIter::new(lhs_layout);
                    let rhs_iter = StridedIter::new(rhs_layout);
                    lhs_iter
                        .zip(rhs_iter)
                        .map(|(li, ri)| op(lhs_storage[li], rhs_storage[ri]))
                        .collect()
                }
            }
        }
    };
    make_tensor(result, shape, O::dtype())
}
