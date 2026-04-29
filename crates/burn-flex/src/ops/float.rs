//! Float tensor operations for the Flex backend.

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::{
    DType, Distribution, ExecutionError, FloatDType, Scalar, TensorData, TensorMetadata,
    ops::{FloatTensorOps, GridSampleOptions, IntTensorOps},
    tensor::{BoolTensor, Device, FloatTensor, IntTensor},
};
use burn_std::{Bytes, IntDType, Shape, Slice, bf16, f16};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use crate::Layout;
use num_traits::ToPrimitive;

use crate::ops::binary::{BinaryOp, binary_op, scalar_op};
use crate::ops::matmul;
use crate::ops::unary;
use crate::{Flex, FlexTensor};

impl FloatTensorOps<Flex> for Flex {
    fn float_from_data(data: TensorData, _device: &Device<Flex>) -> FloatTensor<Flex> {
        FlexTensor::from_data(data)
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        _device: &Device<Flex>,
        dtype: FloatDType,
    ) -> FloatTensor<Flex> {
        let mut seed = crate::backend::SEED.lock().unwrap();
        let mut rng = seed.take().unwrap_or_else(crate::backend::get_seeded_rng);
        let data = match dtype {
            FloatDType::F64 => TensorData::random::<f64, _, _>(shape, distribution, &mut rng),
            FloatDType::F32 | FloatDType::Flex32 => {
                TensorData::random::<f32, _, _>(shape, distribution, &mut rng)
            }
            FloatDType::F16 => TensorData::random::<f16, _, _>(shape, distribution, &mut rng),
            FloatDType::BF16 => TensorData::random::<bf16, _, _>(shape, distribution, &mut rng),
        };
        *seed = Some(rng);
        FlexTensor::from_data(data)
    }

    async fn float_into_data(tensor: FloatTensor<Flex>) -> Result<TensorData, ExecutionError> {
        Ok(tensor.into_data())
    }

    fn float_device(_tensor: &FloatTensor<Flex>) -> Device<Flex> {
        // CPU backend: all tensors are on the default device
        Default::default()
    }

    fn float_to_device(tensor: FloatTensor<Flex>, _device: &Device<Flex>) -> FloatTensor<Flex> {
        // CPU backend: no-op, tensors are always on CPU
        tensor
    }

    fn float_detach(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        tensor
    }

    fn float_into_int(tensor: FloatTensor<Flex>, out_dtype: burn_std::IntDType) -> IntTensor<Flex> {
        let tensor = tensor.to_contiguous();
        let shape = tensor.layout().shape().clone();
        let src = tensor.dtype();
        let out_dt = DType::from(out_dtype);

        // Read source floats as f64 (lossless for f32/f16/bf16).
        macro_rules! read_floats {
            (|$x:ident| $conv:expr) => {
                match src {
                    DType::F32 => tensor
                        .storage::<f32>()
                        .iter()
                        .map(|v| {
                            let $x = *v as f64;
                            $conv
                        })
                        .collect(),
                    DType::F64 => tensor
                        .storage::<f64>()
                        .iter()
                        .map(|v| {
                            let $x = *v;
                            $conv
                        })
                        .collect(),
                    DType::F16 => tensor
                        .storage::<f16>()
                        .iter()
                        .map(|v| {
                            let $x = f32::from(*v) as f64;
                            $conv
                        })
                        .collect(),
                    DType::BF16 => tensor
                        .storage::<bf16>()
                        .iter()
                        .map(|v| {
                            let $x = f32::from(*v) as f64;
                            $conv
                        })
                        .collect(),
                    _ => panic!("float_into_int: unsupported source dtype {:?}", src),
                }
            };
        }

        macro_rules! convert {
            ($int_ty:ty) => {{
                let data: Vec<$int_ty> = read_floats!(|x| x as $int_ty);
                FlexTensor::new(Bytes::from_elems(data), Layout::contiguous(shape), out_dt)
            }};
        }

        match out_dtype {
            IntDType::I64 => convert!(i64),
            IntDType::I32 => convert!(i32),
            IntDType::I16 => convert!(i16),
            IntDType::I8 => convert!(i8),
            IntDType::U64 => convert!(u64),
            IntDType::U32 => convert!(u32),
            IntDType::U16 => convert!(u16),
            IntDType::U8 => convert!(u8),
        }
    }

    fn float_empty(shape: Shape, _device: &Device<Flex>, dtype: FloatDType) -> FloatTensor<Flex> {
        FlexTensor::empty(shape, dtype.into())
    }

    fn float_add(lhs: FloatTensor<Flex>, rhs: FloatTensor<Flex>) -> FloatTensor<Flex> {
        binary_op(lhs, rhs, |a, b| a + b, |a, b| a + b, Some(BinaryOp::Add))
    }

    fn float_add_scalar(lhs: FloatTensor<Flex>, rhs: Scalar) -> FloatTensor<Flex> {
        let rhs_val = rhs.to_f64().unwrap();
        scalar_op(lhs, rhs_val, |a, b| a + b, |a, b| a + b)
    }

    fn float_sub(lhs: FloatTensor<Flex>, rhs: FloatTensor<Flex>) -> FloatTensor<Flex> {
        binary_op(lhs, rhs, |a, b| a - b, |a, b| a - b, Some(BinaryOp::Sub))
    }

    fn float_sub_scalar(lhs: FloatTensor<Flex>, rhs: Scalar) -> FloatTensor<Flex> {
        let rhs_val = rhs.to_f64().unwrap();
        scalar_op(lhs, rhs_val, |a, b| a - b, |a, b| a - b)
    }

    fn float_mul(lhs: FloatTensor<Flex>, rhs: FloatTensor<Flex>) -> FloatTensor<Flex> {
        binary_op(lhs, rhs, |a, b| a * b, |a, b| a * b, Some(BinaryOp::Mul))
    }

    fn float_mul_scalar(lhs: FloatTensor<Flex>, rhs: Scalar) -> FloatTensor<Flex> {
        let rhs_val = rhs.to_f64().unwrap();
        scalar_op(lhs, rhs_val, |a, b| a * b, |a, b| a * b)
    }

    fn float_div(lhs: FloatTensor<Flex>, rhs: FloatTensor<Flex>) -> FloatTensor<Flex> {
        binary_op(lhs, rhs, |a, b| a / b, |a, b| a / b, Some(BinaryOp::Div))
    }

    fn float_div_scalar(lhs: FloatTensor<Flex>, rhs: Scalar) -> FloatTensor<Flex> {
        let rhs_val = rhs.to_f64().unwrap();
        scalar_op(lhs, rhs_val, |a, b| a / b, |a, b| a / b)
    }

    fn float_remainder(lhs: FloatTensor<Flex>, rhs: FloatTensor<Flex>) -> FloatTensor<Flex> {
        // Python/PyTorch-style remainder: result has same sign as divisor
        binary_op(
            lhs,
            rhs,
            |a, b| ((a % b) + b) % b,
            |a, b| ((a % b) + b) % b,
            None,
        )
    }

    fn float_remainder_scalar(lhs: FloatTensor<Flex>, rhs: Scalar) -> FloatTensor<Flex> {
        let rhs_val = rhs.to_f64().unwrap();
        // Python/PyTorch-style remainder: result has same sign as divisor
        scalar_op(
            lhs,
            rhs_val,
            |a, b| ((a % b) + b) % b,
            |a, b| ((a % b) + b) % b,
        )
    }

    fn float_matmul(lhs: FloatTensor<Flex>, rhs: FloatTensor<Flex>) -> FloatTensor<Flex> {
        matmul::matmul(lhs, rhs)
    }

    fn float_cross(
        lhs: FloatTensor<Flex>,
        rhs: FloatTensor<Flex>,
        dim: usize,
    ) -> FloatTensor<Flex> {
        let shape = lhs.layout().shape();
        let ndims = shape.num_dims();
        assert_eq!(
            shape[dim], 3,
            "cross product requires dimension {} to have size 3, got {}",
            dim, shape[dim]
        );

        // Helper to create slices that select index `idx` along `dim`
        let make_slices = |idx: usize| -> alloc::vec::Vec<Slice> {
            (0..ndims)
                .map(|d| {
                    if d == dim {
                        Slice::new(idx as isize, Some((idx + 1) as isize), 1)
                    } else {
                        Slice::new(0, None, 1)
                    }
                })
                .collect()
        };

        // Extract components along the dimension
        // a = [a0, a1, a2], b = [b0, b1, b2]
        let a0 = Self::float_slice(lhs.clone(), &make_slices(0));
        let a1 = Self::float_slice(lhs.clone(), &make_slices(1));
        let a2 = Self::float_slice(lhs, &make_slices(2));

        let b0 = Self::float_slice(rhs.clone(), &make_slices(0));
        let b1 = Self::float_slice(rhs.clone(), &make_slices(1));
        let b2 = Self::float_slice(rhs, &make_slices(2));

        // Cross product: c = a × b
        // c0 = a1*b2 - a2*b1
        // c1 = a2*b0 - a0*b2
        // c2 = a0*b1 - a1*b0
        let c0 = Self::float_sub(
            Self::float_mul(a1.clone(), b2.clone()),
            Self::float_mul(a2.clone(), b1.clone()),
        );
        let c1 = Self::float_sub(
            Self::float_mul(a2, b0.clone()),
            Self::float_mul(a0.clone(), b2),
        );
        let c2 = Self::float_sub(Self::float_mul(a0, b1), Self::float_mul(a1, b0));

        // Concatenate along the dimension
        Self::float_cat(vec![c0, c1, c2], dim)
    }

    fn float_recip(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::recip(tensor)
    }

    fn float_swap_dims(tensor: FloatTensor<Flex>, dim1: usize, dim2: usize) -> FloatTensor<Flex> {
        tensor.transpose(dim1, dim2)
    }

    fn float_permute(tensor: FloatTensor<Flex>, axes: &[usize]) -> FloatTensor<Flex> {
        tensor.permute(axes)
    }

    fn float_flip(tensor: FloatTensor<Flex>, axes: &[usize]) -> FloatTensor<Flex> {
        crate::ops::flip::flip(tensor, axes)
    }

    fn float_cat(tensors: Vec<FloatTensor<Flex>>, dim: usize) -> FloatTensor<Flex> {
        crate::ops::cat::cat(tensors, dim)
    }

    fn float_reshape(tensor: FloatTensor<Flex>, shape: Shape) -> FloatTensor<Flex> {
        tensor.reshape(shape)
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Flex>,
        indices: IntTensor<Flex>,
    ) -> FloatTensor<Flex> {
        match tensor.dtype() {
            DType::F32 => crate::ops::gather_scatter::gather::<f32>(tensor, dim, indices),
            DType::F64 => crate::ops::gather_scatter::gather::<f64>(tensor, dim, indices),
            DType::F16 => crate::ops::gather_scatter::gather::<f16>(tensor, dim, indices),
            DType::BF16 => crate::ops::gather_scatter::gather::<bf16>(tensor, dim, indices),
            _ => panic!("float_gather: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn float_scatter_add(
        dim: usize,
        tensor: FloatTensor<Flex>,
        indices: IntTensor<Flex>,
        value: FloatTensor<Flex>,
    ) -> FloatTensor<Flex> {
        match tensor.dtype() {
            DType::F32 => {
                crate::ops::gather_scatter::scatter_add::<f32>(tensor, dim, indices, value)
            }
            DType::F64 => {
                crate::ops::gather_scatter::scatter_add::<f64>(tensor, dim, indices, value)
            }
            DType::F16 => {
                crate::ops::gather_scatter::scatter_add::<f16>(tensor, dim, indices, value)
            }
            DType::BF16 => {
                crate::ops::gather_scatter::scatter_add::<bf16>(tensor, dim, indices, value)
            }
            _ => panic!("float_scatter_add: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn float_scatter_nd(
        data: FloatTensor<Flex>,
        indices: IntTensor<Flex>,
        values: FloatTensor<Flex>,
        reduction: burn_backend::tensor::IndexingUpdateOp,
    ) -> FloatTensor<Flex> {
        match data.dtype() {
            DType::F32 => {
                crate::ops::gather_scatter::scatter_nd::<f32>(data, indices, values, reduction)
            }
            DType::F64 => {
                crate::ops::gather_scatter::scatter_nd::<f64>(data, indices, values, reduction)
            }
            DType::F16 => {
                crate::ops::gather_scatter::scatter_nd::<f16>(data, indices, values, reduction)
            }
            DType::BF16 => {
                crate::ops::gather_scatter::scatter_nd::<bf16>(data, indices, values, reduction)
            }
            _ => panic!("float_scatter_nd: unsupported dtype {:?}", data.dtype()),
        }
    }

    fn float_gather_nd(data: FloatTensor<Flex>, indices: IntTensor<Flex>) -> FloatTensor<Flex> {
        match data.dtype() {
            DType::F32 => crate::ops::gather_scatter::gather_nd::<f32>(data, indices),
            DType::F64 => crate::ops::gather_scatter::gather_nd::<f64>(data, indices),
            DType::F16 => crate::ops::gather_scatter::gather_nd::<f16>(data, indices),
            DType::BF16 => crate::ops::gather_scatter::gather_nd::<bf16>(data, indices),
            _ => panic!("float_gather_nd: unsupported dtype {:?}", data.dtype()),
        }
    }

    fn float_select(
        tensor: FloatTensor<Flex>,
        dim: usize,
        indices: IntTensor<Flex>,
    ) -> FloatTensor<Flex> {
        match tensor.dtype() {
            DType::F32 => crate::ops::gather_scatter::select::<f32>(tensor, dim, indices),
            DType::F64 => crate::ops::gather_scatter::select::<f64>(tensor, dim, indices),
            DType::F16 => crate::ops::gather_scatter::select::<f16>(tensor, dim, indices),
            DType::BF16 => crate::ops::gather_scatter::select::<bf16>(tensor, dim, indices),
            _ => panic!("float_select: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn float_select_add(
        tensor: FloatTensor<Flex>,
        dim: usize,
        indices: IntTensor<Flex>,
        value: FloatTensor<Flex>,
    ) -> FloatTensor<Flex> {
        match tensor.dtype() {
            DType::F32 => {
                crate::ops::gather_scatter::select_add::<f32>(tensor, dim, indices, value)
            }
            DType::F64 => {
                crate::ops::gather_scatter::select_add::<f64>(tensor, dim, indices, value)
            }
            DType::F16 => {
                crate::ops::gather_scatter::select_add::<f16>(tensor, dim, indices, value)
            }
            DType::BF16 => {
                crate::ops::gather_scatter::select_add::<bf16>(tensor, dim, indices, value)
            }
            _ => panic!("float_select_add: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn float_slice(tensor: FloatTensor<Flex>, slices: &[Slice]) -> FloatTensor<Flex> {
        crate::ops::slice::slice(tensor, slices)
    }

    fn float_slice_assign(
        tensor: FloatTensor<Flex>,
        slices: &[Slice],
        value: FloatTensor<Flex>,
    ) -> FloatTensor<Flex> {
        crate::ops::slice::slice_assign(tensor, slices, value)
    }

    fn float_mask_where(
        tensor: FloatTensor<Flex>,
        mask: BoolTensor<Flex>,
        value: FloatTensor<Flex>,
    ) -> FloatTensor<Flex> {
        match tensor.dtype() {
            DType::F32 => crate::ops::mask::mask_where_f32(tensor, mask, value),
            DType::F64 => crate::ops::mask::mask_where_f64(tensor, mask, value),
            DType::F16 => crate::ops::mask::mask_where_f16(tensor, mask, value),
            DType::BF16 => crate::ops::mask::mask_where_bf16(tensor, mask, value),
            dtype => panic!("float_mask_where: unsupported dtype {:?}", dtype),
        }
    }

    fn float_mask_fill(
        tensor: FloatTensor<Flex>,
        mask: BoolTensor<Flex>,
        value: Scalar,
    ) -> FloatTensor<Flex> {
        match tensor.dtype() {
            DType::F32 => crate::ops::mask::mask_fill_f32(tensor, mask, value.to_f32().unwrap()),
            DType::F64 => crate::ops::mask::mask_fill_f64(tensor, mask, value.to_f64().unwrap()),
            DType::F16 => crate::ops::mask::mask_fill_f16(
                tensor,
                mask,
                f16::from_f64(value.to_f64().unwrap()),
            ),
            DType::BF16 => crate::ops::mask::mask_fill_bf16(
                tensor,
                mask,
                bf16::from_f64(value.to_f64().unwrap()),
            ),
            dtype => panic!("float_mask_fill: unsupported dtype {:?}", dtype),
        }
    }

    fn float_equal(
        lhs: FloatTensor<Flex>,
        rhs: FloatTensor<Flex>,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::equal(lhs, rhs, out_dtype)
    }

    fn float_equal_elem(
        lhs: FloatTensor<Flex>,
        rhs: Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::equal_elem(lhs, rhs.to_f64().unwrap(), out_dtype)
    }

    fn float_greater(
        lhs: FloatTensor<Flex>,
        rhs: FloatTensor<Flex>,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::greater(lhs, rhs, out_dtype)
    }

    fn float_greater_elem(
        lhs: FloatTensor<Flex>,
        rhs: Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::greater_elem(lhs, rhs.to_f64().unwrap(), out_dtype)
    }

    fn float_greater_equal(
        lhs: FloatTensor<Flex>,
        rhs: FloatTensor<Flex>,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::greater_equal(lhs, rhs, out_dtype)
    }

    fn float_greater_equal_elem(
        lhs: FloatTensor<Flex>,
        rhs: Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::greater_equal_elem(lhs, rhs.to_f64().unwrap(), out_dtype)
    }

    fn float_lower(
        lhs: FloatTensor<Flex>,
        rhs: FloatTensor<Flex>,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::lower(lhs, rhs, out_dtype)
    }

    fn float_lower_elem(
        lhs: FloatTensor<Flex>,
        rhs: Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::lower_elem(lhs, rhs.to_f64().unwrap(), out_dtype)
    }

    fn float_lower_equal(
        lhs: FloatTensor<Flex>,
        rhs: FloatTensor<Flex>,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::lower_equal(lhs, rhs, out_dtype)
    }

    fn float_lower_equal_elem(
        lhs: FloatTensor<Flex>,
        rhs: Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::lower_equal_elem(lhs, rhs.to_f64().unwrap(), out_dtype)
    }

    fn float_not_equal(
        lhs: FloatTensor<Flex>,
        rhs: FloatTensor<Flex>,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::not_equal(lhs, rhs, out_dtype)
    }

    fn float_not_equal_elem(
        lhs: FloatTensor<Flex>,
        rhs: Scalar,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::not_equal_elem(lhs, rhs.to_f64().unwrap(), out_dtype)
    }

    fn float_neg(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::unary_op(tensor, |x: f32| -x, |x: f64| -x)
    }

    fn float_clamp(tensor: FloatTensor<Flex>, min: Scalar, max: Scalar) -> FloatTensor<Flex> {
        let min32 = min.to_f32().unwrap();
        let max32 = max.to_f32().unwrap();
        let min64 = min.to_f64().unwrap();
        let max64 = max.to_f64().unwrap();
        unary::unary_op(
            tensor,
            move |x: f32| x.clamp(min32, max32),
            move |x: f64| x.clamp(min64, max64),
        )
    }

    fn float_clamp_min(tensor: FloatTensor<Flex>, min: Scalar) -> FloatTensor<Flex> {
        let min32 = min.to_f32().unwrap();
        let min64 = min.to_f64().unwrap();
        unary::unary_op(
            tensor,
            move |x: f32| x.max(min32),
            move |x: f64| x.max(min64),
        )
    }

    fn float_clamp_max(tensor: FloatTensor<Flex>, max: Scalar) -> FloatTensor<Flex> {
        let max32 = max.to_f32().unwrap();
        let max64 = max.to_f64().unwrap();
        unary::unary_op(
            tensor,
            move |x: f32| x.min(max32),
            move |x: f64| x.min(max64),
        )
    }

    fn float_sign(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::unary_op(
            tensor,
            |x: f32| {
                if x.is_nan() {
                    x
                } else if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            },
            |x: f64| {
                if x.is_nan() {
                    x
                } else if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            },
        )
    }

    fn float_mean(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        crate::ops::reduce::mean(tensor)
    }

    fn float_max(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        crate::ops::reduce::max(tensor)
    }

    fn float_max_dim(tensor: FloatTensor<Flex>, dim: usize) -> FloatTensor<Flex> {
        crate::ops::reduce::max_dim(tensor, dim)
    }

    fn float_min(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        crate::ops::reduce::min(tensor)
    }

    fn float_min_dim(tensor: FloatTensor<Flex>, dim: usize) -> FloatTensor<Flex> {
        crate::ops::reduce::min_dim(tensor, dim)
    }

    fn float_max_dim_with_indices(
        tensor: FloatTensor<Flex>,
        dim: usize,
        indices_dtype: burn_std::IntDType,
    ) -> (FloatTensor<Flex>, IntTensor<Flex>) {
        let (values, indices) = crate::ops::reduce::max_dim_with_indices(tensor, dim);
        if indices.dtype() != DType::from(indices_dtype) {
            (values, Flex::int_cast(indices, indices_dtype))
        } else {
            (values, indices)
        }
    }

    fn float_min_dim_with_indices(
        tensor: FloatTensor<Flex>,
        dim: usize,
        indices_dtype: burn_std::IntDType,
    ) -> (FloatTensor<Flex>, IntTensor<Flex>) {
        let (values, indices) = crate::ops::reduce::min_dim_with_indices(tensor, dim);
        if indices.dtype() != DType::from(indices_dtype) {
            (values, Flex::int_cast(indices, indices_dtype))
        } else {
            (values, indices)
        }
    }

    fn float_any(tensor: FloatTensor<Flex>, out_dtype: burn_std::BoolDType) -> BoolTensor<Flex> {
        crate::ops::comparison::any_float(tensor, out_dtype)
    }

    fn float_any_dim(
        tensor: FloatTensor<Flex>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::any_float_dim(tensor, dim, out_dtype)
    }

    fn float_all(tensor: FloatTensor<Flex>, out_dtype: burn_std::BoolDType) -> BoolTensor<Flex> {
        crate::ops::comparison::all_float(tensor, out_dtype)
    }

    fn float_all_dim(
        tensor: FloatTensor<Flex>,
        dim: usize,
        out_dtype: burn_std::BoolDType,
    ) -> BoolTensor<Flex> {
        crate::ops::comparison::all_float_dim(tensor, dim, out_dtype)
    }

    fn float_sum(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        crate::ops::reduce::sum(tensor)
    }

    fn float_sum_dim(tensor: FloatTensor<Flex>, dim: usize) -> FloatTensor<Flex> {
        crate::ops::reduce::sum_dim(tensor, dim)
    }

    fn float_mean_dim(tensor: FloatTensor<Flex>, dim: usize) -> FloatTensor<Flex> {
        crate::ops::reduce::mean_dim(tensor, dim)
    }

    fn float_prod(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        crate::ops::reduce::prod(tensor)
    }

    fn float_prod_dim(tensor: FloatTensor<Flex>, dim: usize) -> FloatTensor<Flex> {
        crate::ops::reduce::prod_dim(tensor, dim)
    }

    fn float_cumsum(tensor: FloatTensor<Flex>, dim: usize) -> FloatTensor<Flex> {
        match tensor.dtype() {
            DType::F32 => crate::ops::cumulative::cumsum_f32(tensor, dim),
            DType::F64 => crate::ops::cumulative::cumsum_f64(tensor, dim),
            DType::F16 => {
                crate::ops::cumulative::cumsum_half(tensor, dim, f16::to_f32, f16::from_f32)
            }
            DType::BF16 => {
                crate::ops::cumulative::cumsum_half(tensor, dim, bf16::to_f32, bf16::from_f32)
            }
            _ => panic!("float_cumsum: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn float_cumprod(tensor: FloatTensor<Flex>, dim: usize) -> FloatTensor<Flex> {
        match tensor.dtype() {
            DType::F32 => crate::ops::cumulative::cumprod_f32(tensor, dim),
            DType::F64 => crate::ops::cumulative::cumprod_f64(tensor, dim),
            DType::F16 => {
                crate::ops::cumulative::cumprod_half(tensor, dim, f16::to_f32, f16::from_f32)
            }
            DType::BF16 => {
                crate::ops::cumulative::cumprod_half(tensor, dim, bf16::to_f32, bf16::from_f32)
            }
            _ => panic!("float_cumprod: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn float_cummin(tensor: FloatTensor<Flex>, dim: usize) -> FloatTensor<Flex> {
        match tensor.dtype() {
            DType::F32 => crate::ops::cumulative::cummin_f32(tensor, dim),
            DType::F64 => crate::ops::cumulative::cummin_f64(tensor, dim),
            DType::F16 => {
                crate::ops::cumulative::cummin_half(tensor, dim, f16::to_f32, f16::from_f32)
            }
            DType::BF16 => {
                crate::ops::cumulative::cummin_half(tensor, dim, bf16::to_f32, bf16::from_f32)
            }
            _ => panic!("float_cummin: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn float_cummax(tensor: FloatTensor<Flex>, dim: usize) -> FloatTensor<Flex> {
        match tensor.dtype() {
            DType::F32 => crate::ops::cumulative::cummax_f32(tensor, dim),
            DType::F64 => crate::ops::cumulative::cummax_f64(tensor, dim),
            DType::F16 => {
                crate::ops::cumulative::cummax_half(tensor, dim, f16::to_f32, f16::from_f32)
            }
            DType::BF16 => {
                crate::ops::cumulative::cummax_half(tensor, dim, bf16::to_f32, bf16::from_f32)
            }
            _ => panic!("float_cummax: unsupported dtype {:?}", tensor.dtype()),
        }
    }

    fn float_cast(tensor: FloatTensor<Flex>, dtype: FloatDType) -> FloatTensor<Flex> {
        use crate::Layout;
        use burn_std::{Bytes, bf16, f16};

        let src_dtype = tensor.dtype();
        let target_dtype = DType::from(dtype);

        // No-op if already the same dtype
        if src_dtype == target_dtype {
            return tensor;
        }

        let tensor = tensor.to_contiguous();
        let shape = tensor.layout().shape().clone();

        // Convert to f64 intermediate, then to target
        let f64_values: Vec<f64> = match src_dtype {
            DType::F32 => {
                let src: &[f32] = tensor.storage();
                src.iter().map(|&v| v as f64).collect()
            }
            DType::F64 => {
                let src: &[f64] = tensor.storage();
                src.to_vec()
            }
            DType::F16 => {
                let src: &[f16] = tensor.storage();
                src.iter().map(|&v| v.to_f32() as f64).collect()
            }
            DType::BF16 => {
                let src: &[bf16] = tensor.storage();
                src.iter().map(|&v| v.to_f32() as f64).collect()
            }
            _ => panic!("float_cast: unsupported source dtype {:?}", src_dtype),
        };

        // Convert from f64 to target dtype
        match target_dtype {
            DType::F32 => {
                let result: Vec<f32> = f64_values.iter().map(|&v| v as f32).collect();
                let bytes = Bytes::from_elems(result);
                FlexTensor::new(bytes, Layout::contiguous(shape), DType::F32)
            }
            DType::F64 => {
                let bytes = Bytes::from_elems(f64_values);
                FlexTensor::new(bytes, Layout::contiguous(shape), DType::F64)
            }
            DType::F16 => {
                let result: Vec<f16> = f64_values.iter().map(|&v| f16::from_f64(v)).collect();
                let bytes = Bytes::from_elems(result);
                FlexTensor::new(bytes, Layout::contiguous(shape), DType::F16)
            }
            DType::BF16 => {
                let result: Vec<bf16> = f64_values.iter().map(|&v| bf16::from_f64(v)).collect();
                let bytes = Bytes::from_elems(result);
                FlexTensor::new(bytes, Layout::contiguous(shape), DType::BF16)
            }
            _ => panic!("float_cast: unsupported target dtype {:?}", target_dtype),
        }
    }

    fn float_exp(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::exp(tensor)
    }

    fn float_log(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::log(tensor)
    }

    fn float_log1p(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::log1p(tensor)
    }

    fn float_powf(lhs: FloatTensor<Flex>, rhs: FloatTensor<Flex>) -> FloatTensor<Flex> {
        binary_op(lhs, rhs, |a: f32, b| a.powf(b), |a: f64, b| a.powf(b), None)
    }

    fn float_powf_scalar_impl(tensor: FloatTensor<Flex>, value: Scalar) -> FloatTensor<Flex> {
        let exp = value.to_f64().unwrap();
        scalar_op(tensor, exp, |a: f32, b| a.powf(b), |a: f64, b| a.powf(b))
    }

    fn float_sqrt(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::sqrt(tensor)
    }

    fn float_abs(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::abs(tensor)
    }

    fn float_cos(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::cos(tensor)
    }

    fn float_sin(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::sin(tensor)
    }

    fn float_tan(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::tan(tensor)
    }

    fn float_cosh(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::cosh(tensor)
    }

    fn float_sinh(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::sinh(tensor)
    }

    fn float_tanh(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::tanh(tensor)
    }

    fn float_acos(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::acos(tensor)
    }

    fn float_acosh(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::acosh(tensor)
    }

    fn float_asin(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::asin(tensor)
    }

    fn float_asinh(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::asinh(tensor)
    }

    fn float_atan(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::atan(tensor)
    }

    fn float_atanh(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::atanh(tensor)
    }

    fn float_atan2(lhs: FloatTensor<Flex>, rhs: FloatTensor<Flex>) -> FloatTensor<Flex> {
        binary_op(
            lhs,
            rhs,
            |a: f32, b| a.atan2(b),
            |a: f64, b| a.atan2(b),
            None,
        )
    }

    fn float_round(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::round(tensor)
    }

    fn float_floor(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::floor(tensor)
    }

    fn float_ceil(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::ceil(tensor)
    }

    fn float_trunc(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::trunc(tensor)
    }

    fn float_erf(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        unary::erf(tensor)
    }

    fn float_argmax(
        tensor: FloatTensor<Flex>,
        dim: usize,
        out_dtype: burn_std::IntDType,
    ) -> IntTensor<Flex> {
        let result = crate::ops::reduce::argmax(tensor, dim);
        if result.dtype() != DType::from(out_dtype) {
            Flex::int_cast(result, out_dtype)
        } else {
            result
        }
    }

    fn float_argtopk(
        _tensor: FloatTensor<Flex>,
        _dim: usize,
        _k: usize,
        _out_dtype: burn_std::IntDType,
    ) -> IntTensor<Flex> {
        panic!("argtopk not implemented for flex")
    }

    fn float_argmin(
        tensor: FloatTensor<Flex>,
        dim: usize,
        out_dtype: burn_std::IntDType,
    ) -> IntTensor<Flex> {
        let result = crate::ops::reduce::argmin(tensor, dim);
        if result.dtype() != DType::from(out_dtype) {
            Flex::int_cast(result, out_dtype)
        } else {
            result
        }
    }

    fn float_expand(tensor: FloatTensor<Flex>, shape: Shape) -> FloatTensor<Flex> {
        crate::ops::expand::expand(tensor, shape)
    }

    fn float_unfold(
        tensor: FloatTensor<Flex>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> FloatTensor<Flex> {
        // unfold is now type-agnostic (zero-copy strided view)
        crate::ops::unfold::unfold(tensor, dim, size, step)
    }

    fn float_grid_sample_2d(
        tensor: FloatTensor<Flex>,
        grid: FloatTensor<Flex>,
        options: GridSampleOptions,
    ) -> FloatTensor<Flex> {
        crate::ops::grid_sample::grid_sample_2d(tensor, grid, options)
    }

    fn float_zeros(shape: Shape, _device: &Device<Flex>, dtype: FloatDType) -> FloatTensor<Flex> {
        FlexTensor::zeros(shape, dtype.into())
    }

    fn float_ones(shape: Shape, _device: &Device<Flex>, dtype: FloatDType) -> FloatTensor<Flex> {
        let dt: burn_backend::DType = dtype.into();
        match dt {
            DType::F32 => FlexTensor::filled_typed(shape, dt, 1.0f32),
            DType::F64 => FlexTensor::filled_typed(shape, dt, 1.0f64),
            DType::F16 => FlexTensor::filled_typed(shape, dt, f16::ONE),
            DType::BF16 => FlexTensor::filled_typed(shape, dt, bf16::ONE),
            _ => unreachable!(),
        }
    }

    fn float_full(
        shape: Shape,
        fill_value: Scalar,
        _device: &Device<Flex>,
        dtype: FloatDType,
    ) -> FloatTensor<Flex> {
        let dt: burn_backend::DType = dtype.into();
        match dt {
            DType::F32 => FlexTensor::filled_typed(shape, dt, fill_value.to_f32().unwrap()),
            DType::F64 => FlexTensor::filled_typed(shape, dt, fill_value.to_f64().unwrap()),
            DType::F16 => {
                FlexTensor::filled_typed(shape, dt, f16::from_f32(fill_value.to_f32().unwrap()))
            }
            DType::BF16 => {
                FlexTensor::filled_typed(shape, dt, bf16::from_f32(fill_value.to_f32().unwrap()))
            }
            _ => unreachable!(),
        }
    }

    fn float_transpose(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        let ndims = tensor.layout().num_dims();
        if ndims < 2 {
            return tensor;
        }
        tensor.transpose(ndims - 2, ndims - 1)
    }

    fn float_repeat_dim(tensor: FloatTensor<Flex>, dim: usize, times: usize) -> FloatTensor<Flex> {
        crate::ops::repeat_dim::repeat_dim(tensor, dim, times)
    }

    fn float_sort(tensor: FloatTensor<Flex>, dim: usize, descending: bool) -> FloatTensor<Flex> {
        crate::ops::sort::sort(tensor, dim, descending)
    }

    fn float_sort_with_indices(
        tensor: FloatTensor<Flex>,
        dim: usize,
        descending: bool,
        indices_dtype: burn_std::IntDType,
    ) -> (FloatTensor<Flex>, IntTensor<Flex>) {
        let (values, indices) = crate::ops::sort::sort_with_indices(tensor, dim, descending);
        let indices = if indices.dtype() != DType::from(indices_dtype) {
            Flex::int_cast(indices, indices_dtype)
        } else {
            indices
        };
        (values, indices)
    }

    fn float_argsort(
        tensor: FloatTensor<Flex>,
        dim: usize,
        descending: bool,
        out_dtype: burn_std::IntDType,
    ) -> IntTensor<Flex> {
        let indices = crate::ops::sort::argsort(tensor, dim, descending);
        if indices.dtype() != DType::from(out_dtype) {
            Flex::int_cast(indices, out_dtype)
        } else {
            indices
        }
    }

    fn float_powi(lhs: FloatTensor<Flex>, rhs: IntTensor<Flex>) -> FloatTensor<Flex> {
        let dtype = lhs.dtype();
        Self::float_powf(lhs, Flex::int_into_float(rhs, dtype.into()))
    }

    fn float_powi_scalar(lhs: FloatTensor<Flex>, rhs: Scalar) -> FloatTensor<Flex> {
        match rhs.to_i64().unwrap() {
            0 => Self::float_ones(lhs.shape(), &Default::default(), lhs.dtype().into()),
            1 => lhs,
            2 => Self::float_mul(lhs.clone(), lhs),
            -1 => Self::float_recip(lhs),
            -2 => Self::float_recip(Self::float_mul(lhs.clone(), lhs)),
            _ => Self::float_powf_scalar_impl(lhs, rhs),
        }
    }

    fn float_powf_scalar(tensor: FloatTensor<Flex>, value: Scalar) -> FloatTensor<Flex> {
        if let Some(exp) = value.try_as_integer() {
            Self::float_powi_scalar(tensor, exp)
        } else {
            Self::float_powf_scalar_impl(tensor, value)
        }
    }

    fn float_max_abs(tensor: FloatTensor<Flex>) -> FloatTensor<Flex> {
        let abs = unary::abs(tensor);
        crate::ops::reduce::max(abs)
    }

    fn float_max_abs_dim(tensor: FloatTensor<Flex>, dim: usize) -> FloatTensor<Flex> {
        let abs = unary::abs(tensor);
        crate::ops::reduce::max_dim(abs, dim)
    }

    fn float_is_nan(tensor: FloatTensor<Flex>, out_dtype: burn_std::BoolDType) -> BoolTensor<Flex> {
        unary::float_predicate(tensor, out_dtype, |x: f32| x.is_nan(), |x: f64| x.is_nan())
    }

    fn float_is_inf(tensor: FloatTensor<Flex>, out_dtype: burn_std::BoolDType) -> BoolTensor<Flex> {
        unary::float_predicate(
            tensor,
            out_dtype,
            |x: f32| x.is_infinite(),
            |x: f64| x.is_infinite(),
        )
    }
}

// Tests kept here exercise flex-specific behavior: direct `Flex::`
// backend-op calls with explicit IntDType/FloatDType to pin dtype storage
// selection (U8/I32/I64, F16/F64). Plain arithmetic, math, cast, cross,
// unfold, and random smoke tests have been dropped in favor of the
// equivalent coverage in burn-backend-tests, which exercises every backend.
// When adding new tests, keep them here only if they probe flex dtype
// storage or flex internals; otherwise add them to
// crates/burn-backend-tests/tests/tensor/float/ops/.
#[cfg(test)]
mod tests {
    use burn_backend::TensorData;

    use crate::Flex;

    #[test]
    fn test_float_into_int_i32() {
        use burn_backend::ops::FloatTensorOps;
        use burn_std::IntDType;

        let t = crate::FlexTensor::from_data(TensorData::from([1.5f32, -2.7, 0.0, 255.9]));
        let result = Flex::float_into_int(t, IntDType::I32);
        assert_eq!(result.dtype(), burn_backend::DType::I32);
        let data: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1, -2, 0, 255]);
    }

    #[test]
    fn test_float_into_int_u8() {
        use burn_backend::ops::FloatTensorOps;
        use burn_std::IntDType;

        let t = crate::FlexTensor::from_data(TensorData::from([0.0f32, 1.9, 127.5, 255.0]));
        let result = Flex::float_into_int(t, IntDType::U8);
        assert_eq!(result.dtype(), burn_backend::DType::U8);
        let data: Vec<u8> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![0, 1, 127, 255]);
    }

    #[test]
    fn test_float_argmax_i32_out_dtype() {
        use burn_backend::ops::FloatTensorOps;
        use burn_std::IntDType;

        let t = crate::FlexTensor::from_data(TensorData::from([[1.0f32, 3.0, 2.0]]));
        let result = Flex::float_argmax(t, 1, IntDType::I32);
        assert_eq!(result.dtype(), burn_backend::DType::I32);
        let data: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1]);
    }

    #[test]
    fn test_float_argmin_i32_out_dtype() {
        use burn_backend::ops::FloatTensorOps;
        use burn_std::IntDType;

        let t = crate::FlexTensor::from_data(TensorData::from([[3.0f32, 1.0, 2.0]]));
        let result = Flex::float_argmin(t, 1, IntDType::I32);
        assert_eq!(result.dtype(), burn_backend::DType::I32);
        let data: Vec<i32> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1]);
    }

    #[test]
    fn test_float_argmax_i64_out_dtype() {
        use burn_backend::ops::FloatTensorOps;
        use burn_std::IntDType;

        let t = crate::FlexTensor::from_data(TensorData::from([[1.0f32, 3.0, 2.0]]));
        let result = Flex::float_argmax(t, 1, IntDType::I64);
        assert_eq!(result.dtype(), burn_backend::DType::I64);
        let data: Vec<i64> = result.into_data().to_vec().unwrap();
        assert_eq!(data, vec![1]);
    }

    #[test]
    fn test_float_max_dim_with_indices_i32() {
        use burn_backend::ops::FloatTensorOps;
        use burn_std::IntDType;

        let t = crate::FlexTensor::from_data(TensorData::from([[1.0f32, 5.0], [3.0, 2.0]]));
        let (values, indices) = Flex::float_max_dim_with_indices(t, 1, IntDType::I32);
        assert_eq!(indices.dtype(), burn_backend::DType::I32);
        let idx: Vec<i32> = indices.into_data().to_vec().unwrap();
        assert_eq!(idx, vec![1, 0]);
        let vals: Vec<f32> = values.into_data().to_vec().unwrap();
        assert_eq!(vals, vec![5.0, 3.0]);
    }

    #[test]
    fn test_float_min_dim_with_indices_i32() {
        use burn_backend::ops::FloatTensorOps;
        use burn_std::IntDType;

        let t = crate::FlexTensor::from_data(TensorData::from([[1.0f32, 5.0], [3.0, 2.0]]));
        let (values, indices) = Flex::float_min_dim_with_indices(t, 1, IntDType::I32);
        assert_eq!(indices.dtype(), burn_backend::DType::I32);
        let idx: Vec<i32> = indices.into_data().to_vec().unwrap();
        assert_eq!(idx, vec![0, 1]);
        let vals: Vec<f32> = values.into_data().to_vec().unwrap();
        assert_eq!(vals, vec![1.0, 2.0]);
    }

    #[test]
    fn test_float_random_f64() {
        use burn_backend::{DType, FloatDType, ops::FloatTensorOps};

        let shape = burn_std::Shape::from(vec![100]);
        let dist = burn_backend::Distribution::Uniform(0.0, 1.0);
        let device = crate::FlexDevice;
        let t = Flex::float_random(shape, dist, &device, FloatDType::F64);
        assert_eq!(t.dtype(), DType::F64);
        let data: Vec<f64> = t.into_data().to_vec().unwrap();
        assert!(data.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn test_float_random_f16() {
        use burn_backend::{DType, FloatDType, ops::FloatTensorOps};

        let shape = burn_std::Shape::from(vec![100]);
        let dist = burn_backend::Distribution::Uniform(0.0, 1.0);
        let device = crate::FlexDevice;
        let t = Flex::float_random(shape, dist, &device, FloatDType::F16);
        assert_eq!(t.dtype(), DType::F16);
    }
}
