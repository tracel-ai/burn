use alloc::vec::Vec;
use burn_backend::{
    Backend, BoolDType, Distribution, ExecutionError, FloatDType, IntDType, Scalar, Shape, Slice,
    TensorData, TensorMetadata,
    ops::FloatTensorOps,
    tensor::{BoolTensor, FloatTensor, IntTensor},
};
use burn_std::ops::PadMode;
use core::future::Future;

use super::{DualFloatTensor, ForwardDiff, align_tangent, bool_dtype, int_dtype};

type Dual<B> = DualFloatTensor<B>;

fn zero_like<B: Backend>(tensor: &FloatTensor<B>) -> FloatTensor<B> {
    B::float_zeros(tensor.shape(), &tensor.device(), tensor.dtype().into())
}

fn scalar_zero() -> Scalar {
    Scalar::from(0.0f64)
}

impl<B: Backend> FloatTensorOps<ForwardDiff<B>> for ForwardDiff<B> {
    fn float_from_data(data: TensorData, device: &B::Device) -> Dual<B> {
        DualFloatTensor::constant(B::float_from_data(data, device))
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &B::Device,
        dtype: FloatDType,
    ) -> Dual<B> {
        DualFloatTensor::constant(B::float_random(shape, distribution, device, dtype))
    }

    fn float_into_data(
        tensor: Dual<B>,
    ) -> impl Future<Output = Result<TensorData, ExecutionError>> + Send {
        B::float_into_data(tensor.primal)
    }

    fn float_to_device(tensor: Dual<B>, device: &B::Device) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_to_device(tensor.primal, device),
            tangent: tensor
                .tangent
                .map(|tangent| B::float_to_device(tangent, device)),
        }
    }

    fn float_into_int(tensor: Dual<B>, out_dtype: IntDType) -> IntTensor<B> {
        B::float_into_int(tensor.primal, out_dtype)
    }

    fn float_empty(shape: Shape, device: &B::Device, dtype: FloatDType) -> Dual<B> {
        DualFloatTensor::constant(B::float_empty(shape, device, dtype))
    }

    fn float_add(lhs: Dual<B>, rhs: Dual<B>) -> Dual<B> {
        let primal = B::float_add(lhs.primal.clone(), rhs.primal.clone());
        let shape = primal.shape();
        let tangent = match (lhs.tangent, rhs.tangent) {
            (Some(lhs), Some(rhs)) => Some(B::float_add(
                align_tangent::<B>(lhs, &shape),
                align_tangent::<B>(rhs, &shape),
            )),
            (Some(lhs), None) => Some(align_tangent::<B>(lhs, &shape)),
            (None, Some(rhs)) => Some(align_tangent::<B>(rhs, &shape)),
            (None, None) => None,
        };
        DualFloatTensor { primal, tangent }
    }

    fn float_add_scalar(lhs: Dual<B>, rhs: Scalar) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_add_scalar(lhs.primal, rhs),
            tangent: lhs.tangent,
        }
    }

    fn float_sub(lhs: Dual<B>, rhs: Dual<B>) -> Dual<B> {
        let primal = B::float_sub(lhs.primal.clone(), rhs.primal.clone());
        let shape = primal.shape();
        let tangent = match (lhs.tangent, rhs.tangent) {
            (Some(lhs), Some(rhs)) => Some(B::float_sub(
                align_tangent::<B>(lhs, &shape),
                align_tangent::<B>(rhs, &shape),
            )),
            (Some(lhs), None) => Some(align_tangent::<B>(lhs, &shape)),
            (None, Some(rhs)) => Some(B::float_neg(align_tangent::<B>(rhs, &shape))),
            (None, None) => None,
        };
        DualFloatTensor { primal, tangent }
    }

    fn float_sub_scalar(lhs: Dual<B>, rhs: Scalar) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_sub_scalar(lhs.primal, rhs),
            tangent: lhs.tangent,
        }
    }

    fn float_mul(lhs: Dual<B>, rhs: Dual<B>) -> Dual<B> {
        let primal = B::float_mul(lhs.primal.clone(), rhs.primal.clone());
        let shape = primal.shape();
        let tangent = match (lhs.tangent, rhs.tangent) {
            (Some(lhs_tangent), Some(rhs_tangent)) => Some(B::float_add(
                B::float_mul(align_tangent::<B>(lhs_tangent, &shape), rhs.primal),
                B::float_mul(lhs.primal, align_tangent::<B>(rhs_tangent, &shape)),
            )),
            (Some(lhs_tangent), None) => Some(B::float_mul(
                align_tangent::<B>(lhs_tangent, &shape),
                rhs.primal,
            )),
            (None, Some(rhs_tangent)) => Some(B::float_mul(
                lhs.primal,
                align_tangent::<B>(rhs_tangent, &shape),
            )),
            (None, None) => None,
        };
        DualFloatTensor { primal, tangent }
    }

    fn float_mul_scalar(lhs: Dual<B>, rhs: Scalar) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_mul_scalar(lhs.primal, rhs),
            tangent: lhs.tangent.map(|tangent| B::float_mul_scalar(tangent, rhs)),
        }
    }

    fn float_div(lhs: Dual<B>, rhs: Dual<B>) -> Dual<B> {
        let primal = B::float_div(lhs.primal.clone(), rhs.primal.clone());
        let shape = primal.shape();
        let tangent = match (lhs.tangent, rhs.tangent) {
            (Some(lhs_tangent), Some(rhs_tangent)) => {
                let lhs_tangent = align_tangent::<B>(lhs_tangent, &shape);
                let rhs_tangent = align_tangent::<B>(rhs_tangent, &shape);
                let numerator = B::float_sub(
                    B::float_mul(lhs_tangent, rhs.primal.clone()),
                    B::float_mul(lhs.primal, rhs_tangent),
                );
                Some(B::float_div(
                    numerator,
                    B::float_mul(rhs.primal.clone(), rhs.primal),
                ))
            }
            (Some(lhs_tangent), None) => Some(B::float_div(
                align_tangent::<B>(lhs_tangent, &shape),
                rhs.primal,
            )),
            (None, Some(rhs_tangent)) => {
                let rhs_tangent = align_tangent::<B>(rhs_tangent, &shape);
                let numerator = B::float_neg(B::float_mul(lhs.primal, rhs_tangent));
                Some(B::float_div(
                    numerator,
                    B::float_mul(rhs.primal.clone(), rhs.primal),
                ))
            }
            (None, None) => None,
        };
        DualFloatTensor { primal, tangent }
    }

    fn float_div_scalar(lhs: Dual<B>, rhs: Scalar) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_div_scalar(lhs.primal, rhs),
            tangent: lhs.tangent.map(|tangent| B::float_div_scalar(tangent, rhs)),
        }
    }

    fn float_remainder(lhs: Dual<B>, rhs: Dual<B>) -> Dual<B> {
        DualFloatTensor::constant(B::float_remainder(lhs.primal, rhs.primal))
    }

    fn float_remainder_scalar(lhs: Dual<B>, rhs: Scalar) -> Dual<B> {
        DualFloatTensor::constant(B::float_remainder_scalar(lhs.primal, rhs))
    }

    fn float_fmod(lhs: Dual<B>, rhs: Dual<B>) -> Dual<B> {
        DualFloatTensor::constant(B::float_fmod(lhs.primal, rhs.primal))
    }

    fn float_fmod_scalar(lhs: Dual<B>, rhs: Scalar) -> Dual<B> {
        DualFloatTensor::constant(B::float_fmod_scalar(lhs.primal, rhs))
    }

    fn float_matmul(lhs: Dual<B>, rhs: Dual<B>) -> Dual<B> {
        let primal = B::float_matmul(lhs.primal.clone(), rhs.primal.clone());
        let tangent = match (lhs.tangent, rhs.tangent) {
            (Some(lhs_tangent), Some(rhs_tangent)) => Some(B::float_add(
                B::float_matmul(lhs_tangent, rhs.primal),
                B::float_matmul(lhs.primal, rhs_tangent),
            )),
            (Some(lhs_tangent), None) => Some(B::float_matmul(lhs_tangent, rhs.primal)),
            (None, Some(rhs_tangent)) => Some(B::float_matmul(lhs.primal, rhs_tangent)),
            (None, None) => None,
        };
        DualFloatTensor { primal, tangent }
    }

    fn float_cross(lhs: Dual<B>, rhs: Dual<B>, dim: usize) -> Dual<B> {
        let primal = B::float_cross(lhs.primal.clone(), rhs.primal.clone(), dim);
        let tangent = match (lhs.tangent, rhs.tangent) {
            (Some(lhs_tangent), Some(rhs_tangent)) => Some(B::float_add(
                B::float_cross(lhs_tangent, rhs.primal, dim),
                B::float_cross(lhs.primal, rhs_tangent, dim),
            )),
            (Some(lhs_tangent), None) => Some(B::float_cross(lhs_tangent, rhs.primal, dim)),
            (None, Some(rhs_tangent)) => Some(B::float_cross(lhs.primal, rhs_tangent, dim)),
            (None, None) => None,
        };
        DualFloatTensor { primal, tangent }
    }

    fn float_recip(tensor: Dual<B>) -> Dual<B> {
        let primal = B::float_recip(tensor.primal.clone());
        let tangent = tensor.tangent.map(|tangent| {
            B::float_neg(B::float_div(
                tangent,
                B::float_mul(tensor.primal.clone(), tensor.primal),
            ))
        });
        DualFloatTensor { primal, tangent }
    }

    fn float_swap_dims(tensor: Dual<B>, dim1: usize, dim2: usize) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_swap_dims(tensor.primal, dim1, dim2),
            tangent: tensor
                .tangent
                .map(|tangent| B::float_swap_dims(tangent, dim1, dim2)),
        }
    }

    fn float_permute(tensor: Dual<B>, axes: &[usize]) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_permute(tensor.primal, axes),
            tangent: tensor
                .tangent
                .map(|tangent| B::float_permute(tangent, axes)),
        }
    }

    fn float_flip(tensor: Dual<B>, axes: &[usize]) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_flip(tensor.primal, axes),
            tangent: tensor.tangent.map(|tangent| B::float_flip(tangent, axes)),
        }
    }

    fn float_reshape(tensor: Dual<B>, shape: Shape) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_reshape(tensor.primal, shape.clone()),
            tangent: tensor
                .tangent
                .map(|tangent| B::float_reshape(tangent, shape)),
        }
    }

    fn float_gather(dim: usize, tensor: Dual<B>, indices: IntTensor<B>) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_gather(dim, tensor.primal, indices.clone()),
            tangent: tensor
                .tangent
                .map(|tangent| B::float_gather(dim, tangent, indices)),
        }
    }

    fn float_scatter(
        dim: usize,
        tensor: Dual<B>,
        indices: IntTensor<B>,
        value: Dual<B>,
        update: burn_backend::tensor::IndexingUpdateOp,
    ) -> Dual<B> {
        let tensor_shape = tensor.primal.shape();
        let tensor_device = tensor.primal.device();
        let tensor_dtype = tensor.primal.dtype();
        let value_shape = value.primal.shape();
        let value_device = value.primal.device();
        let value_dtype = value.primal.dtype();
        let primal = B::float_scatter(dim, tensor.primal, indices.clone(), value.primal, update);
        let tangent = match (tensor.tangent, value.tangent) {
            (Some(tensor_tangent), Some(value_tangent)) => Some(B::float_scatter(
                dim,
                tensor_tangent,
                indices,
                value_tangent,
                update,
            )),
            (Some(tensor_tangent), None) => {
                let zero = B::float_zeros(value_shape, &value_device, value_dtype.into());
                Some(B::float_scatter(dim, tensor_tangent, indices, zero, update))
            }
            (None, Some(value_tangent)) => {
                let zero = B::float_zeros(tensor_shape, &tensor_device, tensor_dtype.into());
                Some(B::float_scatter(dim, zero, indices, value_tangent, update))
            }
            (None, None) => None,
        };
        DualFloatTensor { primal, tangent }
    }

    fn float_select(tensor: Dual<B>, dim: usize, indices: IntTensor<B>) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_select(tensor.primal, dim, indices.clone()),
            tangent: tensor
                .tangent
                .map(|tangent| B::float_select(tangent, dim, indices)),
        }
    }

    fn float_select_assign(
        tensor: Dual<B>,
        dim: usize,
        indices: IntTensor<B>,
        value: Dual<B>,
        update: burn_backend::tensor::IndexingUpdateOp,
    ) -> Dual<B> {
        let tensor_shape = tensor.primal.shape();
        let tensor_device = tensor.primal.device();
        let tensor_dtype = tensor.primal.dtype();
        let value_shape = value.primal.shape();
        let value_device = value.primal.device();
        let value_dtype = value.primal.dtype();
        let primal =
            B::float_select_assign(tensor.primal, dim, indices.clone(), value.primal, update);
        let tangent = match (tensor.tangent, value.tangent) {
            (Some(tensor_tangent), Some(value_tangent)) => Some(B::float_select_assign(
                tensor_tangent,
                dim,
                indices,
                value_tangent,
                update,
            )),
            (Some(tensor_tangent), None) => {
                let zero = B::float_zeros(value_shape, &value_device, value_dtype.into());
                Some(B::float_select_assign(
                    tensor_tangent,
                    dim,
                    indices,
                    zero,
                    update,
                ))
            }
            (None, Some(value_tangent)) => {
                let zero = B::float_zeros(tensor_shape, &tensor_device, tensor_dtype.into());
                Some(B::float_select_assign(
                    zero,
                    dim,
                    indices,
                    value_tangent,
                    update,
                ))
            }
            (None, None) => None,
        };
        DualFloatTensor { primal, tangent }
    }

    fn float_slice(tensor: Dual<B>, slices: &[Slice]) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_slice(tensor.primal, slices),
            tangent: tensor
                .tangent
                .map(|tangent| B::float_slice(tangent, slices)),
        }
    }

    fn float_slice_assign(tensor: Dual<B>, slices: &[Slice], value: Dual<B>) -> Dual<B> {
        let tensor_shape = tensor.primal.shape();
        let tensor_device = tensor.primal.device();
        let tensor_dtype = tensor.primal.dtype();
        let value_shape = value.primal.shape();
        let value_device = value.primal.device();
        let value_dtype = value.primal.dtype();
        let primal = B::float_slice_assign(tensor.primal, slices, value.primal);
        let tangent = match (tensor.tangent, value.tangent) {
            (Some(tensor_tangent), Some(value_tangent)) => {
                Some(B::float_slice_assign(tensor_tangent, slices, value_tangent))
            }
            (Some(tensor_tangent), None) => {
                let zero = B::float_zeros(value_shape, &value_device, value_dtype.into());
                Some(B::float_slice_assign(tensor_tangent, slices, zero))
            }
            (None, Some(value_tangent)) => {
                let zero = B::float_zeros(tensor_shape, &tensor_device, tensor_dtype.into());
                Some(B::float_slice_assign(zero, slices, value_tangent))
            }
            (None, None) => None,
        };
        DualFloatTensor { primal, tangent }
    }

    fn float_mask_where(tensor: Dual<B>, mask: BoolTensor<B>, value: Dual<B>) -> Dual<B> {
        let primal = B::float_mask_where(tensor.primal, mask.clone(), value.primal);
        let primal_shape = primal.shape();
        let primal_device = primal.device();
        let primal_dtype = primal.dtype();
        let tangent = match (tensor.tangent, value.tangent) {
            (Some(tensor_tangent), Some(value_tangent)) => {
                let tensor_tangent = align_tangent::<B>(tensor_tangent, &primal_shape);
                let value_tangent = align_tangent::<B>(value_tangent, &primal_shape);
                Some(B::float_mask_where(tensor_tangent, mask, value_tangent))
            }
            (Some(tensor_tangent), None) => {
                let tensor_tangent = align_tangent::<B>(tensor_tangent, &primal_shape);
                let zero = B::float_zeros(primal_shape, &primal_device, primal_dtype.into());
                Some(B::float_mask_where(tensor_tangent, mask, zero))
            }
            (None, Some(value_tangent)) => {
                let value_tangent = align_tangent::<B>(value_tangent, &primal_shape);
                let zero = B::float_zeros(primal_shape, &primal_device, primal_dtype.into());
                Some(B::float_mask_where(zero, mask, value_tangent))
            }
            (None, None) => None,
        };
        DualFloatTensor { primal, tangent }
    }

    fn float_mask_fill(tensor: Dual<B>, mask: BoolTensor<B>, value: Scalar) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_mask_fill(tensor.primal, mask.clone(), value),
            tangent: tensor
                .tangent
                .map(|tangent| B::float_mask_fill(tangent, mask, scalar_zero())),
        }
    }

    fn float_sum(tensor: Dual<B>) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_sum(tensor.primal),
            tangent: tensor.tangent.map(B::float_sum),
        }
    }

    fn float_sum_dim(tensor: Dual<B>, dim: usize) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_sum_dim(tensor.primal, dim),
            tangent: tensor.tangent.map(|tangent| B::float_sum_dim(tangent, dim)),
        }
    }

    fn float_mean_dim(tensor: Dual<B>, dim: usize) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_mean_dim(tensor.primal, dim),
            tangent: tensor
                .tangent
                .map(|tangent| B::float_mean_dim(tangent, dim)),
        }
    }

    fn float_cumsum(tensor: Dual<B>, dim: usize) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_cumsum(tensor.primal, dim),
            tangent: tensor.tangent.map(|tangent| B::float_cumsum(tangent, dim)),
        }
    }

    fn float_cumprod(tensor: Dual<B>, dim: usize) -> Dual<B> {
        let primal = B::float_cumprod(tensor.primal.clone(), dim);
        let tangent = tensor.tangent.map(|dx| {
            let device = tensor.primal.device();
            let bool_dtype = bool_dtype::<B>(&device);
            let float_dtype = tensor.primal.dtype().into();

            // zeros_seen[i] counts zeros in x[0..=i].
            let zero_mask = B::bool_into_float(
                B::float_equal_elem(tensor.primal.clone(), 0.into(), bool_dtype),
                float_dtype,
            );
            let zeros_seen = B::float_cumsum(zero_mask.clone(), dim);
            let before_first = B::bool_into_float(
                B::float_lower_elem(zeros_seen.clone(), 0.5.into(), bool_dtype),
                float_dtype,
            );
            let at_first = B::float_mul(
                zero_mask.clone(),
                B::bool_into_float(
                    B::float_lower_elem(zeros_seen, 1.5.into(), bool_dtype),
                    float_dtype,
                ),
            );

            // Classic case: dy = y * cumsum(dx / x_safe).
            let x_safe = B::float_add(tensor.primal.clone(), zero_mask);
            let quotient = B::float_div(dx.clone(), x_safe);
            let cumquot = B::float_cumsum(quotient, dim);
            let classic = B::float_mul(primal.clone(), cumquot);
            let term_before = B::float_mul(classic, before_first);

            // First-zero and after: dy[i >= j0] = cumprod(x_one) * cumsum(at_first * dx)
            let x_one = B::float_add(tensor.primal, at_first.clone());
            let cumprod_one = B::float_cumprod(x_one, dim);
            let dx_at_first = B::float_mul(at_first, dx);
            let dx_first_propagated = B::float_cumsum(dx_at_first, dim);
            let term_at_and_after = B::float_mul(cumprod_one, dx_first_propagated);

            B::float_add(term_before, term_at_and_after)
        });
        DualFloatTensor { primal, tangent }
    }

    fn float_cummin(tensor: Dual<B>, dim: usize) -> Dual<B> {
        let primal = B::float_cummin(tensor.primal.clone(), dim);
        let tangent = tensor.tangent.map(|dx| {
            let device = tensor.primal.device();
            let is_source = B::float_equal(
                primal.clone(),
                tensor.primal.clone(),
                bool_dtype::<B>(&device),
            );
            let shape = tensor.primal.shape();
            let dim_size = shape[dim];
            let mut arange_dims = alloc::vec![1usize; shape.num_dims()];
            arange_dims[dim] = dim_size;
            let arange = B::int_reshape(
                B::int_arange(0..dim_size as i64, &device, int_dtype::<B>(&device)),
                Shape::from(arange_dims),
            );
            let arange = B::int_expand(arange, shape);
            let masked = B::int_mul(arange, B::bool_into_int(is_source, int_dtype::<B>(&device)));
            let source_idx = B::int_cummax(masked, dim);
            B::float_gather(dim, dx, source_idx)
        });
        DualFloatTensor { primal, tangent }
    }

    fn float_cummax(tensor: Dual<B>, dim: usize) -> Dual<B> {
        let primal = B::float_cummax(tensor.primal.clone(), dim);
        let tangent = tensor.tangent.map(|dx| {
            let device = tensor.primal.device();
            let is_source = B::float_equal(
                primal.clone(),
                tensor.primal.clone(),
                bool_dtype::<B>(&device),
            );
            let shape = tensor.primal.shape();
            let dim_size = shape[dim];
            let mut arange_dims = alloc::vec![1usize; shape.num_dims()];
            arange_dims[dim] = dim_size;
            let arange = B::int_reshape(
                B::int_arange(0..dim_size as i64, &device, int_dtype::<B>(&device)),
                Shape::from(arange_dims),
            );
            let arange = B::int_expand(arange, shape);
            let masked = B::int_mul(arange, B::bool_into_int(is_source, int_dtype::<B>(&device)));
            let source_idx = B::int_cummax(masked, dim);
            B::float_gather(dim, dx, source_idx)
        });
        DualFloatTensor { primal, tangent }
    }

    fn float_cast(tensor: Dual<B>, dtype: FloatDType) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_cast(tensor.primal, dtype),
            tangent: tensor.tangent.map(|tangent| B::float_cast(tangent, dtype)),
        }
    }

    fn float_exp(tensor: Dual<B>) -> Dual<B> {
        let primal = B::float_exp(tensor.primal.clone());
        let tangent = tensor
            .tangent
            .map(|tangent| B::float_mul(tangent, primal.clone()));
        DualFloatTensor { primal, tangent }
    }

    fn float_log(tensor: Dual<B>) -> Dual<B> {
        let primal = B::float_log(tensor.primal.clone());
        let tangent = tensor
            .tangent
            .map(|tangent| B::float_div(tangent, tensor.primal));
        DualFloatTensor { primal, tangent }
    }

    fn float_log1p(tensor: Dual<B>) -> Dual<B> {
        let primal = B::float_log1p(tensor.primal.clone());
        let tangent = tensor.tangent.map(|tangent| {
            B::float_div(
                tangent,
                B::float_add_scalar(tensor.primal.clone(), Scalar::from(1.0f64)),
            )
        });
        DualFloatTensor { primal, tangent }
    }

    fn float_powf(lhs: Dual<B>, rhs: Dual<B>) -> Dual<B> {
        let primal = B::float_powf(lhs.primal.clone(), rhs.primal.clone());
        let shape = primal.shape();
        let tangent = match (lhs.tangent, rhs.tangent) {
            (Some(lhs_tangent), Some(rhs_tangent)) => {
                let lhs_tangent = align_tangent::<B>(lhs_tangent, &shape);
                let rhs_tangent = align_tangent::<B>(rhs_tangent, &shape);
                let rhs_minus_one = B::float_sub_scalar(rhs.primal.clone(), Scalar::from(1.0f64));
                let d_lhs = B::float_mul(
                    lhs_tangent,
                    B::float_mul(
                        rhs.primal.clone(),
                        B::float_powf(lhs.primal.clone(), rhs_minus_one),
                    ),
                );
                let rhs_term = B::float_mul(
                    primal.clone(),
                    B::float_mul(rhs_tangent, B::float_log(lhs.primal)),
                );
                Some(B::float_add(d_lhs, rhs_term))
            }
            (Some(lhs_tangent), None) => {
                let lhs_tangent = align_tangent::<B>(lhs_tangent, &shape);
                let rhs_minus_one = B::float_sub_scalar(rhs.primal.clone(), Scalar::from(1.0f64));
                Some(B::float_mul(
                    lhs_tangent,
                    B::float_mul(rhs.primal, B::float_powf(lhs.primal, rhs_minus_one)),
                ))
            }
            (None, Some(rhs_tangent)) => {
                let rhs_tangent = align_tangent::<B>(rhs_tangent, &shape);
                Some(B::float_mul(
                    primal.clone(),
                    B::float_mul(rhs_tangent, B::float_log(lhs.primal)),
                ))
            }
            (None, None) => None,
        };
        DualFloatTensor { primal, tangent }
    }

    fn float_powf_scalar_impl(tensor: Dual<B>, value: Scalar) -> Dual<B> {
        let exponent = value.elem::<f64>();
        let primal = B::float_powf_scalar_impl(tensor.primal.clone(), value);
        let tangent = if exponent == 0.0 {
            tensor.tangent.map(|_| zero_like::<B>(&primal))
        } else {
            tensor.tangent.map(|tangent| {
                let power =
                    B::float_powf_scalar_impl(tensor.primal.clone(), Scalar::from(exponent - 1.0));
                B::float_mul(B::float_mul_scalar(tangent, Scalar::from(exponent)), power)
            })
        };
        DualFloatTensor { primal, tangent }
    }

    fn float_sqrt(tensor: Dual<B>) -> Dual<B> {
        let primal = B::float_sqrt(tensor.primal.clone());
        let tangent = tensor.tangent.map(|tangent| {
            B::float_div(
                tangent,
                B::float_mul_scalar(primal.clone(), Scalar::from(2.0f64)),
            )
        });
        DualFloatTensor { primal, tangent }
    }

    fn float_abs(tensor: Dual<B>) -> Dual<B> {
        let primal = B::float_abs(tensor.primal.clone());
        let tangent = tensor
            .tangent
            .map(|tangent| B::float_mul(tangent, B::float_sign(tensor.primal)));
        DualFloatTensor { primal, tangent }
    }

    fn float_cos(tensor: Dual<B>) -> Dual<B> {
        let primal = B::float_cos(tensor.primal.clone());
        let tangent = tensor
            .tangent
            .map(|tangent| B::float_neg(B::float_mul(tangent, B::float_sin(tensor.primal))));
        DualFloatTensor { primal, tangent }
    }

    fn float_sin(tensor: Dual<B>) -> Dual<B> {
        let primal = B::float_sin(tensor.primal.clone());
        let tangent = tensor
            .tangent
            .map(|tangent| B::float_mul(tangent, B::float_cos(tensor.primal)));
        DualFloatTensor { primal, tangent }
    }

    fn float_tan(tensor: Dual<B>) -> Dual<B> {
        let primal = B::float_tan(tensor.primal.clone());
        let tangent = tensor.tangent.map(|tangent| {
            B::float_mul(
                tangent,
                B::float_add_scalar(
                    B::float_mul(primal.clone(), primal.clone()),
                    Scalar::from(1.0f64),
                ),
            )
        });
        DualFloatTensor { primal, tangent }
    }

    fn float_cosh(tensor: Dual<B>) -> Dual<B> {
        let primal = B::float_cosh(tensor.primal.clone());
        let tangent = tensor
            .tangent
            .map(|tangent| B::float_mul(tangent, B::float_sinh(tensor.primal)));
        DualFloatTensor { primal, tangent }
    }

    fn float_sinh(tensor: Dual<B>) -> Dual<B> {
        let primal = B::float_sinh(tensor.primal.clone());
        let tangent = tensor
            .tangent
            .map(|tangent| B::float_mul(tangent, B::float_cosh(tensor.primal)));
        DualFloatTensor { primal, tangent }
    }

    fn float_tanh(tensor: Dual<B>) -> Dual<B> {
        let primal = B::float_tanh(tensor.primal.clone());
        let tangent = tensor.tangent.map(|tangent| {
            let square = B::float_mul(primal.clone(), primal.clone());
            let one_minus_square = B::float_sub(
                B::float_ones(square.shape(), &square.device(), square.dtype().into()),
                square,
            );
            B::float_mul(tangent, one_minus_square)
        });
        DualFloatTensor { primal, tangent }
    }

    fn float_acos(tensor: Dual<B>) -> Dual<B> {
        let primal = B::float_acos(tensor.primal.clone());
        let tangent = tensor.tangent.map(|tangent| {
            let square = B::float_mul(tensor.primal.clone(), tensor.primal.clone());
            let one_minus_square = B::float_sub(
                B::float_ones(square.shape(), &square.device(), square.dtype().into()),
                square,
            );
            B::float_neg(B::float_div(tangent, B::float_sqrt(one_minus_square)))
        });
        DualFloatTensor { primal, tangent }
    }

    fn float_acosh(tensor: Dual<B>) -> Dual<B> {
        let primal = B::float_acosh(tensor.primal.clone());
        let tangent = tensor.tangent.map(|tangent| {
            let square_minus_one = B::float_sub_scalar(
                B::float_mul(tensor.primal.clone(), tensor.primal),
                Scalar::from(1.0f64),
            );
            B::float_div(tangent, B::float_sqrt(square_minus_one))
        });
        DualFloatTensor { primal, tangent }
    }

    fn float_asin(tensor: Dual<B>) -> Dual<B> {
        let primal = B::float_asin(tensor.primal.clone());
        let tangent = tensor.tangent.map(|tangent| {
            let square = B::float_mul(tensor.primal.clone(), tensor.primal.clone());
            let one_minus_square = B::float_sub(
                B::float_ones(square.shape(), &square.device(), square.dtype().into()),
                square,
            );
            B::float_div(tangent, B::float_sqrt(one_minus_square))
        });
        DualFloatTensor { primal, tangent }
    }

    fn float_asinh(tensor: Dual<B>) -> Dual<B> {
        let primal = B::float_asinh(tensor.primal.clone());
        let tangent = tensor.tangent.map(|tangent| {
            let square_plus_one = B::float_add_scalar(
                B::float_mul(tensor.primal.clone(), tensor.primal),
                Scalar::from(1.0f64),
            );
            B::float_div(tangent, B::float_sqrt(square_plus_one))
        });
        DualFloatTensor { primal, tangent }
    }

    fn float_atan(tensor: Dual<B>) -> Dual<B> {
        let primal = B::float_atan(tensor.primal.clone());
        let tangent = tensor.tangent.map(|tangent| {
            let denominator = B::float_add_scalar(
                B::float_mul(tensor.primal.clone(), tensor.primal),
                Scalar::from(1.0f64),
            );
            B::float_div(tangent, denominator)
        });
        DualFloatTensor { primal, tangent }
    }

    fn float_atanh(tensor: Dual<B>) -> Dual<B> {
        let primal = B::float_atanh(tensor.primal.clone());
        let tangent = tensor.tangent.map(|tangent| {
            let denominator = B::float_sub(
                B::float_ones(
                    tensor.primal.shape(),
                    &tensor.primal.device(),
                    tensor.primal.dtype().into(),
                ),
                B::float_mul(tensor.primal.clone(), tensor.primal),
            );
            B::float_div(tangent, denominator)
        });
        DualFloatTensor { primal, tangent }
    }

    fn float_atan2(lhs: Dual<B>, rhs: Dual<B>) -> Dual<B> {
        let primal = B::float_atan2(lhs.primal.clone(), rhs.primal.clone());
        let shape = primal.shape();
        let tangent = match (lhs.tangent, rhs.tangent) {
            (Some(lhs_tangent), Some(rhs_tangent)) => {
                let lhs_tangent = align_tangent::<B>(lhs_tangent, &shape);
                let rhs_tangent = align_tangent::<B>(rhs_tangent, &shape);
                let numerator = B::float_sub(
                    B::float_mul(rhs.primal.clone(), lhs_tangent),
                    B::float_mul(lhs.primal.clone(), rhs_tangent),
                );
                let denominator = B::float_add(
                    B::float_mul(lhs.primal.clone(), lhs.primal),
                    B::float_mul(rhs.primal.clone(), rhs.primal),
                );
                Some(B::float_div(numerator, denominator))
            }
            (Some(lhs_tangent), None) => {
                let lhs_tangent = align_tangent::<B>(lhs_tangent, &shape);
                let denominator = B::float_add(
                    B::float_mul(lhs.primal.clone(), lhs.primal),
                    B::float_mul(rhs.primal.clone(), rhs.primal.clone()),
                );
                Some(B::float_div(
                    B::float_mul(rhs.primal, lhs_tangent),
                    denominator,
                ))
            }
            (None, Some(rhs_tangent)) => {
                let rhs_tangent = align_tangent::<B>(rhs_tangent, &shape);
                let denominator = B::float_add(
                    B::float_mul(lhs.primal.clone(), lhs.primal.clone()),
                    B::float_mul(rhs.primal.clone(), rhs.primal),
                );
                Some(B::float_neg(B::float_div(
                    B::float_mul(lhs.primal, rhs_tangent),
                    denominator,
                )))
            }
            (None, None) => None,
        };
        DualFloatTensor { primal, tangent }
    }

    fn float_round(tensor: Dual<B>) -> Dual<B> {
        DualFloatTensor::constant(B::float_round(tensor.primal))
    }

    fn float_floor(tensor: Dual<B>) -> Dual<B> {
        DualFloatTensor::constant(B::float_floor(tensor.primal))
    }

    fn float_ceil(tensor: Dual<B>) -> Dual<B> {
        DualFloatTensor::constant(B::float_ceil(tensor.primal))
    }

    fn float_trunc(tensor: Dual<B>) -> Dual<B> {
        DualFloatTensor::constant(B::float_trunc(tensor.primal))
    }

    fn float_erf(tensor: Dual<B>) -> Dual<B> {
        let primal = B::float_erf(tensor.primal.clone());
        let tangent = tensor.tangent.map(|tangent| {
            let square = B::float_mul(tensor.primal.clone(), tensor.primal);
            let gaussian = B::float_exp(B::float_neg(square));
            let factor = Scalar::from(2.0f64 / core::f64::consts::PI.sqrt());
            B::float_mul(B::float_mul_scalar(tangent, factor), gaussian)
        });
        DualFloatTensor { primal, tangent }
    }

    fn float_clamp_min(tensor: Dual<B>, min: Scalar) -> Dual<B> {
        let dtype = bool_dtype::<B>(&tensor.primal.device());
        let mask = B::float_lower_elem(tensor.primal.clone(), min, dtype);
        let primal = B::float_mask_fill(tensor.primal, mask.clone(), min);
        let tangent = tensor
            .tangent
            .map(|tangent| B::float_mask_fill(tangent, mask, scalar_zero()));
        DualFloatTensor { primal, tangent }
    }

    fn float_clamp_max(tensor: Dual<B>, max: Scalar) -> Dual<B> {
        let dtype = bool_dtype::<B>(&tensor.primal.device());
        let mask = B::float_greater_elem(tensor.primal.clone(), max, dtype);
        let primal = B::float_mask_fill(tensor.primal, mask.clone(), max);
        let tangent = tensor
            .tangent
            .map(|tangent| B::float_mask_fill(tangent, mask, scalar_zero()));
        DualFloatTensor { primal, tangent }
    }

    fn float_clamp(tensor: Dual<B>, min: Scalar, max: Scalar) -> Dual<B> {
        let dtype = bool_dtype::<B>(&tensor.primal.device());
        let lower = B::float_lower_elem(tensor.primal.clone(), min, dtype);
        let upper = B::float_greater_elem(tensor.primal.clone(), max, dtype);
        let mask = B::bool_or(lower, upper);
        let primal = B::float_clamp(tensor.primal, min, max);
        let tangent = tensor
            .tangent
            .map(|tangent| B::float_mask_fill(tangent, mask, scalar_zero()));
        DualFloatTensor { primal, tangent }
    }

    fn float_argmax(tensor: Dual<B>, dim: usize, out_dtype: IntDType) -> IntTensor<B> {
        B::float_argmax(tensor.primal, dim, out_dtype)
    }

    fn float_argmin(tensor: Dual<B>, dim: usize, out_dtype: IntDType) -> IntTensor<B> {
        B::float_argmin(tensor.primal, dim, out_dtype)
    }

    fn float_max_dim(tensor: Dual<B>, dim: usize) -> Dual<B> {
        let dtype = int_dtype::<B>(&tensor.primal.device());
        let (primal, indices) = B::float_max_dim_with_indices(tensor.primal, dim, dtype);
        let tangent = tensor
            .tangent
            .map(|tangent| B::float_gather(dim, tangent, indices));
        DualFloatTensor { primal, tangent }
    }

    fn float_max_dim_with_indices(
        tensor: Dual<B>,
        dim: usize,
        indices_dtype: IntDType,
    ) -> (Dual<B>, IntTensor<B>) {
        let (primal, indices) = B::float_max_dim_with_indices(tensor.primal, dim, indices_dtype);
        let tangent = tensor
            .tangent
            .map(|tangent| B::float_gather(dim, tangent, indices.clone()));
        (DualFloatTensor { primal, tangent }, indices)
    }

    fn float_min_dim(tensor: Dual<B>, dim: usize) -> Dual<B> {
        let dtype = int_dtype::<B>(&tensor.primal.device());
        let (primal, indices) = B::float_min_dim_with_indices(tensor.primal, dim, dtype);
        let tangent = tensor
            .tangent
            .map(|tangent| B::float_gather(dim, tangent, indices));
        DualFloatTensor { primal, tangent }
    }

    fn float_min_dim_with_indices(
        tensor: Dual<B>,
        dim: usize,
        indices_dtype: IntDType,
    ) -> (Dual<B>, IntTensor<B>) {
        let (primal, indices) = B::float_min_dim_with_indices(tensor.primal, dim, indices_dtype);
        let tangent = tensor
            .tangent
            .map(|tangent| B::float_gather(dim, tangent, indices.clone()));
        (DualFloatTensor { primal, tangent }, indices)
    }

    fn float_sort(tensor: Dual<B>, dim: usize, descending: bool) -> Dual<B> {
        let dtype = int_dtype::<B>(&tensor.primal.device());
        let indices = B::float_argsort(tensor.primal.clone(), dim, descending, dtype);
        let primal = B::float_sort(tensor.primal, dim, descending);
        let tangent = tensor
            .tangent
            .map(|tangent| B::float_gather(dim, tangent, indices));
        DualFloatTensor { primal, tangent }
    }

    fn float_sort_with_indices(
        tensor: Dual<B>,
        dim: usize,
        descending: bool,
        indices_dtype: IntDType,
    ) -> (Dual<B>, IntTensor<B>) {
        let (primal, indices) =
            B::float_sort_with_indices(tensor.primal, dim, descending, indices_dtype);
        let tangent = tensor
            .tangent
            .map(|tangent| B::float_gather(dim, tangent, indices.clone()));
        (DualFloatTensor { primal, tangent }, indices)
    }

    fn float_argsort(
        tensor: Dual<B>,
        dim: usize,
        descending: bool,
        out_dtype: IntDType,
    ) -> IntTensor<B> {
        B::float_argsort(tensor.primal, dim, descending, out_dtype)
    }

    fn float_sign(tensor: Dual<B>) -> Dual<B> {
        DualFloatTensor::constant(B::float_sign(tensor.primal))
    }

    fn float_expand(tensor: Dual<B>, shape: Shape) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_expand(tensor.primal, shape.clone()),
            tangent: tensor
                .tangent
                .map(|tangent| B::float_expand(tangent, shape)),
        }
    }

    fn float_unfold(tensor: Dual<B>, dim: usize, size: usize, step: usize) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_unfold(tensor.primal, dim, size, step),
            tangent: tensor
                .tangent
                .map(|tangent| B::float_unfold(tangent, dim, size, step)),
        }
    }

    fn float_cat(tensors: Vec<Dual<B>>, dim: usize) -> Dual<B> {
        let any_tangent = tensors.iter().any(|tensor| tensor.tangent.is_some());
        let primals = tensors.iter().map(|tensor| tensor.primal.clone()).collect();
        let primal = B::float_cat(primals, dim);
        let tangent = if any_tangent {
            let tangents = tensors
                .into_iter()
                .map(|tensor| {
                    tensor
                        .tangent
                        .unwrap_or_else(|| zero_like::<B>(&tensor.primal))
                })
                .collect();
            Some(B::float_cat(tangents, dim))
        } else {
            None
        };
        DualFloatTensor { primal, tangent }
    }

    fn float_pad(tensor: Dual<B>, padding: &[(usize, usize)], mode: PadMode) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_pad(tensor.primal, padding, mode),
            tangent: tensor
                .tangent
                .map(|tangent| B::float_pad(tangent, padding, mode)),
        }
    }

    fn float_repeat_dim(tensor: Dual<B>, dim: usize, times: usize) -> Dual<B> {
        DualFloatTensor {
            primal: B::float_repeat_dim(tensor.primal, dim, times),
            tangent: tensor
                .tangent
                .map(|tangent| B::float_repeat_dim(tangent, dim, times)),
        }
    }

    fn float_detach(tensor: Dual<B>) -> Dual<B> {
        DualFloatTensor::constant(B::float_detach(tensor.primal))
    }

    fn float_set_require_grad(tensor: Dual<B>, _require_grad: bool) -> Dual<B> {
        tensor
    }

    fn float_is_require_grad(_tensor: &Dual<B>) -> bool {
        false
    }

    fn float_equal(lhs: Dual<B>, rhs: Dual<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        B::float_equal(lhs.primal, rhs.primal, out_dtype)
    }

    fn float_not_equal(lhs: Dual<B>, rhs: Dual<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        B::float_not_equal(lhs.primal, rhs.primal, out_dtype)
    }

    fn float_equal_elem(lhs: Dual<B>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<B> {
        B::float_equal_elem(lhs.primal, rhs, out_dtype)
    }

    fn float_not_equal_elem(lhs: Dual<B>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<B> {
        B::float_not_equal_elem(lhs.primal, rhs, out_dtype)
    }

    fn float_greater(lhs: Dual<B>, rhs: Dual<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        B::float_greater(lhs.primal, rhs.primal, out_dtype)
    }

    fn float_greater_elem(lhs: Dual<B>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<B> {
        B::float_greater_elem(lhs.primal, rhs, out_dtype)
    }

    fn float_greater_equal(lhs: Dual<B>, rhs: Dual<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        B::float_greater_equal(lhs.primal, rhs.primal, out_dtype)
    }

    fn float_greater_equal_elem(lhs: Dual<B>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<B> {
        B::float_greater_equal_elem(lhs.primal, rhs, out_dtype)
    }

    fn float_lower(lhs: Dual<B>, rhs: Dual<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        B::float_lower(lhs.primal, rhs.primal, out_dtype)
    }

    fn float_lower_elem(lhs: Dual<B>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<B> {
        B::float_lower_elem(lhs.primal, rhs, out_dtype)
    }

    fn float_lower_equal(lhs: Dual<B>, rhs: Dual<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        B::float_lower_equal(lhs.primal, rhs.primal, out_dtype)
    }

    fn float_lower_equal_elem(lhs: Dual<B>, rhs: Scalar, out_dtype: BoolDType) -> BoolTensor<B> {
        B::float_lower_equal_elem(lhs.primal, rhs, out_dtype)
    }

    fn float_any(tensor: Dual<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        B::float_any(tensor.primal, out_dtype)
    }

    fn float_any_dim(tensor: Dual<B>, dim: usize, out_dtype: BoolDType) -> BoolTensor<B> {
        B::float_any_dim(tensor.primal, dim, out_dtype)
    }

    fn float_all(tensor: Dual<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        B::float_all(tensor.primal, out_dtype)
    }

    fn float_all_dim(tensor: Dual<B>, dim: usize, out_dtype: BoolDType) -> BoolTensor<B> {
        B::float_all_dim(tensor.primal, dim, out_dtype)
    }

    fn float_is_nan(tensor: Dual<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        B::float_is_nan(tensor.primal, out_dtype)
    }

    fn float_is_inf(tensor: Dual<B>, out_dtype: BoolDType) -> BoolTensor<B> {
        B::float_is_inf(tensor.primal, out_dtype)
    }

    fn float_grid_sample_2d(
        tensor: Dual<B>,
        grid: Dual<B>,
        options: burn_backend::ops::GridSampleOptions,
    ) -> Dual<B> {
        burn_backend::ops::grid_sample::float_grid_sample_2d_ref::<ForwardDiff<B>>(
            tensor, grid, options,
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{ForwardDiff, FuncTensor, jvp, value_and_jvp, vjp};
    use alloc::boxed::Box;
    use alloc::vec;
    use alloc::vec::Vec;
    use burn_autodiff::Autodiff;
    use burn_backend::{Scalar, TensorData};
    use burn_flex::{Flex, FlexDevice};

    type AB = Autodiff<Flex>;
    type FD = ForwardDiff<Flex>;

    fn values(tensor: FuncTensor<Flex, 1>) -> Vec<f32> {
        tensor.into_data_sync().try_to_vec::<f32>().unwrap()
    }

    fn dot(lhs: FuncTensor<Flex, 2>, rhs: FuncTensor<Flex, 2>) -> f32 {
        let product = (lhs * rhs).sum();
        values(product)[0]
    }

    #[test]
    fn jvp_linear_function() {
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32]), &device);
        let v = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32]), &device);

        let f = |x: &FuncTensor<FD, 1>| (x.clone() * 3.0f64).sum();
        let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &x, &v);

        assert_eq!(
            tangent.into_data_sync().try_to_vec::<f32>().unwrap(),
            vec![3.0]
        );
    }

    #[test]
    fn value_and_jvp_broadcast() {
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 2>::from_data(
            TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]),
            &device,
        );
        let v = FuncTensor::<Flex, 2>::from_data(
            TensorData::from([[1.0f32, 0.0], [0.0, 1.0]]),
            &device,
        );
        let b = FuncTensor::<Flex, 2>::from_data(TensorData::from([[2.0f32, 3.0]]), &device);
        let vb = FuncTensor::<Flex, 2>::from_data(TensorData::from([[0.5f32, -0.5]]), &device);

        let f = |(x, b): &(FuncTensor<FD, 2>, FuncTensor<FD, 2>)| (x.clone() * b.clone()).sum();

        let (value, tangent): (FuncTensor<Flex, 1>, FuncTensor<Flex, 1>) =
            value_and_jvp::<Flex, _, _, _>(f, &(x, b), &(v, vb));

        assert_eq!(
            value.into_data_sync().try_to_vec::<f32>().unwrap(),
            vec![26.0]
        );
        assert_eq!(
            tangent.into_data_sync().try_to_vec::<f32>().unwrap(),
            vec![4.0]
        );
    }

    #[test]
    fn jvp_lower_rank_broadcast() {
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 2.0, 3.0]), &device);
        let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 0.0, 0.0]), &device);
        let b = FuncTensor::<Flex, 2>::from_data(
            TensorData::from([[1.0f32, 1.0, 1.0], [2.0, 2.0, 2.0]]),
            &device,
        );

        let f = |(x, b): &(FuncTensor<FD, 1>, FuncTensor<FD, 2>)| {
            (x.clone().reshape::<2, _>([1, 3]) + b.clone()).sum()
        };
        let (value, tangent): (FuncTensor<Flex, 1>, FuncTensor<Flex, 1>) =
            value_and_jvp::<Flex, _, _, _>(
                f,
                &(x, b),
                &(u, FuncTensor::<Flex, 2>::zeros([2, 3], &device)),
            );
        assert_eq!(
            value.into_data_sync().try_to_vec::<f32>().unwrap(),
            vec![21.0]
        );
        assert_eq!(
            tangent.into_data_sync().try_to_vec::<f32>().unwrap(),
            vec![2.0]
        );
    }

    #[test]
    fn jvp_arithmetic_and_reduction() {
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([0.5f32, -0.25]), &device);
        let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 2.0]), &device);

        let f = |x: &FuncTensor<FD, 1>| (x.clone() * x.clone() + x.clone().sin()).sum();
        let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &x, &u);

        let expected = (2.0 * 0.5 + 0.5f32.cos()) * 1.0 + (2.0 * -0.25 + (-0.25f32).cos()) * 2.0;
        assert!((values(tangent)[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn jvp_broadcasted_tangent() {
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 2>::from_data(
            TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]),
            &device,
        );
        let dx = FuncTensor::<Flex, 2>::from_data(
            TensorData::from([[1.0f32, 0.0], [0.0, 1.0]]),
            &device,
        );
        let b = FuncTensor::<Flex, 2>::from_data(TensorData::from([[2.0f32, 3.0]]), &device);
        let db = FuncTensor::<Flex, 2>::from_data(TensorData::from([[0.5f32, -0.5]]), &device);

        let f = |(x, b): &(FuncTensor<FD, 2>, FuncTensor<FD, 2>)| (x.clone() * b.clone()).sum();
        let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &(x, b), &(dx, db));

        assert!((values(tangent)[0] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn jvp_powf_negative_base_constant_exponent() {
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([-2.0f32]), &device);
        let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32]), &device);
        let exponent = FuncTensor::<FD, 1>::from_data(TensorData::from([3.0f32]), &device);

        let f = move |x: &FuncTensor<FD, 1>| x.clone().powf(exponent.clone()).sum();
        let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &x, &u);

        let value = values(tangent)[0];
        assert!(value.is_finite());
        assert!((value - 12.0).abs() < 1e-5);
    }

    #[test]
    fn adjoint_equivalence_for_matmul() {
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 2>::from_data(
            TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]),
            &device,
        );
        let w = FuncTensor::<Flex, 2>::from_data(
            TensorData::from([[0.5f32, -0.5], [1.0, 2.0]]),
            &device,
        );
        let dx = FuncTensor::<Flex, 2>::from_data(
            TensorData::from([[0.1f32, 0.2], [0.3, 0.4]]),
            &device,
        );
        let dw = FuncTensor::<Flex, 2>::from_data(
            TensorData::from([[0.5f32, 0.6], [0.7, 0.8]]),
            &device,
        );
        let cotangent = FuncTensor::<Flex, 2>::from_data(
            TensorData::from([[1.0f32, 2.0], [3.0, 4.0]]),
            &device,
        );

        let f = |(x, w): &(FuncTensor<AB, 2>, FuncTensor<AB, 2>)| x.clone().matmul(w.clone());

        let f_jvp = |(x, w): &(FuncTensor<FD, 2>, FuncTensor<FD, 2>)| x.clone().matmul(w.clone());
        let jvp_out: FuncTensor<Flex, 2> =
            jvp::<Flex, _, _, _>(f_jvp, &(x.clone(), w.clone()), &(dx.clone(), dw.clone()));

        let primals = (x, w);
        let (_, pullback) = vjp::<Flex, AB, _, FuncTensor<Flex, 2>, _>(f, &primals);
        let (grad_x, grad_w) = pullback(&cotangent);

        let lhs = dot(jvp_out, cotangent);
        let rhs = dot(dx, grad_x) + dot(dw, grad_w);

        assert!((lhs - rhs).abs() < 1e-5, "lhs={lhs}, rhs={rhs}");
    }

    #[test]
    fn value_and_jvp_returns_primal_and_tangent() {
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32]), &device);
        let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32]), &device);

        let f = |x: &FuncTensor<FD, 1>| (x.clone() * x.clone()).sum();
        let (primal, tangent): (FuncTensor<Flex, 1>, FuncTensor<Flex, 1>) =
            value_and_jvp::<Flex, _, _, _>(f, &x, &u);

        assert_eq!(
            primal.into_data_sync().try_to_vec::<f32>().unwrap(),
            vec![4.0]
        );
        assert_eq!(
            tangent.into_data_sync().try_to_vec::<f32>().unwrap(),
            vec![4.0]
        );
    }

    #[test]
    fn jvp_scalar_power() {
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32]), &device);
        let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32]), &device);
        let f = |x: &FuncTensor<FD, 1>| x.clone().powf_scalar(Scalar::from(3.0f64)).sum();
        let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &x, &u);
        assert!((values(tangent)[0] - 12.0).abs() < 1e-5);
    }

    #[test]
    #[allow(clippy::type_complexity)]
    fn jvp_unary_functions() {
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([0.5f32]), &device);
        let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32]), &device);

        let checks: Vec<(
            &str,
            Box<dyn Fn(&FuncTensor<FD, 1>) -> FuncTensor<FD, 1>>,
            f32,
        )> = vec![
            (
                "exp",
                Box::new(|x: &FuncTensor<FD, 1>| x.clone().exp().sum()),
                0.5f32.exp(),
            ),
            (
                "log",
                Box::new(|x: &FuncTensor<FD, 1>| x.clone().log().sum()),
                2.0,
            ),
            (
                "sqrt",
                Box::new(|x: &FuncTensor<FD, 1>| x.clone().sqrt().sum()),
                1.0 / (2.0 * 0.5f32.sqrt()),
            ),
            (
                "cos",
                Box::new(|x: &FuncTensor<FD, 1>| x.clone().cos().sum()),
                -0.5f32.sin(),
            ),
            (
                "tanh",
                Box::new(|x: &FuncTensor<FD, 1>| x.clone().tanh().sum()),
                1.0 - 0.5f32.tanh().powi(2),
            ),
            (
                "sin",
                Box::new(|x: &FuncTensor<FD, 1>| x.clone().sin().sum()),
                0.5f32.cos(),
            ),
            (
                "tan",
                Box::new(|x: &FuncTensor<FD, 1>| x.clone().tan().sum()),
                1.0 + 0.5f32.tan().powi(2),
            ),
            (
                "sinh",
                Box::new(|x: &FuncTensor<FD, 1>| x.clone().sinh().sum()),
                0.5f32.cosh(),
            ),
            (
                "cosh",
                Box::new(|x: &FuncTensor<FD, 1>| x.clone().cosh().sum()),
                0.5f32.sinh(),
            ),
            (
                "erf",
                Box::new(|x: &FuncTensor<FD, 1>| x.clone().erf().sum()),
                2.0 / core::f32::consts::PI.sqrt() * (-(0.5f32.powi(2))).exp(),
            ),
            (
                "abs",
                Box::new(|x: &FuncTensor<FD, 1>| x.clone().abs().sum()),
                1.0,
            ),
            (
                "recip",
                Box::new(|x: &FuncTensor<FD, 1>| x.clone().recip().sum()),
                -1.0 / (0.5f32.powi(2)),
            ),
        ];

        for (name, f, expected) in checks {
            let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(&f, &x, &u);
            let actual = values(tangent)[0];
            assert!(
                (actual - expected).abs() < 1e-4,
                "{name}: {actual} vs {expected}"
            );
        }
    }

    #[test]
    fn jvp_cumprod_matches_finite_difference() {
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32, 3.0, 4.0]), &device);
        let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 0.0, 0.0]), &device);
        let f = |x: &FuncTensor<FD, 1>| x.clone().cumprod(0).sum();
        let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &x, &u);
        assert!((values(tangent)[0] - 16.0).abs() < 1e-3);
    }

    #[test]
    fn jvp_cummin_matches_finite_difference() {
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([3.0f32, 1.0, 2.0]), &device);
        let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([0.0f32, 1.0, 0.0]), &device);
        let f = |x: &FuncTensor<FD, 1>| x.clone().cummin(0).sum();
        let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &x, &u);
        assert!((values(tangent)[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn jvp_cummax_matches_finite_difference() {
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 3.0, 2.0]), &device);
        let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([0.0f32, 1.0, 0.0]), &device);
        let f = |x: &FuncTensor<FD, 1>| x.clone().cummax(0).sum();
        let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &x, &u);
        assert!((values(tangent)[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn jvp_clamp() {
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([0.0f32, 0.5, 2.0]), &device);
        let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 1.0, 1.0]), &device);
        let f = |x: &FuncTensor<FD, 1>| x.clone().clamp(0.2f64.into(), 1.0f64.into()).sum();
        let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &x, &u);
        assert!((values(tangent)[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn jvp_max_dim_and_min_dim() {
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 2>::from_data(
            TensorData::from([[1.0f32, 5.0], [4.0, 2.0]]),
            &device,
        );
        let u = FuncTensor::<Flex, 2>::from_data(
            TensorData::from([[0.1f32, 0.2], [0.3, 0.4]]),
            &device,
        );

        let f_max = |x: &FuncTensor<FD, 2>| x.clone().max_dim(1).sum();
        let t_max: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f_max, &x, &u);
        assert!((values(t_max)[0] - 0.5).abs() < 1e-5);

        let f_min = |x: &FuncTensor<FD, 2>| x.clone().min_dim(1).sum();
        let t_min: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f_min, &x, &u);
        assert!((values(t_min)[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn jvp_slice_assign() {
        let device = FlexDevice;
        let x =
            FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 2.0, 3.0, 4.0]), &device);
        let u =
            FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 1.0, 1.0, 1.0]), &device);
        let v = FuncTensor::<Flex, 1>::from_data(TensorData::from([10.0f32, 20.0]), &device);
        let dv = FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32, 3.0]), &device);

        let f = |(x, v): &(FuncTensor<FD, 1>, FuncTensor<FD, 1>)| {
            x.clone()
                .slice_assign(&[burn_std::Slice::new(1, Some(3), 1)], v.clone())
                .sum()
        };
        let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &(x, v), &(u, dv));
        assert!((values(tangent)[0] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn jvp_mask_where_and_fill() {
        use burn_backend::ops::{BoolTensorOps, FloatTensorOps};
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 2.0, 3.0]), &device);
        let y = FuncTensor::<Flex, 1>::from_data(TensorData::from([10.0f32, 20.0, 30.0]), &device);
        let dx = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32, 2.0, 3.0]), &device);
        let dy = FuncTensor::<Flex, 1>::from_data(TensorData::from([10.0f32, 20.0, 30.0]), &device);
        let mask = burn_flex::Flex::bool_from_data(TensorData::from([true, false, true]), &device);

        let mask_clone = mask.clone();
        let f_where = move |(x, y): &(FuncTensor<FD, 1>, FuncTensor<FD, 1>)| {
            let prim = FD::float_mask_where(
                x.primitive().clone(),
                mask_clone.clone(),
                y.primitive().clone(),
            );
            FuncTensor::<FD, 1>::from_primitive(prim).sum()
        };
        let tangent_where: FuncTensor<Flex, 1> =
            jvp::<Flex, _, _, _>(f_where, &(x.clone(), y), &(dx.clone(), dy));
        assert!((values(tangent_where)[0] - 42.0).abs() < 1e-5);

        let f_fill = move |x: &FuncTensor<FD, 1>| {
            let prim = FD::float_mask_fill(x.primitive().clone(), mask.clone(), 0.0f64.into());
            FuncTensor::<FD, 1>::from_primitive(prim).sum()
        };
        let tangent_fill: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f_fill, &x, &dx);
        assert!((values(tangent_fill)[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn jvp_sort_and_topk() {
        use burn_backend::ops::FloatTensorOps;
        let device = FlexDevice;
        let x =
            FuncTensor::<Flex, 1>::from_data(TensorData::from([3.0f32, 1.0, 4.0, 2.0]), &device);
        let dx =
            FuncTensor::<Flex, 1>::from_data(TensorData::from([0.1f32, 0.2, 0.3, 0.4]), &device);

        let f_sort = |x: &FuncTensor<FD, 1>| {
            let prim = FD::float_sort(x.primitive().clone(), 0, false);
            FuncTensor::<FD, 1>::from_primitive(prim)
        };
        let t_sort: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f_sort, &x, &dx);
        assert_eq!(values(t_sort), vec![0.2, 0.4, 0.1, 0.3]);

        let f_topk = |x: &FuncTensor<FD, 1>| {
            let prim = FD::float_topk(x.primitive().clone(), 0, 2);
            FuncTensor::<FD, 1>::from_primitive(prim)
        };
        let t_topk: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f_topk, &x, &dx);
        assert_eq!(values(t_topk), vec![0.3, 0.1]);
    }

    #[test]
    fn jvp_pad() {
        use burn_backend::ops::FloatTensorOps;
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([2.0f32, 3.0]), &device);
        let dx = FuncTensor::<Flex, 1>::from_data(TensorData::from([0.5f32, 1.5]), &device);

        let f_pad = |x: &FuncTensor<FD, 1>| {
            let prim = FD::float_pad(x.primitive().clone(), &[(1, 1)], 0.0f64.into());
            FuncTensor::<FD, 1>::from_primitive(prim).sum()
        };
        let t_pad: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f_pad, &x, &dx);
        assert!((values(t_pad)[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn jvp_powf_zero_base() {
        let device = FlexDevice;
        let x = FuncTensor::<Flex, 1>::from_data(TensorData::from([0.0f32]), &device);
        let u = FuncTensor::<Flex, 1>::from_data(TensorData::from([1.0f32]), &device);
        let exponent = FuncTensor::<FD, 1>::from_data(TensorData::from([3.0f32]), &device);

        let f = move |x: &FuncTensor<FD, 1>| x.clone().powf(exponent.clone()).sum();
        let tangent: FuncTensor<Flex, 1> = jvp::<Flex, _, _, _>(f, &x, &u);

        let value = values(tangent)[0];
        assert!(value.is_finite());
        assert_eq!(value, 0.0);
    }
}
