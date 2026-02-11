// Language
use crate::rand::get_seeded_rng;
use alloc::vec::Vec;
use burn_backend::backend::ExecutionError;
use burn_backend::ops::IntTensorOps;
use burn_backend::tensor::{FloatTensor, IntTensor};
use burn_backend::{Distribution, IntDType, Scalar, TensorMetadata};

use burn_backend::ElementConversion;

// Current crate
use crate::{NdArray, cast_to_dtype, execute_with_dtype, tensor::NdArrayTensor};
use crate::{NdArrayDevice, SEED, slice};
use crate::{SharedArray, element::QuantElement};
use crate::{cat_with_dtype, execute_with_float_dtype};
use crate::{element::FloatNdArrayElement, ops::matmul::matmul};
use crate::{element::IntNdArrayElement, execute_with_int_dtype};

// Workspace crates
use super::{NdArrayBitOps, NdArrayMathOps, NdArrayOps};
use burn_backend::{DType, Shape, TensorData, backend::Backend};

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> IntTensorOps<Self>
    for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
    fn int_from_data(data: TensorData, _device: &NdArrayDevice) -> NdArrayTensor {
        if data.dtype.is_int() || data.dtype.is_uint() {
            NdArrayTensor::from_data(data)
        } else {
            unimplemented!("Unsupported dtype for `int_from_data`: {:?}", data.dtype)
        }
    }

    async fn int_into_data(tensor: NdArrayTensor) -> Result<TensorData, ExecutionError> {
        Ok(tensor.into_data())
    }

    fn int_to_device(tensor: NdArrayTensor, _device: &NdArrayDevice) -> NdArrayTensor {
        tensor
    }

    fn int_reshape(tensor: NdArrayTensor, shape: Shape) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |array| NdArrayOps::reshape(array, shape))
    }

    fn int_slice(tensor: NdArrayTensor, slices: &[burn_backend::Slice]) -> NdArrayTensor {
        slice!(tensor, slices)
    }

    fn int_device(_tensor: &NdArrayTensor) -> <NdArray<E> as Backend>::Device {
        NdArrayDevice::Cpu
    }

    fn int_empty(
        shape: Shape,
        device: &<NdArray<E> as Backend>::Device,
        dtype: IntDType,
    ) -> NdArrayTensor {
        Self::int_zeros(shape, device, dtype)
    }

    fn int_matmul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        execute_with_int_dtype!((lhs, rhs), matmul)
    }

    fn int_mask_where(
        tensor: NdArrayTensor,
        mask: NdArrayTensor,
        source: NdArrayTensor,
    ) -> NdArrayTensor {
        execute_with_int_dtype!((tensor, source), |tensor, source| {
            NdArrayOps::mask_where(tensor, mask.bool(), source)
        })
    }

    fn int_mask_fill(tensor: NdArrayTensor, mask: NdArrayTensor, value: Scalar) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |array| NdArrayOps::mask_fill(
            array,
            mask.bool(),
            value.elem()
        ))
    }

    fn int_slice_assign(
        tensor: NdArrayTensor,
        slices: &[burn_backend::Slice],
        value: NdArrayTensor,
    ) -> NdArrayTensor {
        execute_with_int_dtype!((tensor, value), |tensor, value| NdArrayOps::slice_assign(
            tensor, slices, value
        ))
    }

    fn int_cat(tensors: Vec<NdArrayTensor>, dim: usize) -> NdArrayTensor {
        cat_with_dtype!(tensors, dim, [I64, I32, I16, I8, U64, U32, U16, U8])
    }

    fn int_equal(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::equal)
    }

    fn int_equal_elem(lhs: NdArrayTensor, rhs: Scalar) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |array| NdArrayMathOps::equal_elem(array, rhs.elem()))
    }

    fn int_greater(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::greater)
    }

    fn int_greater_elem(lhs: NdArrayTensor, rhs: Scalar) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |array| NdArrayMathOps::greater_elem(array, rhs.elem()))
    }

    fn int_greater_equal(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::greater_equal)
    }

    fn int_greater_equal_elem(lhs: NdArrayTensor, rhs: Scalar) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |array| NdArrayMathOps::greater_equal_elem(
            array,
            rhs.elem()
        ))
    }

    fn int_lower(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::lower)
    }

    fn int_lower_elem(lhs: NdArrayTensor, rhs: Scalar) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |array| NdArrayMathOps::lower_elem(array, rhs.elem()))
    }

    fn int_lower_equal(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::lower_equal)
    }

    fn int_lower_equal_elem(lhs: NdArrayTensor, rhs: Scalar) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |array| NdArrayMathOps::lower_equal_elem(
            array,
            rhs.elem()
        ))
    }

    fn int_add(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::add)
    }

    fn int_add_scalar(lhs: NdArrayTensor, rhs: Scalar) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |array| NdArrayMathOps::add_scalar(array, rhs.elem()))
    }

    fn int_sub(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::sub)
    }

    fn int_sub_scalar(lhs: NdArrayTensor, rhs: Scalar) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |array| NdArrayMathOps::sub_scalar(array, rhs.elem()))
    }

    fn int_mul(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::mul)
    }

    fn int_mul_scalar(lhs: NdArrayTensor, rhs: Scalar) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |array| NdArrayMathOps::mul_scalar(array, rhs.elem()))
    }

    fn int_div(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::div)
    }

    fn int_div_scalar(lhs: NdArrayTensor, rhs: Scalar) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |array| NdArrayMathOps::div_scalar(array, rhs.elem()))
    }

    fn int_remainder(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::remainder)
    }

    fn int_remainder_scalar(lhs: NdArrayTensor, rhs: Scalar) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |array| NdArrayMathOps::remainder_scalar(
            array,
            rhs.elem()
        ))
    }

    fn int_sum(tensor: NdArrayTensor) -> NdArrayTensor {
        // Use view() for zero-copy on borrowed storage
        execute_with_int_dtype!(tensor, E, |array: SharedArray<E>| NdArrayMathOps::sum_view(
            array.view()
        ))
    }

    fn int_sum_dim(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |array| NdArrayMathOps::sum_dim(array, dim))
    }

    fn int_prod(tensor: NdArrayTensor) -> NdArrayTensor {
        // Use view() for zero-copy on borrowed storage
        execute_with_int_dtype!(
            tensor,
            E,
            |array: SharedArray<E>| NdArrayMathOps::prod_view(array.view())
        )
    }

    fn int_prod_dim(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |array| NdArrayMathOps::prod_dim(array, dim))
    }

    fn int_mean(tensor: NdArrayTensor) -> NdArrayTensor {
        // Use view() for zero-copy on borrowed storage
        execute_with_int_dtype!(
            tensor,
            E,
            |array: SharedArray<E>| NdArrayMathOps::mean_view(array.view())
        )
    }

    fn int_mean_dim(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |array| NdArrayMathOps::mean_dim(array, dim))
    }

    fn int_max(tensor: NdArrayTensor) -> NdArrayTensor {
        // Use view() for zero-copy on borrowed storage
        execute_with_int_dtype!(tensor, E, |array: SharedArray<E>| NdArrayMathOps::max_view(
            array.view()
        ))
    }

    fn int_min(tensor: NdArrayTensor) -> NdArrayTensor {
        // Use view() for zero-copy on borrowed storage
        execute_with_int_dtype!(tensor, E, |array: SharedArray<E>| NdArrayMathOps::min_view(
            array.view()
        ))
    }

    fn int_cumsum(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |array| NdArrayMathOps::cumsum(array, dim))
    }

    fn int_cumprod(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |array| NdArrayMathOps::cumprod(array, dim))
    }

    fn int_cummin(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |array| NdArrayMathOps::cummin(array, dim))
    }

    fn int_cummax(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |array| NdArrayMathOps::cummax(array, dim))
    }

    fn int_gather(dim: usize, tensor: NdArrayTensor, indices: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, E, |array| -> NdArrayTensor {
            execute_with_int_dtype!(indices, |idx_array| NdArrayOps::gather(
                dim, array, idx_array
            ))
        })
    }

    fn int_scatter_add(
        dim: usize,
        tensor: NdArrayTensor,
        indices: NdArrayTensor,
        value: NdArrayTensor,
    ) -> NdArrayTensor {
        execute_with_int_dtype!((tensor, value), I, |tensor, value| -> NdArrayTensor {
            execute_with_int_dtype!(indices, |idx_array| NdArrayOps::<I>::scatter(
                dim, tensor, idx_array, value
            ))
        })
    }

    fn int_select(tensor: NdArrayTensor, dim: usize, indices: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, E, |array| -> NdArrayTensor {
            execute_with_int_dtype!(indices, |idx_array| NdArrayMathOps::select(
                array, dim, idx_array
            ))
        })
    }

    fn int_select_add(
        tensor: NdArrayTensor,
        dim: usize,
        indices: NdArrayTensor,
        value: NdArrayTensor,
    ) -> NdArrayTensor {
        execute_with_int_dtype!((tensor, value), I, |tensor, value| -> NdArrayTensor {
            execute_with_int_dtype!(indices, |idx_array| NdArrayMathOps::<I>::select_assign(
                tensor, dim, idx_array, value
            ))
        })
    }
    fn int_argmax(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        // Use view() for zero-copy on borrowed storage
        execute_with_int_dtype!(tensor, E, |array: SharedArray<E>| {
            NdArrayMathOps::argmax_view::<I>(array.view(), dim)
        })
    }

    fn int_argmin(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        // Use view() for zero-copy on borrowed storage
        execute_with_int_dtype!(tensor, E, |array: SharedArray<E>| {
            NdArrayMathOps::argmin_view::<I>(array.view(), dim)
        })
    }

    fn int_clamp_min(tensor: NdArrayTensor, min: Scalar) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |array| NdArrayMathOps::clamp_min(array, min.elem()))
    }

    fn int_clamp_max(tensor: NdArrayTensor, max: Scalar) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |array| NdArrayMathOps::clamp_max(array, max.elem()))
    }

    fn int_clamp(tensor: NdArrayTensor, min: Scalar, max: Scalar) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |array| NdArrayMathOps::clamp(
            array,
            min.elem(),
            max.elem()
        ))
    }

    fn int_abs(tensor: NdArrayTensor) -> NdArrayTensor {
        match tensor.dtype() {
            DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                execute_with_dtype!(tensor, I, NdArrayMathOps::abs, [
                    I64 => i64, I32 => i32, I16 => i16, I8 => i8
                ])
            }
            // Already unsigned
            DType::U64 | DType::U32 | DType::U16 | DType::U8 => tensor,
            other => panic!("Unsupported dtype: {other:?}"),
        }
    }

    fn int_into_float(tensor: NdArrayTensor) -> FloatTensor<Self> {
        execute_with_int_dtype!(tensor, IntElem, |array: SharedArray<IntElem>| array
            .mapv(|a: IntElem| a.elem::<E>())
            .into_shared())
    }

    fn int_swap_dims(tensor: NdArrayTensor, dim1: usize, dim2: usize) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |array| NdArrayOps::swap_dims(array, dim1, dim2))
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &NdArrayDevice,
    ) -> NdArrayTensor {
        let mut seed = SEED.lock().unwrap();
        let mut rng = seed.take().unwrap_or_else(get_seeded_rng);

        let effective_distribution = if distribution == Distribution::Default {
            Distribution::Uniform(0.0, 255.0) // Assuming UniformInt is the integer variant
        } else {
            distribution
        };

        let tensor = Self::int_from_data(
            TensorData::random::<I, _, _>(shape, effective_distribution, &mut rng),
            device,
        );
        *seed = Some(rng);
        tensor
    }

    fn int_powi(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), I, |lhs, rhs| NdArrayMathOps::elementwise_op(
            lhs,
            rhs,
            |a: &I, b: &I| { (a.elem::<i64>().pow(b.elem::<u32>())).elem() }
        ))
    }

    fn int_powf(lhs: NdArrayTensor, rhs: FloatTensor<Self>) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, I, |array| -> NdArrayTensor {
            execute_with_float_dtype!(rhs, E, |rhs_array| {
                NdArrayMathOps::elementwise_op(array, rhs_array, |a: &I, b: &E| {
                    (a.elem::<i64>().pow(*b as u32)).elem()
                })
            })
        })
    }

    fn int_powf_scalar_impl(lhs: NdArrayTensor, rhs: Scalar) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, I, |array| {
            NdArrayMathOps::elementwise_op_scalar(array, |a: I| {
                (a.elem::<i64>().pow(rhs.elem())).elem()
            })
        })
    }

    fn int_permute(tensor: NdArrayTensor, axes: &[usize]) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |array| NdArrayOps::permute(array, axes))
    }

    fn int_flip(tensor: NdArrayTensor, axes: &[usize]) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |array| NdArrayOps::flip(array, axes))
    }

    fn int_sign(tensor: NdArrayTensor) -> NdArrayTensor {
        match tensor.dtype() {
            DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                execute_with_dtype!(tensor, I, NdArrayMathOps::sign_op, [
                    I64 => i64, I32 => i32, I16 => i16, I8 => i8
                ])
            }
            DType::U64 | DType::U32 | DType::U16 | DType::U8 => {
                Self::int_greater_elem(tensor, 0.into())
            }
            other => panic!("Unsupported dtype: {other:?}"),
        }
    }

    fn int_expand(tensor: NdArrayTensor, shape: Shape) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |array| NdArrayOps::expand(array, shape))
    }

    fn bitwise_and(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayBitOps::bitand)
    }

    fn bitwise_and_scalar(lhs: NdArrayTensor, rhs: Scalar) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |array| NdArrayBitOps::bitand_scalar(array, rhs.elem()))
    }

    fn bitwise_or(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayBitOps::bitor)
    }

    fn bitwise_or_scalar(lhs: NdArrayTensor, rhs: Scalar) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |array| NdArrayBitOps::bitor_scalar(array, rhs.elem()))
    }

    fn bitwise_xor(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayBitOps::bitxor)
    }

    fn bitwise_xor_scalar(lhs: NdArrayTensor, rhs: Scalar) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |array| NdArrayBitOps::bitxor_scalar(array, rhs.elem()))
    }

    fn bitwise_not(tensor: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, NdArrayBitOps::bitnot)
    }

    fn bitwise_left_shift(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), I, |lhs, rhs| {
            NdArrayMathOps::elementwise_op(lhs, rhs, |a: &I, b: &I| {
                (a.elem::<i64>() << (b.elem::<u32>())).elem()
            })
        })
    }

    fn bitwise_left_shift_scalar(lhs: NdArrayTensor, rhs: Scalar) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, I, |array| {
            NdArrayMathOps::elementwise_op_scalar(array, |a: I| {
                (a.elem::<i64>() << rhs.elem::<u32>()).elem()
            })
        })
    }

    fn bitwise_right_shift(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), I, |lhs, rhs| {
            NdArrayMathOps::elementwise_op(lhs, rhs, |a: &I, b: &I| {
                (a.elem::<i64>() >> (b.elem::<u32>())).elem()
            })
        })
    }

    fn bitwise_right_shift_scalar(lhs: NdArrayTensor, rhs: Scalar) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, I, |array| {
            NdArrayMathOps::elementwise_op_scalar(array, |a: I| {
                (a.elem::<i64>() >> rhs.elem::<u32>()).elem()
            })
        })
    }

    fn int_cast(tensor: IntTensor<Self>, dtype: IntDType) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, |array| cast_to_dtype(array, dtype.into()))
    }

    fn int_unfold(
        tensor: IntTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, |array| NdArrayOps::unfold(array, dim, size, step))
    }
}
