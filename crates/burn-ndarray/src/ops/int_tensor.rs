// Language
use crate::rand::get_seeded_rng;
use alloc::vec::Vec;
use burn_tensor::{Distribution, ops::IntTensor};
use burn_tensor::{IntDType, ops::IntTensorOps};
use burn_tensor::{TensorMetadata, ops::FloatTensor};

use burn_tensor::ElementConversion;

// Current crate
use crate::{NdArray, cast_to_dtype, execute_with_dtype, tensor::NdArrayTensor};
use crate::{NdArrayDevice, SEED};
use crate::{SharedArray, element::QuantElement};
use crate::{cat_with_dtype, execute_with_float_dtype};
use crate::{element::FloatNdArrayElement, ops::matmul::matmul};
use crate::{element::IntNdArrayElement, execute_with_int_dtype};

// Workspace crates
use super::{NdArrayBitOps, NdArrayMathOps, NdArrayOps};
use burn_tensor::{DType, Shape, TensorData, backend::Backend};

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

    async fn int_into_data(tensor: NdArrayTensor) -> TensorData {
        tensor.into_data()
    }

    fn int_to_device(tensor: NdArrayTensor, _device: &NdArrayDevice) -> NdArrayTensor {
        tensor
    }

    fn int_reshape(tensor: NdArrayTensor, shape: Shape) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |tensor| NdArrayOps::reshape(tensor, shape))
    }

    fn int_slice(tensor: NdArrayTensor, slices: &[burn_tensor::Slice]) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |tensor| NdArrayOps::slice(tensor, slices))
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
            NdArrayMathOps::mask_where(tensor, mask.bool(), source)
        })
    }

    fn int_mask_fill(tensor: NdArrayTensor, mask: NdArrayTensor, value: I) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |tensor| NdArrayMathOps::mask_fill(
            tensor,
            mask.bool(),
            value.elem()
        ))
    }

    fn int_slice_assign(
        tensor: NdArrayTensor,
        slices: &[burn_tensor::Slice],
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

    fn int_equal_elem(lhs: NdArrayTensor, rhs: I) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |lhs| NdArrayMathOps::equal_elem(lhs, rhs.elem()))
    }

    fn int_greater(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::greater)
    }

    fn int_greater_elem(lhs: NdArrayTensor, rhs: I) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |lhs| NdArrayMathOps::greater_elem(lhs, rhs.elem()))
    }

    fn int_greater_equal(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::greater_equal)
    }

    fn int_greater_equal_elem(lhs: NdArrayTensor, rhs: I) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |lhs| NdArrayMathOps::greater_equal_elem(
            lhs,
            rhs.elem()
        ))
    }

    fn int_lower(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::lower)
    }

    fn int_lower_elem(lhs: NdArrayTensor, rhs: I) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |lhs| NdArrayMathOps::lower_elem(lhs, rhs.elem()))
    }

    fn int_lower_equal(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::lower_equal)
    }

    fn int_lower_equal_elem(lhs: NdArrayTensor, rhs: I) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |lhs| NdArrayMathOps::lower_equal_elem(lhs, rhs.elem()))
    }

    fn int_add(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::add)
    }

    fn int_add_scalar(lhs: NdArrayTensor, rhs: I) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |lhs| NdArrayMathOps::add_scalar(lhs, rhs.elem()))
    }

    fn int_sub(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::sub)
    }

    fn int_sub_scalar(lhs: NdArrayTensor, rhs: I) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |lhs| NdArrayMathOps::sub_scalar(lhs, rhs.elem()))
    }

    fn int_mul(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::mul)
    }

    fn int_mul_scalar(lhs: NdArrayTensor, rhs: I) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |lhs| NdArrayMathOps::mul_scalar(lhs, rhs.elem()))
    }

    fn int_div(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::div)
    }

    fn int_div_scalar(lhs: NdArrayTensor, rhs: I) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |lhs| NdArrayMathOps::div_scalar(lhs, rhs.elem()))
    }

    fn int_remainder(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::remainder)
    }

    fn int_remainder_scalar(lhs: NdArrayTensor, rhs: I) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |lhs| NdArrayMathOps::remainder_scalar(lhs, rhs.elem()))
    }

    fn int_neg(tensor: NdArrayTensor) -> NdArrayTensor {
        Self::int_mul_scalar(tensor, (-1).elem())
    }

    fn int_sum(tensor: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, NdArrayMathOps::sum)
    }

    fn int_sum_dim(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |tensor| NdArrayMathOps::sum_dim(tensor, dim))
    }

    fn int_prod(tensor: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, NdArrayMathOps::prod)
    }

    fn int_prod_dim(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |tensor| NdArrayMathOps::prod_dim(tensor, dim))
    }

    fn int_mean(tensor: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, NdArrayMathOps::mean)
    }

    fn int_mean_dim(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |tensor| NdArrayMathOps::mean_dim(tensor, dim))
    }

    fn int_cumsum(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |tensor| NdArrayMathOps::cumsum(tensor, dim))
    }

    fn int_cummax(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |tensor| NdArrayMathOps::cummax(tensor, dim))
    }

    fn int_gather(dim: usize, tensor: NdArrayTensor, indices: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, E, |tensor: SharedArray<E>| -> NdArrayTensor {
            execute_with_int_dtype!(indices, |indices| NdArrayMathOps::gather(
                dim, tensor, indices
            ))
        })
    }

    fn int_scatter(
        dim: usize,
        tensor: NdArrayTensor,
        indices: NdArrayTensor,
        value: NdArrayTensor,
    ) -> NdArrayTensor {
        execute_with_int_dtype!((tensor, value), I, |tensor, value| -> NdArrayTensor {
            execute_with_int_dtype!(indices, |indices| NdArrayMathOps::<I>::scatter(
                dim, tensor, indices, value
            ))
        })
    }

    fn int_select(tensor: NdArrayTensor, dim: usize, indices: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, E, |tensor: SharedArray<E>| -> NdArrayTensor {
            execute_with_int_dtype!(indices, |indices| NdArrayMathOps::select(
                tensor, dim, indices
            ))
        })
    }

    fn int_select_assign(
        tensor: NdArrayTensor,
        dim: usize,
        indices: NdArrayTensor,
        value: NdArrayTensor,
    ) -> NdArrayTensor {
        execute_with_int_dtype!((tensor, value), I, |tensor, value| -> NdArrayTensor {
            execute_with_int_dtype!(indices, |indices| NdArrayMathOps::<I>::select_assign(
                tensor, dim, indices, value
            ))
        })
    }
    fn int_argmax(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |tensor| NdArrayMathOps::argmax::<I>(tensor, dim))
    }

    fn int_argmin(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |tensor| NdArrayMathOps::argmin::<I>(tensor, dim))
    }

    fn int_clamp_min(tensor: NdArrayTensor, min: I) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |tensor| NdArrayMathOps::clamp_min(
            tensor,
            min.elem()
        ))
    }

    fn int_clamp_max(tensor: NdArrayTensor, max: I) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |tensor| NdArrayMathOps::clamp_max(
            tensor,
            max.elem()
        ))
    }

    fn int_clamp(tensor: NdArrayTensor, min: I, max: I) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |tensor| NdArrayMathOps::clamp(
            tensor,
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
        execute_with_int_dtype!(tensor, I, |t: SharedArray<I>| t
            .mapv(|a| a.elem::<E>())
            .into_shared())
    }

    fn int_swap_dims(tensor: NdArrayTensor, dim1: usize, dim2: usize) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |tensor| NdArrayOps::swap_dims(tensor, dim1, dim2))
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &NdArrayDevice,
    ) -> NdArrayTensor {
        let mut seed = SEED.lock().unwrap();
        let mut rng = if let Some(rng_seeded) = seed.as_ref() {
            rng_seeded.clone()
        } else {
            get_seeded_rng()
        };

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
        execute_with_int_dtype!(lhs, I, |lhs| -> NdArrayTensor {
            execute_with_float_dtype!(rhs, E, |rhs| {
                NdArrayMathOps::elementwise_op(lhs, rhs, |a: &I, b: &E| {
                    (a.elem::<i64>().pow(*b as u32)).elem()
                })
            })
        })
    }

    fn int_powf_scalar(lhs: NdArrayTensor, rhs: f32) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, I, |lhs| {
            NdArrayMathOps::elementwise_op_scalar(lhs, |a: I| {
                (a.elem::<i64>().pow(rhs as u32)).elem()
            })
        })
    }

    fn int_permute(tensor: NdArrayTensor, axes: &[usize]) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |tensor| NdArrayOps::permute(tensor, axes))
    }

    fn int_flip(tensor: NdArrayTensor, axes: &[usize]) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |tensor| NdArrayOps::flip(tensor, axes))
    }

    fn int_sign(tensor: NdArrayTensor) -> NdArrayTensor {
        match tensor.dtype() {
            DType::I64 | DType::I32 | DType::I16 | DType::I8 => {
                execute_with_dtype!(tensor, I, NdArrayMathOps::sign_op, [
                    I64 => i64, I32 => i32, I16 => i16, I8 => i8
                ])
            }
            DType::U64 | DType::U32 | DType::U16 | DType::U8 => {
                Self::int_greater_elem(tensor, 0.elem())
            }
            other => panic!("Unsupported dtype: {other:?}"),
        }
    }

    fn int_expand(tensor: NdArrayTensor, shape: Shape) -> NdArrayTensor {
        execute_with_int_dtype!(tensor, |tensor| NdArrayOps::expand(tensor, shape))
    }

    fn bitwise_and(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayBitOps::bitand)
    }

    fn bitwise_and_scalar(lhs: NdArrayTensor, rhs: I) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |lhs| NdArrayBitOps::bitand_scalar(lhs, rhs.elem()))
    }

    fn bitwise_or(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayBitOps::bitor)
    }

    fn bitwise_or_scalar(lhs: NdArrayTensor, rhs: I) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |lhs| NdArrayBitOps::bitor_scalar(lhs, rhs.elem()))
    }

    fn bitwise_xor(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
        execute_with_int_dtype!((lhs, rhs), NdArrayBitOps::bitxor)
    }

    fn bitwise_xor_scalar(lhs: NdArrayTensor, rhs: I) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, |lhs| NdArrayBitOps::bitxor_scalar(lhs, rhs.elem()))
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

    fn bitwise_left_shift_scalar(lhs: NdArrayTensor, rhs: I) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, I, |lhs| {
            NdArrayMathOps::elementwise_op_scalar(lhs, |a: I| {
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

    fn bitwise_right_shift_scalar(lhs: NdArrayTensor, rhs: I) -> NdArrayTensor {
        execute_with_int_dtype!(lhs, I, |lhs| {
            NdArrayMathOps::elementwise_op_scalar(lhs, |a: I| {
                (a.elem::<i64>() >> rhs.elem::<u32>()).elem()
            })
        })
    }

    fn int_cast(tensor: IntTensor<Self>, dtype: IntDType) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, |tensor| cast_to_dtype(tensor, dtype.into()))
    }

    fn int_unfold(
        tensor: IntTensor<Self>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, |tensor| NdArrayOps::unfold(tensor, dim, size, step))
    }
}
