// Language
use crate::rand::get_seeded_rng;
use alloc::vec::Vec;
use burn_backend::backend::ExecutionError;
use burn_backend::ops::{ConvOptions, IntTensorOps};
use burn_backend::tensor::{FloatTensor, IntTensor};
use burn_backend::{Distribution, IntDType, Scalar, TensorMetadata};
use ndarray::{ArrayViewD, Zip};

use burn_backend::ElementConversion;

// Current crate
use crate::{NdArray, cast_to_dtype, execute_with_dtype, tensor::NdArrayTensor};
use crate::{NdArrayDevice, SEED, slice};
use crate::{SharedArray, element::QuantElement};
use crate::{cat_with_dtype, execute_with_float_dtype};
use crate::{element::FloatNdArrayElement, ops::conv::conv2d, ops::matmul::matmul};
use crate::{element::IntNdArrayElement, execute_with_int_dtype};

// Workspace crates
use super::{NdArrayBitOps, NdArrayMathOps, NdArrayOps};
use burn_backend::{DType, Shape, TensorData, backend::Backend};

fn round_ties_to_even(value: f32) -> i32 {
    if !value.is_finite() {
        return if value.is_sign_negative() {
            i32::MIN
        } else {
            i32::MAX
        };
    }

    let floor = value.floor();
    let frac = value - floor;
    let rounded = if frac < 0.5 {
        floor
    } else if frac > 0.5 {
        floor + 1.0
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    };

    if rounded <= i32::MIN as f32 {
        i32::MIN
    } else if rounded >= i32::MAX as f32 {
        i32::MAX
    } else {
        rounded as i32
    }
}

fn saturate_to_dtype(value: i32, dtype: IntDType) -> i32 {
    match dtype {
        IntDType::I8 => value.clamp(i8::MIN as i32, i8::MAX as i32),
        IntDType::U8 => value.clamp(u8::MIN as i32, u8::MAX as i32),
        IntDType::I16 => value.clamp(i16::MIN as i32, i16::MAX as i32),
        IntDType::U16 => value.clamp(u16::MIN as i32, u16::MAX as i32),
        IntDType::U32 | IntDType::U64 => value.max(0),
        IntDType::I32 | IntDType::I64 => value,
    }
}

fn into_i32_array(tensor: NdArrayTensor) -> SharedArray<i32> {
    match tensor {
        NdArrayTensor::I32(storage) => storage.into_shared(),
        _ => unreachable!(),
    }
}

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

    fn int_matmul_accum(
        lhs: IntTensor<Self>,
        lhs_zero_point: IntTensor<Self>,
        rhs: IntTensor<Self>,
        rhs_zero_point: IntTensor<Self>,
    ) -> IntTensor<Self> {
        let lhs = Self::int_sub(
            Self::int_cast(lhs, IntDType::I32),
            Self::int_cast(lhs_zero_point, IntDType::I32),
        );
        let rhs = Self::int_sub(
            Self::int_cast(rhs, IntDType::I32),
            Self::int_cast(rhs_zero_point, IntDType::I32),
        );

        Self::int_cast(Self::int_matmul(lhs, rhs), IntDType::I32)
    }

    fn int_conv2d_accum(
        x: IntTensor<Self>,
        x_zero_point: IntTensor<Self>,
        weight: IntTensor<Self>,
        weight_zero_point: IntTensor<Self>,
        bias: Option<IntTensor<Self>>,
        options: ConvOptions<2>,
    ) -> IntTensor<Self> {
        let x = Self::int_sub(
            Self::int_cast(x, IntDType::I32),
            Self::int_cast(x_zero_point, IntDType::I32),
        );
        let weight = Self::int_sub(
            Self::int_cast(weight, IntDType::I32),
            Self::int_cast(weight_zero_point, IntDType::I32),
        );
        let bias = bias.map(|bias| into_i32_array(Self::int_cast(bias, IntDType::I32)));

        conv2d::<i32>(into_i32_array(x), into_i32_array(weight), bias, options).into()
    }

    fn int_requantize(
        tensor: IntTensor<Self>,
        scale: FloatTensor<Self>,
        zero_point: IntTensor<Self>,
        dtype: IntDType,
    ) -> IntTensor<Self> {
        let tensor = into_i32_array(Self::int_cast(tensor, IntDType::I32));
        let zero_point = into_i32_array(Self::int_cast(zero_point, IntDType::I32));
        let scale = execute_with_float_dtype!(scale, F, |array: SharedArray<F>| {
            array.mapv(|value| value.elem::<f32>()).into_shared()
        });
        let scale = match scale {
            NdArrayTensor::F32(storage) => storage.into_shared(),
            _ => unreachable!(),
        };

        let shape = tensor.raw_dim();
        let scale = scale
            .broadcast(shape.clone())
            .unwrap_or_else(|| panic!("Scale shape {:?} not broadcastable to {:?}", scale.shape(), tensor.shape()));
        let zero_point = zero_point
            .broadcast(shape.clone())
            .unwrap_or_else(|| {
                panic!(
                    "Zero-point shape {:?} not broadcastable to {:?}",
                    zero_point.shape(),
                    tensor.shape()
                )
            });

        let mut output = ndarray::ArrayD::<i32>::zeros(shape);
        Zip::from(output.view_mut())
            .and(ArrayViewD::from(&tensor))
            .and(scale)
            .and(zero_point)
            .for_each(|out, &value, &scale, &zp| {
                let scaled = value as f32 * scale;
                let rounded = round_ties_to_even(scaled);
                *out = saturate_to_dtype(rounded.saturating_add(zp), dtype);
            });

        let output: NdArrayTensor = output.into_shared().into();
        Self::int_cast(output, dtype)
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::ops::ConvOptions;

    type TestBackend = NdArray<f32, i32, i8>;

    #[test]
    fn int_matmul_accum_u8_i32() {
        let device = NdArrayDevice::Cpu;
        let lhs = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
            TensorData::from([[10u8, 12u8], [8u8, 9u8]]),
            &device,
        );
        let rhs = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
            TensorData::from([[2u8, 4u8], [6u8, 8u8]]),
            &device,
        );
        let lhs_zp = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
            TensorData::from([10u8]),
            &device,
        );
        let rhs_zp = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
            TensorData::from([5u8]),
            &device,
        );

        let output = <TestBackend as IntTensorOps<TestBackend>>::int_matmul_accum(
            lhs, lhs_zp, rhs, rhs_zp,
        );

        output
            .into_data()
            .assert_eq(&TensorData::from([[2i32, 6i32], [5i32, -1i32]]), false);
    }

    #[test]
    fn int_conv2d_accum_u8_i8_i32() {
        let device = NdArrayDevice::Cpu;
        let x = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
            TensorData::from([[[[11u8, 12u8], [13u8, 14u8]]]]),
            &device,
        );
        let weight = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
            TensorData::from([[[[2i8]]]]),
            &device,
        );
        let x_zp = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
            TensorData::from([10u8]),
            &device,
        );
        let w_zp = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
            TensorData::from([1i8]),
            &device,
        );

        let output = <TestBackend as IntTensorOps<TestBackend>>::int_conv2d_accum(
            x,
            x_zp,
            weight,
            w_zp,
            None,
            ConvOptions::new([1, 1], [0, 0], [1, 1], 1),
        );

        output
            .into_data()
            .assert_eq(&TensorData::from([[[[1i32, 2i32], [3i32, 4i32]]]]), false);
    }

    #[test]
    fn int_requantize_supports_u8_and_i8() {
        let device = NdArrayDevice::Cpu;

        let accum = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
            TensorData::from([[-3i32, -1i32, 0i32, 1i32, 3i32]]),
            &device,
        );
        let scale = NdArrayTensor::from_data(TensorData::from([[1.5f32; 5]]));

        let zp_u8 = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
            TensorData::from([128i32]),
            &device,
        );

        let output_u8 = <TestBackend as IntTensorOps<TestBackend>>::int_requantize(
            accum.clone(),
            scale.clone(),
            zp_u8,
            IntDType::U8,
        );
        output_u8
            .into_data()
            .assert_eq(&TensorData::from([[124u8, 126u8, 128u8, 130u8, 132u8]]), false);

        let accum_i8 = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
            TensorData::from([[-200i32, -128i32, 0i32, 127i32, 200i32]]),
            &device,
        );
        let scale_1 = NdArrayTensor::from_data(TensorData::from([[1.0f32; 5]]));
        let zp_i8 = <TestBackend as IntTensorOps<TestBackend>>::int_from_data(
            TensorData::from([0i32]),
            &device,
        );

        let output_i8 = <TestBackend as IntTensorOps<TestBackend>>::int_requantize(
            accum_i8,
            scale_1,
            zp_i8,
            IntDType::I8,
        );
        output_i8
            .into_data()
            .assert_eq(&TensorData::from([[-128i8, -128i8, 0i8, 127i8, 127i8]]), false);
    }
}
