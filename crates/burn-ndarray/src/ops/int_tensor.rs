// Language
use alloc::vec::Vec;
use burn_common::rand::get_seeded_rng;
use burn_tensor::Distribution;
use burn_tensor::ops::FloatTensor;
use burn_tensor::ops::IntTensor;
use burn_tensor::ops::IntTensorOps;

use burn_tensor::ElementConversion;
use core::ops::Range;

use crate::ExpElement;
use crate::execute_with_float_dtype;
// Current crate
use crate::NdArrayTensorInt;
use crate::element::FloatNdArrayElement;
use crate::element::IntNdArrayElement;
use crate::element::QuantElement;
use crate::execute_with_int_dtype;
use crate::{NdArray, tensor::NdArrayTensor};
use crate::{NdArrayDevice, SEED};

// Workspace crates
use burn_tensor::{DType, IntDType, Shape, TensorData, TensorMetadata, backend::Backend};

use super::{NdArrayBitOps, NdArrayMathOps, NdArrayOps};

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> IntTensorOps<Self>
    for NdArray<E, I, Q>
{
    fn int_from_data(data: TensorData, _device: &NdArrayDevice) -> IntTensor<Self> {
        match data.dtype {
            DType::I64 => NdArrayTensorInt::I64(NdArrayTensor::from_data(data)),
            DType::I32 => NdArrayTensorInt::I32(NdArrayTensor::from_data(data)),
            DType::U8 => NdArrayTensorInt::U8(NdArrayTensor::from_data(data)),
            _ => unimplemented!("Unsupported dtype for `int_from_data`"),
        }
    }

    async fn int_into_data(tensor: IntTensor<Self>) -> TensorData {
        match tensor {
            NdArrayTensorInt::I64(tensor) => NdArrayOps::into_data(tensor),
            NdArrayTensorInt::I32(tensor) => NdArrayOps::into_data(tensor),
            NdArrayTensorInt::U8(tensor) => NdArrayOps::into_data(tensor),
        }
    }

    fn int_to_device(tensor: IntTensor<Self>, _device: &NdArrayDevice) -> IntTensor<Self> {
        tensor
    }

    fn int_reshape(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, |tensor| NdArrayOps::reshape(tensor, shape))
    }

    fn int_slice(tensor: IntTensor<Self>, ranges: &[Range<usize>]) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, |tensor| NdArrayOps::slice(tensor, ranges))
    }

    fn int_device(_tensor: &IntTensor<Self>) -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    fn int_empty(shape: Shape, device: &<NdArray<E> as Backend>::Device) -> IntTensor<Self> {
        NdArray::<E>::int_zeros(shape, device)
    }

    fn int_mask_where(
        tensor: IntTensor<Self>,
        mask: NdArrayTensor<bool>,
        source: IntTensor<Self>,
    ) -> IntTensor<Self> {
        execute_with_int_dtype!((tensor, source), |tensor, source| {
            NdArrayMathOps::mask_where(tensor, mask, source)
        })
    }

    fn int_mask_fill(
        tensor: IntTensor<Self>,
        mask: NdArrayTensor<bool>,
        value: I,
    ) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, |tensor| {
            NdArrayMathOps::mask_fill(tensor, mask, value.elem())
        })
    }

    fn int_slice_assign(
        tensor: IntTensor<Self>,
        ranges: &[Range<usize>],
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        execute_with_int_dtype!((tensor, value), |tensor, value| {
            NdArrayOps::slice_assign(tensor, ranges, value)
        })
    }

    fn int_cat(tensors: Vec<IntTensor<Self>>, dim: usize) -> IntTensor<Self> {
        match &tensors[0] {
            NdArrayTensorInt::U8(_) => {
                let tensors = tensors
                    .iter()
                    .map(|t| {
                        if let NdArrayTensorInt::U8(tensor) = t {
                            tensor.array.view()
                        } else {
                            panic!(
                                "Concatenate data type mismatch (expected u8, got {:?})",
                                t.dtype()
                            )
                        }
                    })
                    .collect::<Vec<_>>();
                NdArrayTensorInt::U8(NdArrayOps::concatenate(&tensors, dim))
            }
            NdArrayTensorInt::I32(_) => {
                let tensors = tensors
                    .iter()
                    .map(|t| {
                        if let NdArrayTensorInt::I32(tensor) = t {
                            tensor.array.view()
                        } else {
                            panic!(
                                "Concatenate data type mismatch (expected i46, got {:?})",
                                t.dtype()
                            )
                        }
                    })
                    .collect::<Vec<_>>();
                NdArrayTensorInt::I32(NdArrayOps::concatenate(&tensors, dim))
            }
            NdArrayTensorInt::I64(_) => {
                let tensors = tensors
                    .iter()
                    .map(|t| {
                        if let NdArrayTensorInt::I64(tensor) = t {
                            tensor.array.view()
                        } else {
                            panic!(
                                "Concatenate data type mismatch (expected i46, got {:?})",
                                t.dtype()
                            )
                        }
                    })
                    .collect::<Vec<_>>();
                NdArrayTensorInt::I64(NdArrayOps::concatenate(&tensors, dim))
            }
        }
    }

    fn int_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> NdArrayTensor<bool> {
        execute_with_int_dtype!((lhs, rhs) => |lhs: NdArrayTensor<_>, rhs: NdArrayTensor<_>| {
            NdArrayMathOps::equal(lhs, rhs)
        })
    }

    fn int_equal_elem(lhs: IntTensor<Self>, rhs: I) -> NdArrayTensor<bool> {
        execute_with_int_dtype!(lhs, E => |tensor: NdArrayTensor<E>| {
            NdArrayMathOps::equal_elem(tensor, rhs.elem())
        })
    }

    fn int_greater(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> NdArrayTensor<bool> {
        execute_with_int_dtype!((lhs, rhs) => |lhs: NdArrayTensor<_>, rhs: NdArrayTensor<_>| {
            NdArrayMathOps::greater(lhs, rhs)
        })
    }

    fn int_greater_elem(lhs: IntTensor<Self>, rhs: I) -> NdArrayTensor<bool> {
        execute_with_int_dtype!(lhs, E => |tensor: NdArrayTensor<E>| {
            NdArrayMathOps::greater_elem(tensor, rhs.elem())
        })
    }

    fn int_greater_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> NdArrayTensor<bool> {
        execute_with_int_dtype!((lhs, rhs) => |lhs: NdArrayTensor<_>, rhs: NdArrayTensor<_>| {
            NdArrayMathOps::greater_equal(lhs, rhs)
        })
    }

    fn int_greater_equal_elem(lhs: IntTensor<Self>, rhs: I) -> NdArrayTensor<bool> {
        execute_with_int_dtype!(lhs, E => |tensor: NdArrayTensor<E>| {
            NdArrayMathOps::greater_equal_elem(tensor, rhs.elem())
        })
    }

    fn int_lower(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> NdArrayTensor<bool> {
        execute_with_int_dtype!((lhs, rhs) => |lhs: NdArrayTensor<_>, rhs: NdArrayTensor<_>| {
            NdArrayMathOps::lower(lhs, rhs)
        })
    }

    fn int_lower_elem(lhs: IntTensor<Self>, rhs: I) -> NdArrayTensor<bool> {
        execute_with_int_dtype!(lhs, E => |tensor: NdArrayTensor<E>| {
            NdArrayMathOps::lower_elem(tensor, rhs.elem())
        })
    }

    fn int_lower_equal(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> NdArrayTensor<bool> {
        execute_with_int_dtype!((lhs, rhs) => |lhs: NdArrayTensor<_>, rhs: NdArrayTensor<_>| {
            NdArrayMathOps::lower_equal(lhs, rhs)
        })
    }

    fn int_lower_equal_elem(lhs: IntTensor<Self>, rhs: I) -> NdArrayTensor<bool> {
        execute_with_int_dtype!(lhs, E => |tensor: NdArrayTensor<E>| {
            NdArrayMathOps::lower_equal_elem(tensor, rhs.elem())
        })
    }

    fn int_add(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::add)
    }

    fn int_add_scalar(lhs: IntTensor<Self>, rhs: I) -> IntTensor<Self> {
        execute_with_int_dtype!(lhs, |lhs| NdArrayMathOps::add_scalar(lhs, rhs.elem()))
    }

    fn int_sub(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::sub)
    }

    fn int_sub_scalar(lhs: IntTensor<Self>, rhs: I) -> IntTensor<Self> {
        execute_with_int_dtype!(lhs, |lhs| NdArrayMathOps::sub_scalar(lhs, rhs.elem()))
    }

    fn int_mul(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::mul)
    }

    fn int_mul_scalar(lhs: IntTensor<Self>, rhs: I) -> IntTensor<Self> {
        execute_with_int_dtype!(lhs, |lhs| NdArrayMathOps::mul_scalar(lhs, rhs.elem()))
    }

    fn int_div(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::div)
    }

    fn int_div_scalar(lhs: IntTensor<Self>, rhs: I) -> IntTensor<Self> {
        execute_with_int_dtype!(lhs, |lhs| NdArrayMathOps::div_scalar(lhs, rhs.elem()))
    }

    fn int_remainder(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        execute_with_int_dtype!((lhs, rhs), NdArrayMathOps::remainder)
    }

    fn int_remainder_scalar(lhs: IntTensor<Self>, rhs: I) -> IntTensor<Self> {
        execute_with_int_dtype!(lhs, |lhs| NdArrayMathOps::remainder_scalar(lhs, rhs.elem()))
    }

    fn int_neg(tensor: IntTensor<Self>) -> IntTensor<Self> {
        Self::int_mul_scalar(tensor, (-1).elem())
    }

    fn int_zeros(shape: Shape, device: &<NdArray<E> as Backend>::Device) -> IntTensor<Self> {
        Self::int_from_data(TensorData::zeros::<I, _>(shape), device)
    }

    fn int_ones(shape: Shape, device: &<NdArray<E> as Backend>::Device) -> IntTensor<Self> {
        Self::int_from_data(TensorData::ones::<I, _>(shape), device)
    }

    fn int_full(
        shape: Shape,
        fill_value: I,
        device: &<NdArray<E> as Backend>::Device,
    ) -> IntTensor<Self> {
        Self::int_from_data(TensorData::full(shape, fill_value), device)
    }

    fn int_sum(tensor: IntTensor<Self>) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, NdArrayMathOps::sum)
    }

    fn int_sum_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, |tensor| NdArrayMathOps::sum_dim(tensor, dim))
    }

    fn int_prod(tensor: IntTensor<Self>) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, NdArrayMathOps::prod)
    }

    fn int_prod_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, |tensor| NdArrayMathOps::prod_dim(tensor, dim))
    }

    fn int_mean(tensor: IntTensor<Self>) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, NdArrayMathOps::mean)
    }

    fn int_mean_dim(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, |tensor| NdArrayMathOps::mean_dim(tensor, dim))
    }

    fn int_gather(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, |tensor| {
            execute_with_int_dtype!(indices => |indices| NdArrayMathOps::gather(dim, tensor, indices))
        })
    }

    fn int_scatter(
        dim: usize,
        tensor: IntTensor<Self>,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        execute_with_int_dtype!((tensor, value), |tensor, value| {
            execute_with_int_dtype!(indices => |indices| NdArrayMathOps::scatter(dim, tensor, indices, value))
        })
    }

    fn int_select(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, |tensor| {
            execute_with_int_dtype!(indices => |indices| NdArrayMathOps::select(tensor, dim, indices))
        })
    }

    fn int_select_assign(
        tensor: IntTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
        value: IntTensor<Self>,
    ) -> IntTensor<Self> {
        execute_with_int_dtype!((tensor, value), |tensor, value| {
            execute_with_int_dtype!(indices => |indices| NdArrayMathOps::select_assign(tensor, dim, indices, value))
        })
    }
    fn int_argmax(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor => |tensor| {
            match I::dtype() {
                DType::I64 => NdArrayMathOps::argmax::<i64>(tensor, dim).into(),
                _ => panic!("Unsupported integer type for argmax"),
            }
        })
    }

    fn int_argmin(tensor: IntTensor<Self>, dim: usize) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor => |tensor| {
            match I::dtype() {
                DType::I64 => NdArrayMathOps::argmin::<i64>(tensor, dim).into(),
                _ => panic!("Unsupported integer type for argmax"),
            }
        })
    }

    fn int_clamp_min(tensor: IntTensor<Self>, min: I) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, |tensor| NdArrayMathOps::clamp_min(
            tensor,
            min.elem()
        ))
    }

    fn int_clamp_max(tensor: IntTensor<Self>, max: I) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, |tensor| NdArrayMathOps::clamp_max(
            tensor,
            max.elem()
        ))
    }

    fn int_clamp(tensor: IntTensor<Self>, min: I, max: I) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, |tensor| NdArrayMathOps::clamp(
            tensor,
            min.elem(),
            max.elem()
        ))
    }

    fn int_abs(tensor: IntTensor<Self>) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, I, |tensor: NdArrayTensor<I>| {
            let array = tensor.array.mapv_into(|a| a.abs_elem()).into_shared();

            NdArrayTensor::new(array)
        })
    }

    fn int_into_float(tensor: IntTensor<Self>) -> FloatTensor<Self> {
        execute_with_int_dtype!(tensor, I => |tensor: NdArrayTensor<I> | {
            crate::dispatch_int_to_float_cast!(tensor, E)
        })
    }

    fn int_swap_dims(tensor: IntTensor<Self>, dim1: usize, dim2: usize) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, |tensor| NdArrayOps::swap_dims(tensor, dim1, dim2))
    }

    fn int_random(
        shape: Shape,
        distribution: Distribution,
        device: &NdArrayDevice,
    ) -> IntTensor<Self> {
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

    fn int_powi(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        execute_with_int_dtype!((lhs, rhs), I, |lhs, rhs| NdArrayMathOps::elementwise_op(
            lhs,
            rhs,
            |a: &I, b: &I| a.pow((*b).try_into().unwrap())
        ))
    }

    fn int_powf(lhs: IntTensor<Self>, rhs: FloatTensor<Self>) -> IntTensor<Self> {
        execute_with_int_dtype!(lhs, I, |lhs| {
                execute_with_float_dtype!(rhs, E => |rhs| {
                    NdArrayMathOps::elementwise_op(lhs, rhs, |a: &I, b: &E| {
                    (a.elem::<i64>().pow((*b).elem())).elem()
                })
            })
        })
    }

    fn int_powf_scalar(tensor: IntTensor<Self>, value: f32) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, I, |tensor: NdArrayTensor<I>| {
            let array = if value == 2.0 {
                // Happens often and is faster.
                tensor.array.mapv_into(|a| a * a).into_shared()
            } else if value.floor() == value {
                // Is faster then powf
                tensor
                    .array
                    .mapv_into(|a| a.pow(value as u32))
                    .into_shared()
            } else {
                // Default
                tensor.array.mapv_into(|a| a.powf_elem(value)).into_shared()
            };

            NdArrayTensor::new(array)
        })
    }

    fn int_permute(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, |tensor| NdArrayOps::permute(tensor, axes))
    }

    fn int_flip(tensor: IntTensor<Self>, axes: &[usize]) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, |tensor| NdArrayOps::flip(tensor, axes))
    }

    fn int_sign(tensor: IntTensor<Self>) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, NdArrayMathOps::sign_op)
    }

    fn int_expand(tensor: IntTensor<Self>, shape: Shape) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, |tensor| NdArrayOps::expand(tensor, shape))
    }

    fn int_cast(tensor: IntTensor<Self>, dtype: IntDType) -> IntTensor<Self> {
        fn cast<E1: IntNdArrayElement, E2: IntNdArrayElement>(
            tensor: &NdArrayTensor<E1>,
        ) -> NdArrayTensor<E2> {
            let array = tensor.array.mapv(|a| a.elem()).into_shared();
            NdArrayTensor { array }
        }

        match (&tensor, dtype) {
            // No cast
            (NdArrayTensorInt::I64(_), IntDType::I64) | (NdArrayTensorInt::I32(_), IntDType::I32) | (NdArrayTensorInt::U8(_), IntDType::U8) => {
                tensor
            }
            // I64 to x
            (NdArrayTensorInt::I64(tensor), IntDType::U8) => NdArrayTensorInt::U8(cast(tensor)),
            (NdArrayTensorInt::I64(tensor), IntDType::I32) => NdArrayTensorInt::I32(cast(tensor)),
            // I32 to x
            (NdArrayTensorInt::I32(tensor), IntDType::U8) => NdArrayTensorInt::U8(cast(tensor)),
            (NdArrayTensorInt::I32(tensor), IntDType::U8) => NdArrayTensorInt::U8(cast(tensor)),
            // U8 to x
            (NdArrayTensorInt::U8(tensor), IntDType::I64) => NdArrayTensorInt::I64(cast(tensor)),
            (NdArrayTensorInt::U8(tensor), IntDType::I32) => NdArrayTensorInt::I32(cast(tensor)),
            _ => panic!("Invalid cast types"),
        }
    }

    fn bitwise_and(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        execute_with_int_dtype!((lhs, rhs), |lhs, rhs| { NdArrayBitOps::bitand(lhs, rhs) })
    }

    fn bitwise_and_scalar(lhs: IntTensor<Self>, rhs: I) -> IntTensor<Self> {
        execute_with_int_dtype!(lhs, |lhs| {NdArrayBitOps::bitand_scalar(lhs, rhs.elem())})
    }

    fn bitwise_or(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        execute_with_int_dtype!((lhs, rhs), |lhs, rhs| { NdArrayBitOps::bitor(lhs, rhs) })
    }

    fn bitwise_or_scalar(lhs: IntTensor<Self>, rhs: I) -> IntTensor<Self> {
        execute_with_int_dtype!(lhs, |lhs| NdArrayBitOps::bitor_scalar(lhs, rhs.elem()))
    }

    fn bitwise_xor(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        execute_with_int_dtype!((lhs, rhs), |lhs, rhs| { NdArrayBitOps::bitxor(lhs, rhs) })
    }

    fn bitwise_xor_scalar(lhs: IntTensor<Self>, rhs: I) -> IntTensor<Self> {
        execute_with_int_dtype!(lhs, |lhs| NdArrayBitOps::bitxor_scalar(lhs, rhs.elem()))
    }

    fn bitwise_not(tensor: IntTensor<Self>) -> IntTensor<Self> {
        execute_with_int_dtype!(tensor, |tensor| { NdArrayBitOps::bitnot(tensor) })
    }

    fn bitwise_left_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        execute_with_int_dtype!((lhs, rhs), I, |lhs, rhs| {
            NdArrayMathOps::elementwise_op(lhs, rhs, |a: &I, b: &I| {
                (a.elem::<I>() << (*b).elem::<I>()).elem()
            })
        })
    }

    fn bitwise_left_shift_scalar(lhs: IntTensor<Self>, rhs: I) -> IntTensor<Self> {
        execute_with_int_dtype!(lhs, I, |lhs| {
            NdArrayMathOps::elementwise_op_scalar(lhs, |a: I| {
                (a.elem::<I>() << rhs.elem::<I>()).elem()
            })
        })
    }

    fn bitwise_right_shift(lhs: IntTensor<Self>, rhs: IntTensor<Self>) -> IntTensor<Self> {
        execute_with_int_dtype!((lhs, rhs), I, |lhs, rhs| {
            NdArrayMathOps::elementwise_op(lhs, rhs, |a: &I, b: &I| {
                (a.elem::<I>() >> (*b).elem::<I>()).elem()
            })
        })
    }

    fn bitwise_right_shift_scalar(lhs: IntTensor<Self>, rhs: I) -> IntTensor<Self> {
        execute_with_int_dtype!(lhs, I, |lhs| {
            NdArrayMathOps::elementwise_op_scalar(lhs, |a: I| {
                (a.elem::<I>() >> rhs.elem::<I>()).elem()
            })
        })
    }
}
