// Language
use alloc::vec::Vec;
use burn_tensor::cast::ToElement;
use burn_tensor::ops::FloatTensor;
use core::ops::Range;
use ndarray::Zip;

// Current crate
use super::{matmul::matmul, NdArrayMathOps, NdArrayOps};
use crate::element::{ExpElement, FloatNdArrayElement, IntNdArrayElement, QuantElement};
use crate::{execute_with_float_dtype, NdArrayDevice, NdArrayTensorFloat, SEED};
use crate::{tensor::NdArrayTensor, NdArray};

// Workspace crates
use burn_common::rand::get_seeded_rng;
use burn_tensor::{backend::Backend, ops::FloatTensorOps, ElementConversion, Shape, TensorData};
use burn_tensor::{DType, Distribution, FloatDType};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use libm::erf;

#[cfg(feature = "std")]
#[allow(dead_code)]
fn round_ties_even_wrapper(x: f64) -> f64 {
    x.round_ties_even()
}

#[cfg(not(feature = "std"))]
#[allow(dead_code)]
fn round_ties_even_wrapper(x: f64) -> f64 {
    if (x - x.floor()) == 0.5 {
        (x * 0.5).round() * 2.0
    } else {
        x.round()
    }
}

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> FloatTensorOps<Self>
    for NdArray<E, I, Q>
{
    fn float_from_data(data: TensorData, _device: &NdArrayDevice) -> FloatTensor<Self> {
        match data.dtype {
            DType::F64 => NdArrayTensorFloat::F64(NdArrayTensor::from_data(data)),
            DType::F32 => NdArrayTensorFloat::F32(NdArrayTensor::from_data(data)),
            _ => unimplemented!("Unsupported dtype for `float_from_data`"),
        }
    }

    fn float_random(
        shape: Shape,
        distribution: Distribution,
        device: &NdArrayDevice,
    ) -> FloatTensor<Self> {
        let mut seed = SEED.lock().unwrap();
        let mut rng = if let Some(rng_seeded) = seed.as_ref() {
            rng_seeded.clone()
        } else {
            get_seeded_rng()
        };
        let tensor = Self::float_from_data(
            TensorData::random::<E, _, _>(shape, distribution, &mut rng),
            device,
        );
        *seed = Some(rng);
        tensor
    }

    async fn float_into_data(tensor: FloatTensor<Self>) -> TensorData {
        match tensor {
            NdArrayTensorFloat::F32(tensor) => NdArrayOps::into_data(tensor),
            NdArrayTensorFloat::F64(tensor) => NdArrayOps::into_data(tensor),
        }
    }

    fn float_device(_tensor: &FloatTensor<Self>) -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    fn float_to_device(tensor: FloatTensor<Self>, _device: &NdArrayDevice) -> FloatTensor<Self> {
        tensor
    }

    fn float_empty(shape: Shape, device: &<NdArray<E> as Backend>::Device) -> FloatTensor<Self> {
        NdArray::<E>::float_zeros(shape, device)
    }

    fn float_add(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!((lhs, rhs), NdArrayMathOps::add)
    }

    fn float_add_scalar(lhs: FloatTensor<Self>, rhs: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(lhs, |lhs| NdArrayMathOps::add_scalar(lhs, rhs.elem()))
    }

    fn float_sub(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!((lhs, rhs), NdArrayMathOps::sub)
    }

    fn float_sub_scalar(lhs: FloatTensor<Self>, rhs: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(lhs, |lhs| NdArrayMathOps::sub_scalar(lhs, rhs.elem()))
    }

    fn float_mul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!((lhs, rhs), NdArrayMathOps::mul)
    }

    fn float_mul_scalar(lhs: FloatTensor<Self>, rhs: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(lhs, |lhs| NdArrayMathOps::mul_scalar(lhs, rhs.elem()))
    }

    fn float_div(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!((lhs, rhs), NdArrayMathOps::div)
    }

    fn float_div_scalar(lhs: FloatTensor<Self>, rhs: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(lhs, |lhs| NdArrayMathOps::div_scalar(lhs, rhs.elem()))
    }

    fn float_remainder(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!((lhs, rhs), NdArrayMathOps::remainder)
    }

    fn float_remainder_scalar(lhs: FloatTensor<Self>, rhs: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(lhs, |lhs| NdArrayMathOps::remainder_scalar(lhs, rhs.elem()))
    }

    fn float_matmul(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!((lhs, rhs), matmul)
    }

    fn float_neg(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        Self::float_mul_scalar(tensor, (-1f32).elem::<E>())
    }

    fn float_recip(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, NdArrayMathOps::recip)
    }

    fn float_swap_dims(tensor: FloatTensor<Self>, dim1: usize, dim2: usize) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayOps::swap_dims(tensor, dim1, dim2))
    }

    fn float_reshape(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayOps::reshape(tensor, shape))
    }

    fn float_gather(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: NdArrayTensor<I>,
    ) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::gather(
            dim, tensor, indices
        ))
    }

    fn float_scatter(
        dim: usize,
        tensor: FloatTensor<Self>,
        indices: NdArrayTensor<I>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        execute_with_float_dtype!((tensor, value), |tensor, value| NdArrayMathOps::scatter(
            dim, tensor, indices, value
        ))
    }

    fn float_select(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: NdArrayTensor<I>,
    ) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::select(
            tensor, dim, indices
        ))
    }

    fn float_select_assign(
        tensor: FloatTensor<Self>,
        dim: usize,
        indices: NdArrayTensor<I>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        execute_with_float_dtype!((tensor, value), |tensor, value| {
            NdArrayMathOps::select_assign(tensor, dim, indices, value)
        })
    }

    fn float_slice(tensor: FloatTensor<Self>, ranges: &[Range<usize>]) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayOps::slice(tensor, ranges))
    }

    fn float_slice_assign(
        tensor: FloatTensor<Self>,
        ranges: &[Range<usize>],
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        execute_with_float_dtype!((tensor, value), |tensor, value| {
            NdArrayOps::slice_assign(tensor, ranges, value)
        })
    }

    fn float_mask_where(
        tensor: FloatTensor<Self>,
        mask: NdArrayTensor<bool>,
        value: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        execute_with_float_dtype!((tensor, value), |tensor, value| {
            NdArrayMathOps::mask_where(tensor, mask, value)
        })
    }

    fn float_mask_fill(
        tensor: FloatTensor<Self>,
        mask: NdArrayTensor<bool>,
        value: E,
    ) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::mask_fill(
            tensor,
            mask,
            value.elem()
        ))
    }

    fn float_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> NdArrayTensor<bool> {
        execute_with_float_dtype!((lhs, rhs) => |lhs: NdArrayTensor<_>, rhs: NdArrayTensor<_>| {
            let output = Zip::from(&lhs.array)
                .and(&rhs.array)
                .map_collect(|&lhs_val, &rhs_val| (lhs_val == rhs_val))
                .into_shared();
            NdArrayTensor::new(output)
        })
    }

    fn float_equal_elem(lhs: FloatTensor<Self>, rhs: E) -> NdArrayTensor<bool> {
        execute_with_float_dtype!(lhs, E => |tensor: NdArrayTensor<E>| {
            let array = tensor.array.mapv(|a| a == rhs.elem::<E>()).into_shared();

            NdArrayTensor::new(array)
        })
    }

    fn float_greater(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> NdArrayTensor<bool> {
        let tensor = NdArray::<E>::float_sub(lhs, rhs);
        let zero = 0.elem();
        Self::float_greater_elem(tensor, zero)
    }

    fn float_greater_elem(lhs: FloatTensor<Self>, rhs: E) -> NdArrayTensor<bool> {
        execute_with_float_dtype!(lhs, E => |tensor: NdArrayTensor<E>| {
            let array = tensor.array.mapv(|a| a > rhs.elem::<E>()).into_shared();

            NdArrayTensor::new(array)
        })
    }

    fn float_greater_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> NdArrayTensor<bool> {
        let tensor = NdArray::<E>::float_sub(lhs, rhs);
        let zero = 0.elem();
        Self::float_greater_equal_elem(tensor, zero)
    }

    fn float_greater_equal_elem(lhs: FloatTensor<Self>, rhs: E) -> NdArrayTensor<bool> {
        execute_with_float_dtype!(lhs, E => |tensor: NdArrayTensor<E>| {
            let array = tensor.array.mapv(|a| a >= rhs.elem::<E>()).into_shared();

            NdArrayTensor::new(array)
        })
    }

    fn float_lower(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> NdArrayTensor<bool> {
        let tensor = NdArray::<E>::float_sub(lhs, rhs);
        let zero = 0.elem();
        Self::float_lower_elem(tensor, zero)
    }

    fn float_lower_elem(lhs: FloatTensor<Self>, rhs: E) -> NdArrayTensor<bool> {
        execute_with_float_dtype!(lhs, E => |tensor: NdArrayTensor<E>| {
            let array = tensor.array.mapv(|a| a < rhs.elem::<E>()).into_shared();

            NdArrayTensor::new(array)
        })
    }

    fn float_lower_equal(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> NdArrayTensor<bool> {
        let tensor = NdArray::<E>::float_sub(lhs, rhs);
        let zero = 0.elem();
        Self::float_lower_equal_elem(tensor, zero)
    }

    fn float_lower_equal_elem(lhs: FloatTensor<Self>, rhs: E) -> NdArrayTensor<bool> {
        execute_with_float_dtype!(lhs, E => |tensor: NdArrayTensor<E>| {
            let array = tensor.array.mapv(|a| a <= rhs.elem::<E>()).into_shared();

            NdArrayTensor::new(array)
        })
    }

    fn float_detach(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        tensor
    }

    fn float_mean(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, NdArrayMathOps::mean)
    }

    fn float_sum(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, NdArrayMathOps::sum)
    }

    fn float_mean_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::mean_dim(tensor, dim))
    }

    fn float_sum_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::sum_dim(tensor, dim))
    }

    fn float_argmax(tensor: FloatTensor<Self>, dim: usize) -> NdArrayTensor<I> {
        execute_with_float_dtype!(tensor => |tensor| NdArrayMathOps::argmax(tensor, dim))
    }

    fn float_argmin(tensor: FloatTensor<Self>, dim: usize) -> NdArrayTensor<I> {
        execute_with_float_dtype!(tensor => |tensor| NdArrayMathOps::argmin(tensor, dim))
    }

    fn float_exp(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: NdArrayTensor<E>| {
            let array = tensor.array.mapv_into(|a| a.exp_elem()).into_shared();

            NdArrayTensor::new(array)
        })
    }

    fn float_log(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: NdArrayTensor<E>| {
            let array = tensor.array.mapv_into(|a| a.log_elem()).into_shared();

            NdArrayTensor::new(array)
        })
    }

    fn float_prod(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, NdArrayMathOps::prod)
    }

    fn float_prod_dim(tensor: FloatTensor<Self>, dim: usize) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::prod_dim(tensor, dim))
    }

    fn float_log1p(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: NdArrayTensor<E>| {
            let array = tensor.array.mapv_into(|a| a.log1p_elem()).into_shared();

            NdArrayTensor::new(array)
        })
    }

    fn float_powf_scalar(tensor: FloatTensor<Self>, value: f32) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: NdArrayTensor<E>| {
            let array = if value == 2.0 {
                // Happens often and is faster.
                tensor.array.mapv_into(|a| a * a).into_shared()
            } else if value.floor() == value {
                // Is faster then powf
                tensor
                    .array
                    .mapv_into(|a| a.powi_elem(value as i32))
                    .into_shared()
            } else {
                // Default
                tensor.array.mapv_into(|a| a.powf_elem(value)).into_shared()
            };

            NdArrayTensor::new(array)
        })
    }

    fn float_sqrt(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: NdArrayTensor<E>| {
            let array = tensor.array.mapv_into(|a| a.sqrt_elem()).into_shared();

            NdArrayTensor::new(array)
        })
    }

    fn float_abs(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: NdArrayTensor<E>| {
            let array = tensor.array.mapv_into(|a| a.abs_elem()).into_shared();

            NdArrayTensor::new(array)
        })
    }

    fn float_cos(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: NdArrayTensor<E>| {
            let array = tensor
                .array
                .mapv_into(|a| (a.to_f64()).cos().elem())
                .into_shared();

            NdArrayTensor::new(array)
        })
    }

    fn float_sin(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: NdArrayTensor<E>| {
            let array = tensor
                .array
                .mapv_into(|a| (a.to_f64()).sin().elem())
                .into_shared();

            NdArrayTensor::new(array)
        })
    }

    fn float_tanh(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: NdArrayTensor<E>| {
            let array = tensor
                .array
                .mapv_into(|a| (a.to_f64()).tanh().elem())
                .into_shared();

            NdArrayTensor::new(array)
        })
    }

    fn float_round(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: NdArrayTensor<E>| {
            let array = tensor
                .array
                .mapv_into(|a| round_ties_even_wrapper(a.to_f64()).elem())
                .into_shared();

            NdArrayTensor::new(array)
        })
    }

    fn float_floor(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: NdArrayTensor<E>| {
            let array = tensor
                .array
                .mapv_into(|a| (a.to_f64()).floor().elem())
                .into_shared();

            NdArrayTensor::new(array)
        })
    }

    fn float_ceil(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: NdArrayTensor<E>| {
            let array = tensor
                .array
                .mapv_into(|a| (a.to_f64()).ceil().elem())
                .into_shared();

            NdArrayTensor::new(array)
        })
    }

    fn float_erf(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, E, |tensor: NdArrayTensor<E>| {
            let array = tensor
                .array
                .mapv_into(|a| erf(a.to_f64()).elem())
                .into_shared();

            NdArrayTensor::new(array)
        })
    }

    fn float_cat(tensors: Vec<FloatTensor<Self>>, dim: usize) -> FloatTensor<Self> {
        match &tensors[0] {
            NdArrayTensorFloat::F32(_) => {
                let tensors = tensors
                    .iter()
                    .map(|t| {
                        if let NdArrayTensorFloat::F32(tensor) = t {
                            tensor.array.view()
                        } else {
                            panic!("Concatenate data type mismatch (expected f32, got f64)")
                        }
                    })
                    .collect::<Vec<_>>();
                NdArrayTensorFloat::F32(NdArrayOps::concatenate(&tensors, dim))
            }
            NdArrayTensorFloat::F64(_) => {
                let tensors = tensors
                    .iter()
                    .map(|t| {
                        if let NdArrayTensorFloat::F64(tensor) = t {
                            tensor.array.view()
                        } else {
                            panic!("Concatenate data type mismatch (expected f64, got f32)")
                        }
                    })
                    .collect::<Vec<_>>();
                NdArrayTensorFloat::F64(NdArrayOps::concatenate(&tensors, dim))
            }
        }
    }

    fn float_clamp_min(tensor: FloatTensor<Self>, min: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::clamp_min(
            tensor,
            min.elem()
        ))
    }

    fn float_clamp_max(tensor: FloatTensor<Self>, max: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::clamp_max(
            tensor,
            max.elem()
        ))
    }

    fn float_clamp(tensor: FloatTensor<Self>, min: E, max: E) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayMathOps::clamp(
            tensor,
            min.elem(),
            max.elem()
        ))
    }

    fn float_into_int(tensor: FloatTensor<Self>) -> NdArrayTensor<I> {
        execute_with_float_dtype!(tensor, E => |tensor: NdArrayTensor<E>| {
            let array = tensor.array.mapv(|a| a.elem()).into_shared();
            NdArrayTensor { array }
        })
    }

    fn float_powf(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!((lhs, rhs), E, |lhs, rhs| NdArrayMathOps::elementwise_op(
            lhs,
            rhs,
            |a: &E, b: &E| a.powf(*b)
        ))
    }

    fn float_permute(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayOps::permute(tensor, axes))
    }

    fn float_flip(tensor: FloatTensor<Self>, axes: &[usize]) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayOps::flip(tensor, axes))
    }

    fn float_sign(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, NdArrayMathOps::sign_op)
    }

    fn float_expand(tensor: FloatTensor<Self>, shape: Shape) -> FloatTensor<Self> {
        execute_with_float_dtype!(tensor, |tensor| NdArrayOps::expand(tensor, shape))
    }

    fn float_cast(tensor: FloatTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        fn cast<E1: FloatNdArrayElement, E2: FloatNdArrayElement>(
            tensor: &NdArrayTensor<E1>,
        ) -> NdArrayTensor<E2> {
            let array = tensor.array.mapv(|a| a.elem()).into_shared();
            NdArrayTensor { array }
        }

        match (&tensor, dtype) {
            // No cast
            (NdArrayTensorFloat::F32(_), FloatDType::F32)
            | (NdArrayTensorFloat::F64(_), FloatDType::F64) => tensor,
            // F32 to F64
            (NdArrayTensorFloat::F32(tensor), FloatDType::F64) => {
                NdArrayTensorFloat::F64(cast(tensor))
            }
            // F64 to F32
            (NdArrayTensorFloat::F64(tensor), FloatDType::F32) => {
                NdArrayTensorFloat::F32(cast(tensor))
            }
            _ => panic!("Invalid cast types"),
        }
    }
}
