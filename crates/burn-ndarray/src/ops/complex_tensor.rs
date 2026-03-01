use crate::ops::{NdArrayMathOps, NdArrayOps};
use crate::{execute_with_complex_dtype, execute_with_float_dtype};
use burn_common::rand::get_seeded_rng;
use burn_complex::base::ComplexElem;
use burn_complex::base::element::ToComplex;
use burn_complex::base::{
    ComplexTensor, ComplexTensorBackend, ComplexTensorOps, InterleavedLayout, element::Complex32,
};
use ndarray::{ArrayD, IxDyn};

use crate::{
    FloatNdArrayElement, IntNdArrayElement, NdArray, NdArrayDevice, NdArrayTensor, QuantElement,
    SEED, SharedArray,
};
use burn_tensor::{Distribution, Shape, TensorData, TensorMetadata, backend::Backend};

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> ComplexTensorBackend
    for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
    type InnerBackend = NdArray<E, I, Q>;

    type ComplexTensorPrimitive = NdArrayTensor;
    type ComplexElem = Complex32;

    type Layout = InterleavedLayout;
}

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement>
    burn_complex::base::ComplexTensorOps<NdArray<E, I, Q>> for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
    type Layout = burn_complex::base::InterleavedLayout;
    fn real(tensor: ComplexTensor<Self>) -> NdArrayTensor {
        {
            {
                {
                    match tensor {
                        crate::NdArrayTensor::Complex32(storage) =>
                        {
                            #[allow(unused)]
                            (|array: SharedArray<ComplexElem<Self>>| array.mapv_into(|a| a.real))(
                                storage.into_shared(),
                            )
                            .into()
                        }

                        crate::NdArrayTensor::Complex64(storage) =>
                        {
                            #[allow(unused)]
                            (|array: SharedArray<ComplexElem<Self>>| array.mapv_into(|a| a.real))(
                                storage.into_shared(),
                            )
                            .into()
                        }
                        #[allow(unreachable_patterns)]
                        other => unimplemented!("unsupported dtype: {:?}", other.dtype()),
                    }
                }
            }
        }
    }

    fn imag(tensor: ComplexTensor<Self>) -> NdArrayTensor {
        //tensor.int
        execute_with_complex_dtype!(tensor, |array: SharedArray<ComplexElem<Self>>| {
            array.mapv_into(|a: ComplexElem<Self>| a.imag).into_shared()
        })
    }
    //NOTE: May want to change complex types from ComplexE to Complex<E> in the future to match the element type (and allow quantized complex tensors)
    fn to_complex(tensor: NdArrayTensor) -> NdArrayTensor {
        execute_with_float_dtype!(tensor, E, |array: SharedArray<E>| {
            array
                .mapv_into(|a: E| Complex32::new(a.to_f32(), 0.0))
                .into_shared()
        })
    }
    //     fn complex_from_data(data: TensorData, _device: &NdArrayDevice) -> NdArrayTensor {
    //         NdArrayTensor::from_data(data)
    //     }

    //     fn complex_random(
    //         shape: Shape,
    //         distribution: Distribution,
    //         _device: &NdArrayDevice,
    //     ) -> NdArrayTensor {
    //         let mut seed = SEED.lock().unwrap();
    //         let mut rng = if let Some(rng_seeded) = seed.as_ref() {
    //             rng_seeded.clone()
    //         } else {
    //             get_seeded_rng()
    //         };
    //         let data = TensorData::random::<Complex32, _, _>(shape, distribution, &mut rng);
    //         *seed = Some(rng);
    //         NdArrayTensor::from_data(data)
    //     }

    //     fn complex_shape(tensor: &NdArrayTensor) -> Shape {
    //         tensor.shape()
    //     }

    //     fn complex_to_data(tensor: &NdArrayTensor) -> TensorData {
    //         // Non-consuming view -> clone and consume to leverage the existing path.
    //         tensor.clone().into_data()
    //     }

    //     fn complex_device(_tensor: &NdArrayTensor) -> NdArrayDevice {
    //         NdArrayDevice::Cpu
    //     }

    //     fn complex_to_device(tensor: NdArrayTensor, _device: &NdArrayDevice) -> NdArrayTensor {
    //         tensor
    //     }

    //     fn complex_into_data(tensor: NdArrayTensor) -> TensorData {
    //         tensor.into_data()
    //     }

    //     fn complex_reshape(tensor: NdArrayTensor, shape: Shape) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, |t| NdArrayOps::reshape(t, shape))
    //     }

    //     fn complex_transpose(tensor: NdArrayTensor) -> NdArrayTensor {
    //         // 2D transpose; mirrors float/int backends (uses backend helper if available)
    //         execute_with_complex_dtype!(tensor, |t| NdArrayMathOps::transpose(t))
    //     }

    //     fn complex_mul(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
    //         // Uses Complex32 Mul impl (a+bi)*(c+di)
    //         execute_with_complex_dtype!((lhs, rhs), NdArrayMathOps::mul)
    //     }

    //     fn complex_div(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
    //         // (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c^2 + d^2)
    //         execute_with_complex_dtype!(
    //             (lhs, rhs),
    //             |lhs: SharedArray<Complex32>, rhs: SharedArray<Complex32>| {
    //                 NdArrayMathOps::elementwise_op(lhs, rhs, |a: &Complex32, b: &Complex32| {
    //                     let denom = b.real * b.real + b.imag * b.imag;
    //                     Complex32::new(
    //                         (a.real * b.real + a.imag * b.imag) / denom,
    //                         (a.imag * b.real - a.real * b.imag) / denom,
    //                     )
    //                 })
    //             }
    //         )
    //     }

    //     fn complex_neg(tensor: NdArrayTensor) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, |t| {
    //             NdArrayMathOps::elementwise_op_scalar(t, |z: Complex32| -z)
    //         })
    //     }

    //     fn complex_conj(tensor: NdArrayTensor) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, |t| {
    //             NdArrayMathOps::elementwise_op_scalar(t, |z: Complex32| z.conj())
    //         })
    //     }

    //     fn complex_abs(tensor: NdArrayTensor) -> <Self as Backend>::FloatTensorPrimitive {
    //         // Return Float tensor (E). For Complex32 -> f32 magnitudes, cast to E if needed.
    //         execute_with_complex_dtype!(tensor, |t: SharedArray<Complex32>| -> NdArrayTensor {
    //             let mag: SharedArray<f32> = t.mapv(|z| z.abs()).into_shared();
    //             // If E != f32, cast. Otherwise this is a no-op.
    //             cast_to_dtype(mag, E::dtype())
    //         })
    //     }

    //     fn complex_arg(tensor: NdArrayTensor) -> <Self as Backend>::FloatTensorPrimitive {
    //         // atan2(imag, real)
    //         execute_with_complex_dtype!(tensor, |t: SharedArray<Complex32>| -> NdArrayTensor {
    //             let phase: SharedArray<f32> = t.mapv(|z| z.imag.atan2(z.real)).into_shared();
    //             cast_to_dtype(phase, E::dtype())
    //         })
    //     }

    //     fn complex_from_parts(
    //         real: <Self as Backend>::FloatTensorPrimitive,
    //         imag: <Self as Backend>::FloatTensorPrimitive,
    //     ) -> NdArrayTensor {
    //         // Ensure F32 tensors, then zip to Complex32
    //         let real_f32 = cast_to_dtype(real, DType::F32);
    //         let imag_f32 = cast_to_dtype(imag, DType::F32);
    //         execute_with_complex_dtype!(
    //             (real_f32, imag_f32),
    //             |r: SharedArray<f32>, i: SharedArray<f32>| {
    //                 NdArrayMathOps::elementwise_op(r, i, |a: &f32, b: &f32| Complex32::new(*a, *b))
    //             }
    //         )
    //     }

    //     fn complex_from_polar(
    //         magnitude: <Self as Backend>::FloatTensorPrimitive,
    //         phase: <Self as Backend>::FloatTensorPrimitive,
    //     ) -> NdArrayTensor {
    //         // z = mag * (cos(phase) + i sin(phase))
    //         let mag_f32 = cast_to_dtype(magnitude, DType::F32);
    //         let pha_f32 = cast_to_dtype(phase, DType::F32);
    //         execute_with_complex_dtype!(
    //             (mag_f32, pha_f32),
    //             |m: SharedArray<f32>, p: SharedArray<f32>| {
    //                 NdArrayMathOps::elementwise_op(m, p, |&mag: &f32, &ph: &f32| {
    //                     Complex32::new(mag * ph.cos(), mag * ph.sin())
    //                 })
    //             }
    //         )
    //     }

    //     fn complex_exp(tensor: NdArrayTensor) -> NdArrayTensor {
    //         // exp(a+bi) = e^a (cos b + i sin b)
    //         execute_with_complex_dtype!(tensor, |t| {
    //             NdArrayMathOps::elementwise_op_scalar(t, |z: Complex32| {
    //                 let ea = z.real.exp();
    //                 Complex32::new(ea * z.imag.cos(), ea * z.imag.sin())
    //             })
    //         })
    //     }

    //     fn complex_log(tensor: NdArrayTensor) -> NdArrayTensor {
    //         // ln(a+bi) = ln|z| + i*arg(z)
    //         execute_with_complex_dtype!(tensor, |t| {
    //             NdArrayMathOps::elementwise_op_scalar(t, |z: Complex32| {
    //                 Complex32::new(z.abs().ln(), z.imag.atan2(z.real))
    //             })
    //         })
    //     }

    //     fn complex_sqrt(tensor: NdArrayTensor) -> NdArrayTensor {
    //         // principal sqrt: sqrt(r)(cos φ/2 + i sin φ/2)
    //         execute_with_complex_dtype!(tensor, |t| {
    //             NdArrayMathOps::elementwise_op_scalar(t, |z: Complex32| {
    //                 let r = z.abs();
    //                 let phi = z.imag.atan2(z.real);
    //                 let s = r.sqrt();
    //                 Complex32::new(s * (phi * 0.5).cos(), s * (phi * 0.5).sin())
    //             })
    //         })
    //     }

    //     fn complex_sin(tensor: NdArrayTensor) -> NdArrayTensor {
    //         // sin(a+bi) = sin a cosh b + i cos a sinh b
    //         execute_with_complex_dtype!(tensor, |t| {
    //             NdArrayMathOps::elementwise_op_scalar(t, |z: Complex32| {
    //                 Complex32::new(z.real.sin() * z.imag.cosh(), z.real.cos() * z.imag.sinh())
    //             })
    //         })
    //     }

    //     fn complex_cos(tensor: NdArrayTensor) -> NdArrayTensor {
    //         // cos(a+bi) = cos a cosh b - i sin a sinh b
    //         execute_with_complex_dtype!(tensor, |t| {
    //             NdArrayMathOps::elementwise_op_scalar(t, |z: Complex32| {
    //                 Complex32::new(z.real.cos() * z.imag.cosh(), -z.real.sin() * z.imag.sinh())
    //             })
    //         })
    //     }

    //     // ---------- indexing / view / shape ops ----------

    //     fn select(
    //         tensor: NdArrayTensor,
    //         dim: usize,
    //         indices: burn_tensor::Tensor<NdArray<E, I, Q>, 1, burn_tensor::Int>,
    //     ) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, |t: SharedArray<Complex32>| -> NdArrayTensor {
    //             let idx = indices.into_primitive();
    //             execute_with_int_dtype!(idx, |i| NdArrayMathOps::select(t, dim, i))
    //         })
    //     }

    //     fn select_assign(
    //         tensor: NdArrayTensor,
    //         dim: usize,
    //         indices: burn_tensor::Tensor<NdArray<E, I, Q>, 1, burn_tensor::Int>,
    //         values: NdArrayTensor,
    //     ) -> NdArrayTensor {
    //         execute_with_complex_dtype!((tensor, values), |t, v| -> NdArrayTensor {
    //             let idx = indices.into_primitive();
    //             execute_with_int_dtype!(idx, |i| NdArrayMathOps::select_assign(t, dim, i, v))
    //         })
    //     }

    //     fn complex_slice(tensor: NdArrayTensor, slices: &[burn_tensor::Slice]) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, |t| NdArrayOps::slice(t, slices))
    //     }

    //     fn complex_slice_assign(
    //         tensor: NdArrayTensor,
    //         ranges: &[burn_tensor::Slice],
    //         value: NdArrayTensor,
    //     ) -> NdArrayTensor {
    //         execute_with_complex_dtype!((tensor, value), |t, v| NdArrayOps::slice_assign(
    //             t, ranges, v
    //         ))
    //     }

    //     fn complex_swap_dims(tensor: NdArrayTensor, dim1: usize, dim2: usize) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, |t| NdArrayOps::swap_dims(t, dim1, dim2))
    //     }

    //     fn complex_repeat_dim(tensor: NdArrayTensor, dim: usize, times: usize) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, |t| NdArrayOps::repeat_dim(t, dim, times))
    //     }

    //     fn complex_equal(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
    //         execute_with_complex_dtype!((lhs, rhs), NdArrayMathOps::equal)
    //     }

    //     fn complex_not_equal(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
    //         execute_with_complex_dtype!((lhs, rhs), NdArrayMathOps::not_equal)
    //     }

    //     fn complex_cat(tensors: Vec<NdArrayTensor>, dim: usize) -> NdArrayTensor {
    //         // Reuse your general cat macro, restricted to Complex32
    //         cat_with_dtype!(tensors, dim, [Complex32])
    //     }

    //     fn complex_any(tensor: NdArrayTensor) -> NdArrayTensor {
    //         // any over "truthiness" of complex -> uses ToElement::to_bool (!= 0 or imag != 0)
    //         execute_with_complex_dtype!(tensor, NdArrayMathOps::any)
    //     }

    //     fn complex_any_dim(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, |t| NdArrayMathOps::any_dim(t, dim))
    //     }

    //     fn complex_all(tensor: NdArrayTensor) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, NdArrayMathOps::all)
    //     }

    //     fn complex_all_dim(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, |t| NdArrayMathOps::all_dim(t, dim))
    //     }

    //     fn complex_permute(tensor: NdArrayTensor, axes: &[usize]) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, |t| NdArrayOps::permute(t, axes))
    //     }

    //     fn complex_expand(tensor: NdArrayTensor, shape: Shape) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, |t| NdArrayOps::expand(t, shape))
    //     }

    //     fn complex_flip(tensor: NdArrayTensor, axes: &[usize]) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, |t| NdArrayOps::flip(t, axes))
    //     }

    //     fn complex_unfold(
    //         tensor: NdArrayTensor,
    //         dim: usize,
    //         size: usize,
    //         step: usize,
    //     ) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, |t| NdArrayOps::unfold(t, dim, size, step))
    //     }

    //     fn complex_select(
    //         tensor: NdArrayTensor,
    //         dim: usize,
    //         indices: burn_tensor::ops::IntTensor<NdArray<E, I, Q>>,
    //     ) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, |t: SharedArray<Complex32>| -> NdArrayTensor {
    //             let idx = indices.into_primitive();
    //             execute_with_int_dtype!(idx, |i| NdArrayMathOps::select(t, dim, i))
    //         })
    //     }

    //     // ---------- reductions ----------

    //     fn complex_sum(tensor: NdArrayTensor) -> NdArrayTensor {
    //         // Complex reduction: sum real and imag independently
    //         execute_with_complex_dtype!(tensor, NdArrayMathOps::sum)
    //     }

    //     fn complex_sum_dim(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, |t| NdArrayMathOps::sum_dim(t, dim))
    //     }

    //     fn complex_prod(tensor: NdArrayTensor) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, NdArrayMathOps::prod)
    //     }

    //     fn complex_prod_dim(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, |t| NdArrayMathOps::prod_dim(t, dim))
    //     }

    //     fn complex_mean(tensor: NdArrayTensor) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, NdArrayMathOps::mean)
    //     }

    //     fn complex_mean_dim(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, |t| NdArrayMathOps::mean_dim(t, dim))
    //     }

    //     // ---------- elementwise "mod" & comparisons ----------
    //     // Remainder isn't defined for complex numbers mathematically.
    //     // Here we follow an elementwise remainder on components: (a% c) + i (b % d).

    //     fn complex_remainder(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
    //         execute_with_complex_dtype!(
    //             (lhs, rhs),
    //             |l: SharedArray<Complex32>, r: SharedArray<Complex32>| {
    //                 NdArrayMathOps::elementwise_op(l, r, |a: &Complex32, b: &Complex32| {
    //                     Complex32::new(a.real % b.real, a.imag % b.imag)
    //                 })
    //             }
    //         )
    //     }

    //     fn complex_remainder_scalar(
    //         lhs: NdArrayTensor,
    //         rhs: <NdArray<E, I, Q> as burn_complex::base::ComplexTensorBackend>::ComplexElem,
    //     ) -> NdArrayTensor {
    //         let rhs = rhs.to_complex();
    //         execute_with_complex_dtype!(lhs, |l| {
    //             NdArrayMathOps::elementwise_op_scalar(l, |a: Complex32| {
    //                 Complex32::new(a.real % rhs.real, a.imag % rhs.imag)
    //             })
    //         })
    //     }

    //     fn complex_equal_elem(
    //         lhs: NdArrayTensor,
    //         rhs: <NdArray<E, I, Q> as burn_complex::base::ComplexTensorBackend>::ComplexElem,
    //     ) -> NdArrayTensor {
    //         let rhs = rhs.to_complex();
    //         execute_with_complex_dtype!(lhs, |l| NdArrayMathOps::elementwise_op_scalar(
    //             l,
    //             |a: Complex32| a == rhs
    //         ))
    //     }

    //     fn complex_not_equal_elem(
    //         lhs: NdArrayTensor,
    //         rhs: <NdArray<E, I, Q> as burn_complex::base::ComplexTensorBackend>::ComplexElem,
    //     ) -> NdArrayTensor {
    //         let rhs = rhs.to_complex();
    //         execute_with_complex_dtype!(lhs, |l| NdArrayMathOps::elementwise_op_scalar(
    //             l,
    //             |a: Complex32| a != rhs
    //         ))
    //     }

    //     // ---------- masked ops / gather / scatter ----------

    //     fn complex_mask_where(
    //         tensor: NdArrayTensor,
    //         mask: NdArrayTensor,
    //         source: NdArrayTensor,
    //     ) -> NdArrayTensor {
    //         execute_with_complex_dtype!((tensor, source), |t, s| {
    //             NdArrayMathOps::mask_where(t, mask.bool(), s)
    //         })
    //     }

    //     fn complex_mask_fill(
    //         tensor: NdArrayTensor,
    //         mask: NdArrayTensor,
    //         value: <NdArray<E, I, Q> as burn_complex::base::ComplexTensorBackend>::ComplexElem,
    //     ) -> NdArrayTensor {
    //         let value = value.to_complex();
    //         execute_with_complex_dtype!(tensor, |t| {
    //             NdArrayMathOps::mask_fill(t, mask.bool(), value)
    //         })
    //     }

    //     fn complex_gather(dim: usize, tensor: NdArrayTensor, indices: NdArrayTensor) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, |t: SharedArray<Complex32>| -> NdArrayTensor {
    //             execute_with_complex_dtype!(indices, |i| NdArrayMathOps::gather(dim, t, i))
    //         })
    //     }

    //     fn complex_scatter(
    //         dim: usize,
    //         tensor: NdArrayTensor,
    //         indices: NdArrayTensor,
    //         values: NdArrayTensor,
    //     ) -> NdArrayTensor {
    //         execute_with_complex_dtype!((tensor, values), |t, v| -> NdArrayTensor {
    //             execute_with_complex_dtype!(indices, |i| NdArrayMathOps::scatter(dim, t, i, v))
    //         })
    //     }

    //     fn complex_sign(tensor: NdArrayTensor) -> NdArrayTensor {
    //         // sign(z) := z / |z|, and 0 maps to 0 to avoid NaN
    //         execute_with_complex_dtype!(tensor, |t| {
    //             NdArrayMathOps::elementwise_op_scalar(t, |z: Complex32| {
    //                 let r = z.abs();
    //                 if r == 0.0 {
    //                     Complex32::new(0.0, 0.0)
    //                 } else {
    //                     Complex32::new(z.real / r, z.imag / r)
    //                 }
    //             })
    //         })
    //     }

    //     // ---------- argmax/argmin/max/min & max_abs/min_abs ----------

    //     // ---------- clamp ----------

    //     fn complex_clamp(
    //         tensor: NdArrayTensor,
    //         min: <NdArray<E, I, Q> as burn_complex::base::ComplexTensorBackend>::ComplexElem,
    //         max: <NdArray<E, I, Q> as burn_complex::base::ComplexTensorBackend>::ComplexElem,
    //     ) -> NdArrayTensor {
    //         let min = min.to_complex();
    //         let max = max.to_complex();
    //         execute_with_complex_dtype!(tensor, |t| NdArrayMathOps::clamp(t, min, max))
    //     }

    //     // ---------- pow ----------

    //     fn complex_powi(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
    //         // integer power (component-wise exponent on complex is ambiguous);
    //         // Here use z^n via repeated multiply if NdArrayMathOps::powi exists; otherwise map.
    //         execute_with_complex_dtype!(
    //             (lhs, rhs),
    //             |l: SharedArray<Complex32>, r: SharedArray<Complex32>| {
    //                 NdArrayMathOps::elementwise_op(l, r, |z: &Complex32, n: &Complex32| {
    //                     // Use real part of exponent as integer (common convention for *_powi)
    //                     let k = n.real as i32;
    //                     if k == 0 {
    //                         return Complex32::new(1.0, 0.0);
    //                     }
    //                     let mut base = *z;
    //                     let mut exp = k.abs() as u32;
    //                     let mut acc = Complex32::new(1.0, 0.0);
    //                     while exp > 0 {
    //                         if (exp & 1) == 1 {
    //                             acc = acc * base;
    //                         }
    //                         base = base * base;
    //                         exp >>= 1;
    //                     }
    //                     if k < 0 {
    //                         // 1/acc
    //                         let denom = acc.real * acc.real + acc.imag * acc.imag;
    //                         Complex32::new(acc.real / denom, -acc.imag / denom)
    //                     } else {
    //                         acc
    //                     }
    //                 })
    //             }
    //         )
    //     }

    //     fn complex_powi_scalar(
    //         lhs: NdArrayTensor,
    //         rhs: <NdArray<E, I, Q> as burn_complex::base::ComplexTensorBackend>::ComplexElem,
    //     ) -> NdArrayTensor {
    //         let k = rhs.to_complex().real as i32;
    //         execute_with_complex_dtype!(lhs, |l| {
    //             // fast powi by scalar exponent
    //             if k == 0 {
    //                 NdArrayOps::full_like(l, Complex32::new(1.0, 0.0))
    //             } else {
    //                 NdArrayMathOps::elementwise_op_scalar(l, |mut z: Complex32| {
    //                     // binary exponentiation on scalar
    //                     let mut base = z;
    //                     let mut exp = k.abs() as u32;
    //                     let mut acc = Complex32::new(1.0, 0.0);
    //                     while exp > 0 {
    //                         if (exp & 1) == 1 {
    //                             acc = acc * base;
    //                         }
    //                         base = base * base;
    //                         exp >>= 1;
    //                     }
    //                     if k < 0 {
    //                         let denom = acc.real * acc.real + acc.imag * acc.imag;
    //                         Complex32::new(acc.real / denom, -acc.imag / denom)
    //                     } else {
    //                         acc
    //                     }
    //                 })
    //             }
    //         })
    //     }

    //     // ---------- matmul & scans ----------

    //     fn complex_matmul(lhs: NdArrayTensor, rhs: NdArrayTensor) -> NdArrayTensor {
    //         // If your NdArrayOps::matmul uses elementwise + sum with generic T implementing +,*,
    //         // this will work out of the box thanks to Complex32::add/mul.
    //         execute_with_complex_dtype!((lhs, rhs), NdArrayOps::matmul)
    //     }

    //     fn complex_cumsum(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, |t| NdArrayMathOps::cumsum(t, dim))
    //     }

    //     fn complex_cumprod(tensor: NdArrayTensor, dim: usize) -> NdArrayTensor {
    //         execute_with_complex_dtype!(tensor, |t| NdArrayMathOps::cumprod(t, dim))
    //     }

    //     fn complex_add(
    //         lhs: burn_complex::base::ComplexTensor<NdArray<E, I, Q>>,
    //         rhs: burn_complex::base::ComplexTensor<NdArray<E, I, Q>>,
    //     ) -> burn_complex::base::ComplexTensor<NdArray<E, I, Q>> {
    //         let l = lhs.into_primitive(); // expected: -> NdArrayTensor with Complex32 dtype
    //         let r = rhs.into_primitive();
    //         let out = Self::complex_add_primitive(l, r);
    //         burn_complex::base::ComplexTensor::from_primitive(out)
    //     }

    //     fn complex_sub(
    //         lhs: burn_complex::base::ComplexTensor<NdArray<E, I, Q>>,
    //         rhs: burn_complex::base::ComplexTensor<NdArray<E, I, Q>>,
    //     ) -> burn_complex::base::ComplexTensor<NdArray<E, I, Q>> {
    //         execute_with_complex_dtype!((lhs, rhs), NdArrayMathOps::sub)
    //     }

    //     fn complex_real(tensor: burn_complex::base::ComplexTensor<NdArray<E, I, Q>>) -> NdArrayTensor {
    //         let p = tensor.into_primitive();
    //         execute_with_complex_dtype!(p, |t| {
    //             let r: SharedArray<f32> = t.mapv(|z: Complex32| z.real).into_shared();
    //             cast_to_dtype(r, E::dtype())
    //         })
    //     }

    //     fn complex_imag(tensor: burn_complex::base::ComplexTensor<NdArray<E, I, Q>>) -> NdArrayTensor {
    //         let p = tensor.into_primitive();
    //         execute_with_complex_dtype!(p, |t| {
    //             let im: SharedArray<f32> = t.mapv(|z: Complex32| z.imag).into_shared();
    //             cast_to_dtype(im, E::dtype())
    //         })
    //     }

    //     fn complex_powc(
    //         lhs: ComplexTensor<NdArray<E, I, Q>>,
    //         rhs: ComplexTensor<NdArray<E, I, Q>>,
    //     ) -> ComplexTensor<NdArray<E, I, Q>> {
    //         // a^b = exp(b * ln(a))
    //         let ln_lhs = Self::complex_log(lhs);
    //         let product = Self::complex_mul(rhs, ln_lhs);
    //         Self::complex_exp(product)
    //     }

    //     fn complex_tan(tensor: ComplexTensor<NdArray<E, I, Q>>) -> ComplexTensor<NdArray<E, I, Q>> {
    //         // tan(z) = sin(z) / cos(z)
    //         let sin_z = Self::complex_sin(tensor.clone());
    //         let cos_z = Self::complex_cos(tensor);
    //         Self::complex_div(sin_z, cos_z)
    //     }
    // }
}

// TODO: actually fix this
/// Macro to execute an operation for complex dtypes (currently Complex32).
#[macro_export]
macro_rules! execute_with_complex_dtype {
        // Binary op: type automatically inferred by the compiler
        (($lhs:expr, $rhs:expr), $op:expr) => {{
            $crate::execute_with_complex_dtype!(($lhs, $rhs), E, $op)
        }};
        // Binary op
        (($lhs:expr, $rhs:expr),$element:ident, $op:expr) => {{
            $crate::execute_with_dtype!(($lhs, $rhs), $element, $op, [
                Complex32 => burn_complex::base::element::Complex32,
                Complex64 => burn_complex::base::element::Complex64
            ])
        }};
        // Unary op: type automatically inferred by the compiler
        ($tensor:expr, $op:expr) => {{
            $crate::execute_with_complex_dtype!($tensor, E, $op)
        }};

        // Unary op
        ($tensor:expr, $element:ident, $op:expr) => {{
            $crate::execute_with_dtype!($tensor, $element, $op, [
                Complex32 => burn_complex::base::element::Complex32,
                Complex64 => burn_complex::base::element::Complex64
            ])
        }};
    }
