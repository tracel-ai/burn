use alloc::{vec, vec::Vec};

use burn_backend::{
    DType, ExecutionError, Shape, TensorData, TensorMetadata, TensorPrimitive, get_device_settings,
    ops::{FloatTensorOps, QTensorOps},
    quantization::{
        QParams, QuantLevel, QuantMode, QuantPropagation, QuantScheme, QuantStore, QuantValue,
        QuantizationParametersPrimitive, QuantizedBytes,
    },
    tensor::{FloatTensor, IntTensor, QuantizedTensor},
};
use burn_std::{FloatDType, IntDType};

use crate::{
    NdArray, NdArrayDevice, NdArrayQTensor, NdArrayTensor, SharedArray, element::QuantElement,
    execute_with_dtype, execute_with_int_dtype, execute_with_int_out_dtype,
    execute_with_numeric_dtype, slice,
};

use super::quantization::{QuantizationStrategy, SymmetricQuantization};
use super::{NdArrayMathOps, NdArrayOps};

impl QTensorOps<Self> for NdArray {
    fn q_from_data(data: TensorData, _device: &NdArrayDevice) -> QuantizedTensor<Self> {
        match data.dtype {
            DType::QFloat(scheme) => {
                let shape = data.shape.clone();
                let num_elements = data.num_elements();
                let q_bytes = QuantizedBytes {
                    bytes: data.into_bytes(),
                    scheme,
                    num_elements,
                };

                match scheme {
                    QuantScheme {
                        level: QuantLevel::Tensor | QuantLevel::Block(_),
                        mode: QuantMode::Symmetric,
                        value: QuantValue::Q8F | QuantValue::Q8S,
                        ..
                    } => {
                        // We can load QuantStore::U32 w/ QuantizedBytes impl
                        let (values, qparams) = q_bytes.into_vec_i8();
                        let data = TensorData::new(values, shape);
                        // Overwrite storage
                        let scheme = scheme.with_store(QuantStore::Native);

                        let qparams = qparams
                            .scales
                            .into_iter()
                            .map(|scales| QParams { scales })
                            .collect();

                        NdArrayQTensor {
                            qtensor: NdArrayTensor::from_data(data),
                            scheme,
                            qparams,
                        }
                    }
                    QuantScheme {
                        value:
                            QuantValue::Q4F
                            | QuantValue::Q4S
                            | QuantValue::Q2F
                            | QuantValue::Q2S
                            | QuantValue::E2M1
                            | QuantValue::E4M3
                            | QuantValue::E5M2,
                        ..
                    } => unimplemented!("from_data not supported for scheme {scheme:?}"),
                }
            }
            _ => panic!(
                "Invalid dtype (expected DType::QFloat, got {:?})",
                data.dtype
            ),
        }
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        let shape = tensor.shape();
        let data_f = tensor.into_data();
        let scales = qparams.scales.into_data().convert::<f32>();

        // Implement with ndarray instead of QuantizationStrategy?
        let (data, qparams) = match scheme {
            QuantScheme {
                level: QuantLevel::Tensor,
                mode: QuantMode::Symmetric,
                #[cfg(not(feature = "export_tests"))]
                    value: QuantValue::Q8F | QuantValue::Q8S,
                // For tests, "native" sub-byte quant serves as a reference for value equality.
                // Values are stored as i8 regardless.
                #[cfg(feature = "export_tests")]
                    value:
                    QuantValue::Q8F
                    | QuantValue::Q8S
                    | QuantValue::Q4F
                    | QuantValue::Q4S
                    | QuantValue::Q2F
                    | QuantValue::Q2S,
                store: QuantStore::Native,
                ..
            } => {
                let scales = scales.iter().next().unwrap();
                let strategy = QuantizationStrategy::PerTensorSymmetric(
                    SymmetricQuantization::init(scales, scheme.value),
                );
                let values = strategy.quantize(data_f.as_slice().unwrap());
                (
                    TensorData::quantized(values, shape.clone(), *scheme, &[scales]),
                    vec![QParams { scales }],
                )
            }
            QuantScheme {
                level: QuantLevel::Block(block_size),
                mode: QuantMode::Symmetric,
                #[cfg(not(feature = "export_tests"))]
                    value: QuantValue::Q8F | QuantValue::Q8S,
                #[cfg(feature = "export_tests")]
                    value:
                    QuantValue::Q8F
                    | QuantValue::Q8S
                    | QuantValue::Q4F
                    | QuantValue::Q4S
                    | QuantValue::Q2F
                    | QuantValue::Q2S,
                store: QuantStore::Native,
                ..
            } => {
                let scales = scales.as_slice().unwrap();
                let (strategy, qparams) = scales
                    .iter()
                    .map(|&s| {
                        (
                            SymmetricQuantization::init(s, scheme.value),
                            QParams { scales: s },
                        )
                    })
                    .unzip();
                let strategy = QuantizationStrategy::PerBlockSymmetric(strategy, *block_size);
                let values = strategy.quantize(data_f.as_slice().unwrap());
                (
                    TensorData::quantized(values, shape.clone(), *scheme, scales),
                    qparams,
                )
            }
            scheme => unimplemented!("Quantization not supported for scheme {scheme:?}"),
        };

        let num_elements = data.num_elements();
        let q_bytes = QuantizedBytes {
            bytes: data.into_bytes(),
            scheme: *scheme,
            num_elements,
        };
        let (values, _) = q_bytes.into_vec_i8();
        let data = TensorData::new(values, shape);

        NdArrayQTensor {
            qtensor: NdArrayTensor::from_data(data),
            scheme: *scheme,
            qparams,
        }
    }

    fn dequantize(tensor: QuantizedTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        let strategy = tensor.strategy();
        let scheme = tensor.scheme;
        let shape = tensor.shape();
        let data = match tensor.qtensor {
            NdArrayTensor::I8(storage) => {
                let data = storage.into_shared().into_iter().collect();
                dequantize(data, shape, scheme, &strategy, dtype.into())
            }
            _ => unreachable!(),
        };
        NdArrayTensor::from_data(data)
    }

    /// Matrix multiplication with at least one quantized operand.
    ///
    /// Fast path — BitNet b1.58 ternary weights: when `rhs` is a `Q2S` symmetric per-tensor weight
    /// (values in `{-1, 0, +1}`) and `lhs` is an `f32` activation, the product is computed WITHOUT
    /// dequantizing the weight and WITHOUT a single multiply in the inner loop: `+1 => add`,
    /// `-1 => subtract`, `0 => skip`, then the per-tensor scale `γ` is applied once per output
    /// element — the multiply-free compute path BitNet is built on. The result matches the
    /// dequantize-then-`float_matmul` path to within f32 rounding.
    ///
    /// Every other case (Q8, per-block, non-f32 activation, batched weights, ...) falls through to
    /// the regular `dequantize -> float_matmul` path — byte-for-byte the default behaviour.
    fn q_matmul(lhs: TensorPrimitive<Self>, rhs: TensorPrimitive<Self>) -> TensorPrimitive<Self> {
        if let (TensorPrimitive::Float(l), TensorPrimitive::QFloat(r)) = (&lhs, &rhs)
            && let Some(out) = ternary_matmul(l, r)
        {
            return TensorPrimitive::Float(out);
        }

        // Fallback: identical to the default `QTensorOps::q_matmul` — dequantize any quantized
        // operand, run the regular float matmul, and preserve quantization propagation.
        let mut propagation = QuantPropagation::Inhibit;
        let mut scheme = QuantScheme::default();
        let target_dtype: Option<FloatDType> = match (&lhs, &rhs) {
            (TensorPrimitive::Float(t), _) | (_, TensorPrimitive::Float(t)) => {
                Some(t.dtype().into())
            }
            _ => None,
        };
        let lhs = match lhs {
            TensorPrimitive::Float(lhs) => lhs,
            TensorPrimitive::QFloat(lhs) => {
                let settings = get_device_settings::<Self>(&Self::q_device(&lhs));
                propagation = settings.quantization.propagation;
                scheme = lhs.scheme;
                let float_dtype = target_dtype.unwrap_or(settings.float_dtype);
                Self::dequantize(lhs, float_dtype)
            }
        };
        let rhs = match rhs {
            TensorPrimitive::Float(rhs) => rhs,
            TensorPrimitive::QFloat(rhs) => {
                let settings = get_device_settings::<Self>(&Self::q_device(&rhs));
                propagation = settings.quantization.propagation;
                scheme = rhs.scheme;
                let float_dtype = target_dtype.unwrap_or(settings.float_dtype);
                Self::dequantize(rhs, float_dtype)
            }
        };
        let out_f = <Self as FloatTensorOps<Self>>::float_matmul(lhs, rhs);
        match propagation {
            QuantPropagation::Propagate => {
                TensorPrimitive::QFloat(Self::quantize_dynamic(out_f, &scheme))
            }
            QuantPropagation::Inhibit => TensorPrimitive::Float(out_f),
        }
    }

    fn q_device(_tensor: &QuantizedTensor<Self>) -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    fn q_to_device(
        tensor: QuantizedTensor<Self>,
        _device: &NdArrayDevice,
    ) -> QuantizedTensor<Self> {
        tensor
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: execute_with_dtype!(tensor.qtensor, E, |array: SharedArray<E>| {
                NdArrayOps::reshape(array, shape)
            }),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        let shape = tensor.qtensor.shape();
        let scales = tensor.qparams.iter().map(|q| q.scales).collect::<Vec<_>>();
        Ok(execute_with_numeric_dtype!(
            tensor.qtensor,
            E,
            |array: SharedArray<E>| {
                let values = array.into_iter().collect();
                TensorData::quantized(values, shape, tensor.scheme, &scales)
            }
        ))
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: execute_with_dtype!(tensor.qtensor, E, |array: SharedArray<E>| {
                NdArrayOps::swap_dims(array, dim1, dim2)
            }),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: execute_with_dtype!(tensor.qtensor, E, |array: SharedArray<E>| {
                NdArrayOps::permute(array, axes)
            }),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: execute_with_dtype!(tensor.qtensor, E, |array: SharedArray<E>| {
                NdArrayOps::flip(array, axes)
            }),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_gather(
        dim: usize,
        tensor: QuantizedTensor<Self>,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        let qtensor = execute_with_int_dtype!(indices, IntElem, |idx_array: SharedArray<
            IntElem,
        >|
         -> NdArrayTensor {
            execute_with_numeric_dtype!(tensor.qtensor, E, |array: SharedArray<E>| {
                NdArrayOps::gather(dim, array, idx_array)
            })
        });
        NdArrayQTensor {
            qtensor,
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_select(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        let qtensor = execute_with_int_dtype!(indices, IntElem, |idx_array: SharedArray<
            IntElem,
        >|
         -> NdArrayTensor {
            execute_with_numeric_dtype!(tensor.qtensor, E, |array: SharedArray<E>| {
                NdArrayMathOps::select(array, dim, idx_array)
            })
        });
        NdArrayQTensor {
            qtensor,
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_slice(
        tensor: QuantizedTensor<Self>,
        slices: &[burn_backend::Slice],
    ) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: slice!(tensor.qtensor, slices),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_argmax(tensor: QuantizedTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        execute_with_int_out_dtype!(out_dtype, I, {
            execute_with_numeric_dtype!(tensor.qtensor, E, |array: SharedArray<E>| {
                NdArrayMathOps::argmax::<I>(array, dim)
            })
        })
    }

    fn q_argmin(tensor: QuantizedTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        execute_with_int_out_dtype!(out_dtype, I, {
            execute_with_numeric_dtype!(tensor.qtensor, E, |array: SharedArray<E>| {
                NdArrayMathOps::argmin::<I>(array, dim)
            })
        })
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: execute_with_dtype!(tensor.qtensor, E, |array: SharedArray<E>| {
                NdArrayOps::expand(array, shape)
            }),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }
}

/// Native multiply-free ternary matmul (BitNet b1.58): an `f32` activation `lhs` times a `Q2S`
/// symmetric per-tensor ternary weight `rhs` (values in `{-1, 0, +1}`).
///
/// Returns `None` — so the caller falls back to the regular dequantize path — unless every
/// precondition holds: `rhs` is `Q2S` / `Symmetric` / per-tensor, `rhs` is a 2D weight `[K, N]`,
/// and `lhs` is an `f32` tensor whose last dim is `K`. Leading dims of `lhs` are flattened into the
/// row count `M`, so this covers the `Linear`-style `[.., K] x [K, N] -> [.., N]` case.
///
/// The inner loop contains no multiplies: `+1 => add`, `-1 => subtract`, `0 => skip`. The single
/// per-tensor scale `γ` is applied once per output element at the end — `M·N` multiplies instead of
/// the `M·N·K` of a dense matmul, with zeros never touched.
fn ternary_matmul(
    lhs: &FloatTensor<NdArray>,
    rhs: &NdArrayQTensor,
) -> Option<FloatTensor<NdArray>> {
    // Canonical BitNet b1.58 weight quantization: Q2S, symmetric, per-tensor.
    if !matches!(
        rhs.scheme,
        QuantScheme {
            value: QuantValue::Q2S,
            mode: QuantMode::Symmetric,
            level: QuantLevel::Tensor,
            ..
        }
    ) {
        return None;
    }
    // Only an f32 activation (the reference float dtype) takes the fast path.
    if !matches!(lhs, NdArrayTensor::F32(_)) {
        return None;
    }

    // rhs is a 2D weight [K, N].
    let wdims = rhs.qtensor.shape().to_vec();
    if wdims.len() != 2 {
        return None;
    }
    let (k, n) = (wdims[0], wdims[1]);

    // lhs is [.., M, K] with a matching K; flatten the leading dims into M.
    let ldims = lhs.shape().to_vec();
    if ldims.len() < 2 || *ldims.last().unwrap() != k {
        return None;
    }
    let m: usize = ldims[..ldims.len() - 1].iter().product();

    // Per-tensor scale γ.
    let gamma = rhs.qparams.first()?.scales;

    // Canonical row-major values for both operands (`into_data` normalizes any strided layout).
    let a_data = lhs.clone().into_data();
    let a = a_data.as_slice::<f32>().ok()?;
    let w_data = rhs.qtensor.clone().into_data();
    let w = w_data.as_slice::<i8>().ok()?;
    if a.len() != m * k || w.len() != k * n {
        return None;
    }

    // Multiply-free accumulation; one scale per output element at the end.
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        let arow = &a[i * k..(i + 1) * k];
        let orow = &mut out[i * n..(i + 1) * n];
        for kk in 0..k {
            let x = arow[kk];
            let wrow = &w[kk * n..(kk + 1) * n];
            for (o, &t) in orow.iter_mut().zip(wrow) {
                match t {
                    1 => *o += x,
                    -1 => *o -= x,
                    _ => {} // 0 -> skip
                }
            }
        }
        for o in orow.iter_mut() {
            *o *= gamma;
        }
    }

    let mut out_dims = ldims[..ldims.len() - 1].to_vec();
    out_dims.push(n);
    Some(NdArrayTensor::from_data(TensorData::new(
        out,
        Shape::from(out_dims),
    )))
}

fn dequantize<Q: QuantElement>(
    data: Vec<Q>,
    shape: Shape,
    scheme: QuantScheme,
    strategy: &QuantizationStrategy,
    dtype: DType,
) -> TensorData {
    let qparams = match strategy {
        QuantizationStrategy::PerTensorSymmetric(quant) => vec![quant.scale],
        QuantizationStrategy::PerBlockSymmetric(quant, _block_size) => {
            quant.iter().map(|q| q.scale).collect()
        }
    };
    let q_bytes = QuantizedBytes::new(data, scheme, &qparams);
    let (values, _qparams) = q_bytes.into_vec_i8();
    TensorData::new(strategy.dequantize(&values), shape).convert_dtype(dtype)
}
