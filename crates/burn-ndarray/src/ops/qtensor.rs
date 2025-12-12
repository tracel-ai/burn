use alloc::{vec, vec::Vec};

use burn_tensor::{
    DType, Shape, TensorData, TensorMetadata,
    backend::ExecutionError,
    ops::{FloatTensor, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{
        QParams, QuantLevel, QuantMode, QuantScheme, QuantStore, QuantValue,
        QuantizationParametersPrimitive, QuantizedBytes,
    },
};

use crate::{
    FloatNdArrayElement, NdArray, NdArrayDevice, NdArrayQTensor, NdArrayTensor, SharedArray,
    element::{IntNdArrayElement, QuantElement},
    execute_with_dtype, execute_with_int_dtype, execute_with_numeric_dtype,
};

use super::quantization::{QuantizationStrategy, SymmetricQuantization};
use super::{NdArrayMathOps, NdArrayOps};

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> QTensorOps<Self>
    for NdArray<E, I, Q>
where
    NdArrayTensor: From<SharedArray<E>>,
    NdArrayTensor: From<SharedArray<I>>,
{
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
                        store: QuantStore::Native | QuantStore::U32,
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
        let data = TensorData::new(values, shape).convert::<Q>();

        NdArrayQTensor {
            qtensor: NdArrayTensor::from_data(data),
            scheme: *scheme,
            qparams,
        }
    }

    fn dequantize(tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        let strategy = tensor.strategy();
        let scheme = tensor.scheme;
        let shape = tensor.shape();
        let data = match tensor.qtensor {
            NdArrayTensor::I8(qtensor) => {
                let data = qtensor.into_iter().collect();
                dequantize(data, shape, scheme, &strategy)
            }
            _ => unreachable!(),
        };
        NdArrayTensor::from_data(data)
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
            qtensor: execute_with_dtype!(tensor.qtensor, |qtensor| NdArrayOps::reshape(
                qtensor, shape
            )),
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
            |qtensor: SharedArray<E>| {
                let values = qtensor.into_iter().collect();
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
            qtensor: execute_with_dtype!(tensor.qtensor, |qtensor| NdArrayOps::swap_dims(
                qtensor, dim1, dim2
            )),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: execute_with_dtype!(tensor.qtensor, |qtensor| NdArrayOps::permute(
                qtensor, axes
            )),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: execute_with_dtype!(tensor.qtensor, |qtensor| NdArrayOps::flip(qtensor, axes)),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_gather(
        dim: usize,
        tensor: QuantizedTensor<Self>,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        let qtensor = execute_with_int_dtype!(indices, I, |indices| -> NdArrayTensor {
            execute_with_numeric_dtype!(tensor.qtensor, |qtensor| {
                NdArrayOps::gather(dim, qtensor, indices)
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
        let qtensor = execute_with_int_dtype!(indices, I, |indices| -> NdArrayTensor {
            execute_with_numeric_dtype!(tensor.qtensor, |qtensor| {
                NdArrayMathOps::select(qtensor, dim, indices)
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
        slices: &[burn_tensor::Slice],
    ) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: execute_with_dtype!(tensor.qtensor, |qtensor| NdArrayOps::slice(
                qtensor, slices
            )),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_argmax(tensor: QuantizedTensor<Self>, dim: usize) -> IntTensor<Self> {
        execute_with_numeric_dtype!(tensor.qtensor, |qtensor| NdArrayMathOps::argmax::<I>(
            qtensor, dim
        ))
    }

    fn q_argmin(tensor: QuantizedTensor<Self>, dim: usize) -> IntTensor<Self> {
        execute_with_numeric_dtype!(tensor.qtensor, |qtensor| NdArrayMathOps::argmin::<I>(
            qtensor, dim
        ))
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: execute_with_dtype!(tensor.qtensor, |qtensor| NdArrayOps::expand(
                qtensor, shape
            )),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }
}

fn dequantize<Q: QuantElement>(
    data: Vec<Q>,
    shape: Shape,
    scheme: QuantScheme,
    strategy: &QuantizationStrategy,
) -> TensorData {
    let qparams = match strategy {
        QuantizationStrategy::PerTensorSymmetric(quant) => vec![quant.scale],
        QuantizationStrategy::PerBlockSymmetric(quant, _block_size) => {
            quant.iter().map(|q| q.scale).collect()
        }
    };
    let q_bytes = QuantizedBytes::new(data, scheme, &qparams);
    let (values, _qparams) = q_bytes.into_vec_i8();
    TensorData::new(strategy.dequantize(&values), shape)
}
