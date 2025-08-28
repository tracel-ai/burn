use alloc::vec;
use core::ops::Range;

use burn_tensor::{
    DType, Shape, TensorData, TensorMetadata,
    ops::{FloatTensor, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{
        QParams, QuantLevel, QuantMode, QuantScheme, QuantStore, QuantValue,
        QuantizationParametersPrimitive, QuantizationStrategy, QuantizedBytes,
        SymmetricQuantization,
    },
};

use crate::{
    FloatNdArrayElement, NdArray, NdArrayDevice, NdArrayQTensor, NdArrayTensor, SharedArray,
    element::{IntNdArrayElement, QuantElement},
    execute_with_dtype, execute_with_int_dtype, execute_with_numeric_dtype,
};

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
                        store: QuantStore::Native,
                        ..
                    } => {
                        // We should probably check that `Q` matches i8.. but it's the only valid type now
                        let (values, qparams) = q_bytes.into_vec_i8();
                        let data = TensorData::new(values, shape);

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
                        value: QuantValue::Q4F | QuantValue::Q4S | QuantValue::Q2F | QuantValue::Q2S,
                        store: QuantStore::Native,
                        ..
                    }
                    | QuantScheme {
                        store: QuantStore::U32,
                        ..
                    } => unimplemented!(),
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
        // Implement with ndarray instead of QuantizationStrategy?
        let (strategy, qparams) = match scheme {
            QuantScheme {
                level: QuantLevel::Tensor,
                mode: QuantMode::Symmetric,
                value: QuantValue::Q8F | QuantValue::Q8S,
                store: QuantStore::Native,
                ..
            } => {
                let scales = qparams.scales.into_data().iter().next().unwrap();
                (
                    QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(
                        scales,
                    )),
                    vec![QParams { scales }],
                )
            }
            QuantScheme {
                level: QuantLevel::Block(block_size),
                mode: QuantMode::Symmetric,
                value: QuantValue::Q8F | QuantValue::Q8S,
                store: QuantStore::Native,
                ..
            } => {
                let (strategy, qparams) = qparams
                    .scales
                    .into_data()
                    .iter()
                    .map(|s| (SymmetricQuantization::init(s), QParams { scales: s }))
                    .unzip();
                (
                    QuantizationStrategy::PerBlockSymmetricInt8(strategy, *block_size),
                    qparams,
                )
            }
            QuantScheme {
                value: QuantValue::Q4F | QuantValue::Q4S | QuantValue::Q2F | QuantValue::Q2S,
                store: QuantStore::Native,
                ..
            }
            | QuantScheme {
                store: QuantStore::U32,
                ..
            } => unimplemented!(),
        };

        let shape = tensor.shape();
        let data = tensor.into_data().with_quantization(strategy);
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
        let shape = tensor.qtensor.shape();
        let strategy = tensor.strategy();
        let data: TensorData =
            execute_with_numeric_dtype!(tensor.qtensor, E, |qtensor: SharedArray<E>| {
                let values = qtensor.into_iter().collect();
                TensorData::quantized(values, shape, strategy)
            });
        NdArrayTensor::from_data(data.dequantize().unwrap())
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

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> TensorData {
        let strategy = tensor.strategy();
        let shape = tensor.qtensor.shape();
        execute_with_numeric_dtype!(tensor.qtensor, E, |qtensor: SharedArray<E>| {
            let values = qtensor.into_iter().collect();
            TensorData::quantized(values, shape, strategy)
        })
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
                NdArrayMathOps::gather(dim, qtensor, indices)
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

    fn q_slice(tensor: QuantizedTensor<Self>, ranges: &[Range<usize>]) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: execute_with_dtype!(tensor.qtensor, |qtensor| NdArrayOps::slice(
                qtensor, ranges
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
