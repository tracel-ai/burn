use alloc::vec;
use core::ops::Range;

use burn_tensor::{
    DType, Shape, TensorData, TensorMetadata,
    ops::{FloatTensor, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{
        QParams, QuantInputType, QuantLevel, QuantMode, QuantScheme,
        QuantizationParametersPrimitive, QuantizationStrategy, QuantizedBytes,
        SymmetricQuantization,
    },
};

use crate::{
    FloatNdArrayElement, NdArray, NdArrayDevice, NdArrayQTensor, NdArrayTensor, NdArrayTensorFloat,
    element::{IntNdArrayElement, NdArrayElement, QuantElement},
    new_tensor_float,
};

use super::{NdArrayMathOps, NdArrayOps};

fn into_data<E: NdArrayElement>(tensor: NdArrayTensor<E>) -> TensorData {
    let shape = tensor.shape();
    let values = tensor.array.into_iter().collect();
    TensorData::new(values, shape)
}

fn into_data_f(tensor: NdArrayTensorFloat) -> TensorData {
    match tensor {
        NdArrayTensorFloat::F32(tensor) => into_data(tensor),
        NdArrayTensorFloat::F64(tensor) => into_data(tensor),
    }
}

impl<E: FloatNdArrayElement, I: IntNdArrayElement, Q: QuantElement> QTensorOps<Self>
    for NdArray<E, I, Q>
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
                        q_type: QuantInputType::QInt8,
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
                            qtensor: NdArrayTensor::<Q>::from_data(data),
                            scheme,
                            qparams,
                        }
                    }
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
                q_type: QuantInputType::QInt8,
                ..
            } => {
                let scales = into_data_f(qparams.scales).iter().next().unwrap();
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
                q_type: QuantInputType::QInt8,
                ..
            } => {
                let (strategy, qparams) = into_data_f(qparams.scales)
                    .iter()
                    .map(|s| (SymmetricQuantization::init(s), QParams { scales: s }))
                    .unzip();
                (
                    QuantizationStrategy::PerBlockSymmetricInt8(strategy, *block_size),
                    qparams,
                )
            }
        };

        let shape = tensor.shape();
        let data = into_data_f(tensor).with_quantization(strategy);
        let num_elements = data.num_elements();
        let q_bytes = QuantizedBytes {
            bytes: data.into_bytes(),
            scheme: *scheme,
            num_elements,
        };
        let (values, _) = q_bytes.into_vec_i8();
        let data = TensorData::new(values, shape).convert::<Q>();

        NdArrayQTensor {
            qtensor: NdArrayTensor::<Q>::from_data(data),
            scheme: *scheme,
            qparams,
        }
    }

    fn dequantize(tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        let shape = tensor.qtensor.shape();
        let strategy = tensor.strategy();
        let values = tensor.qtensor.array.into_iter().collect();
        let data = TensorData::quantized(values, shape, strategy);
        new_tensor_float!(NdArrayTensor::from_data(data.dequantize().unwrap()))
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
            qtensor: NdArrayOps::reshape(tensor.qtensor, shape),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> TensorData {
        let strategy = tensor.strategy();
        let shape = tensor.qtensor.shape();
        let values = tensor.qtensor.array.into_iter().collect();
        TensorData::quantized(values, shape, strategy)
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: NdArrayOps::swap_dims(tensor.qtensor, dim1, dim2),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: NdArrayOps::permute(tensor.qtensor, axes),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: NdArrayOps::flip(tensor.qtensor, axes),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_gather(
        dim: usize,
        tensor: QuantizedTensor<Self>,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: NdArrayMathOps::gather(dim, tensor.qtensor, indices),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_select(
        tensor: QuantizedTensor<Self>,
        dim: usize,
        indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: NdArrayMathOps::select(tensor.qtensor, dim, indices),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_slice(tensor: QuantizedTensor<Self>, ranges: &[Range<usize>]) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: NdArrayOps::slice(tensor.qtensor, ranges),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }

    fn q_argmax(tensor: QuantizedTensor<Self>, dim: usize) -> IntTensor<Self> {
        NdArrayMathOps::argmax(tensor.qtensor, dim)
    }

    fn q_argmin(tensor: QuantizedTensor<Self>, dim: usize) -> IntTensor<Self> {
        NdArrayMathOps::argmin(tensor.qtensor, dim)
    }

    fn q_expand(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: NdArrayOps::expand(tensor.qtensor, shape),
            scheme: tensor.scheme,
            qparams: tensor.qparams,
        }
    }
}
