use core::ops::Range;

use burn_tensor::{
    ops::{FloatTensor, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{
        AffineQuantization, Quantization, QuantizationParametersPrimitive, QuantizationScheme,
        QuantizationStrategy, QuantizationType, SymmetricQuantization,
    },
    DType, Shape, TensorData,
};

use crate::{
    element::{NdArrayElement, QuantElement},
    FloatNdArrayElement, NdArray, NdArrayDevice, NdArrayQTensor, NdArrayTensor,
};

use super::{NdArrayMathOps, NdArrayOps};

fn into_data<E: NdArrayElement>(tensor: NdArrayTensor<E>) -> TensorData {
    let shape = tensor.shape();
    let values = tensor.array.into_iter().collect();
    TensorData::new(values, shape)
}

impl<E: FloatNdArrayElement, Q: QuantElement> QTensorOps<Self> for NdArray<E, Q> {
    fn q_from_data(data: TensorData, _device: &NdArrayDevice) -> QuantizedTensor<Self> {
        match data.dtype {
            DType::QFloat(strategy) => match strategy {
                QuantizationStrategy::PerTensorAffineInt8(_) => {
                    let data = data.convert::<i8>();
                    NdArrayQTensor {
                        qtensor: NdArrayTensor::<Q>::from_data(data),
                        scheme: strategy.scheme(),
                        strategy,
                    }
                }
                QuantizationStrategy::PerTensorSymmetricInt8(_) => {
                    let data = data.convert::<i8>();
                    NdArrayQTensor {
                        qtensor: NdArrayTensor::<Q>::from_data(data),
                        scheme: strategy.scheme(),
                        strategy,
                    }
                }
            },
            _ => panic!(
                "Invalid dtype (expected DType::QFloat, got {:?})",
                data.dtype
            ),
        }
    }

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantizationScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        let strategy = match scheme {
            QuantizationScheme::PerTensorAffine(dtype) => match dtype {
                QuantizationType::QInt8 => {
                    QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(
                        into_data(qparams.scale).iter().next().unwrap(),
                        into_data(qparams.offset.unwrap()).iter().next().unwrap(),
                    ))
                }
            },
            QuantizationScheme::PerTensorSymmetric(dtype) => match dtype {
                QuantizationType::QInt8 => QuantizationStrategy::PerTensorSymmetricInt8(
                    SymmetricQuantization::init(into_data(qparams.scale).iter().next().unwrap()),
                ),
            },
        };

        let data = into_data(tensor).with_quantization(strategy);
        NdArrayQTensor {
            qtensor: NdArrayTensor::<Q>::from_data(data),
            strategy,
            scheme: scheme.clone(),
        }
    }

    fn dequantize(tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        let data = into_data(tensor.qtensor);
        let values = match tensor.strategy {
            QuantizationStrategy::PerTensorAffineInt8(s) => s.dequantize(data.as_slice().unwrap()),
            QuantizationStrategy::PerTensorSymmetricInt8(s) => {
                s.dequantize(data.as_slice().unwrap())
            }
        };
        NdArrayTensor::<E>::from_data(TensorData::new(values, data.shape))
    }

    fn q_shape(tensor: &QuantizedTensor<Self>) -> Shape {
        tensor.qtensor.shape()
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
            strategy: tensor.strategy,
        }
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> TensorData {
        let shape = tensor.qtensor.shape();
        let values = tensor.qtensor.array.into_iter().collect();
        TensorData::quantized(values, shape, tensor.strategy)
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: NdArrayOps::swap_dims(tensor.qtensor, dim1, dim2),
            scheme: tensor.scheme,
            strategy: tensor.strategy,
        }
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: NdArrayOps::permute(tensor.qtensor, axes),
            scheme: tensor.scheme,
            strategy: tensor.strategy,
        }
    }

    fn q_flip(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: NdArrayOps::flip(tensor.qtensor, axes),
            scheme: tensor.scheme,
            strategy: tensor.strategy,
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
            strategy: tensor.strategy,
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
            strategy: tensor.strategy,
        }
    }

    fn q_slice(tensor: QuantizedTensor<Self>, ranges: &[Range<usize>]) -> QuantizedTensor<Self> {
        NdArrayQTensor {
            qtensor: NdArrayOps::slice(tensor.qtensor, ranges),
            scheme: tensor.scheme,
            strategy: tensor.strategy,
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
            strategy: tensor.strategy,
        }
    }
}
