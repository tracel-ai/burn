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

fn into_data<E: NdArrayElement, const D: usize>(tensor: NdArrayTensor<E, D>) -> TensorData {
    let shape = tensor.shape();
    let values = tensor.array.into_iter().collect();
    TensorData::new(values, shape)
}

impl<E: FloatNdArrayElement, Q: QuantElement> QTensorOps<Self> for NdArray<E, Q> {
    fn q_from_data<const D: usize>(
        data: TensorData,
        _device: &NdArrayDevice,
    ) -> QuantizedTensor<Self, D> {
        match data.dtype {
            DType::QFloat(strategy) => match strategy {
                QuantizationStrategy::PerTensorAffineInt8(_) => {
                    let data = data.convert::<i8>();
                    NdArrayQTensor {
                        qtensor: NdArrayTensor::<Q, D>::from_data(data),
                        scheme: strategy.scheme(),
                        strategy,
                    }
                }
                QuantizationStrategy::PerTensorSymmetricInt8(_) => {
                    let data = data.convert::<i8>();
                    NdArrayQTensor {
                        qtensor: NdArrayTensor::<Q, D>::from_data(data),
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

    fn quantize<const D: usize>(
        tensor: FloatTensor<Self, D>,
        scheme: &QuantizationScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self, D> {
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
            qtensor: NdArrayTensor::<Q, D>::from_data(data),
            strategy,
            scheme: scheme.clone(),
        }
    }

    fn dequantize<const D: usize>(tensor: QuantizedTensor<Self, D>) -> FloatTensor<Self, D> {
        let data = into_data(tensor.qtensor);
        let values = match tensor.strategy {
            QuantizationStrategy::PerTensorAffineInt8(s) => s.dequantize(data.as_slice().unwrap()),
            QuantizationStrategy::PerTensorSymmetricInt8(s) => {
                s.dequantize(data.as_slice().unwrap())
            }
        };
        NdArrayTensor::<E, D>::from_data(TensorData::new(values, data.shape))
    }

    fn q_shape<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> Shape<D> {
        tensor.qtensor.shape()
    }

    fn q_device<const D: usize>(_tensor: &QuantizedTensor<Self, D>) -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    fn q_to_device<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        _device: &NdArrayDevice,
    ) -> QuantizedTensor<Self, D> {
        tensor
    }

    fn q_reshape<const D1: usize, const D2: usize>(
        tensor: QuantizedTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> QuantizedTensor<Self, D2> {
        NdArrayQTensor {
            qtensor: NdArrayOps::reshape(tensor.qtensor, shape),
            scheme: tensor.scheme,
            strategy: tensor.strategy,
        }
    }

    async fn q_into_data<const D: usize>(tensor: QuantizedTensor<Self, D>) -> TensorData {
        let shape = tensor.qtensor.shape();
        let values = tensor.qtensor.array.into_iter().collect();
        TensorData::quantized(values, shape, tensor.strategy)
    }

    fn q_swap_dims<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self, D> {
        NdArrayQTensor {
            qtensor: NdArrayOps::swap_dims(tensor.qtensor, dim1, dim2),
            scheme: tensor.scheme,
            strategy: tensor.strategy,
        }
    }

    fn q_permute<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        axes: [usize; D],
    ) -> QuantizedTensor<Self, D> {
        NdArrayQTensor {
            qtensor: NdArrayOps::permute(tensor.qtensor, axes),
            scheme: tensor.scheme,
            strategy: tensor.strategy,
        }
    }

    fn q_flip<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        axes: &[usize],
    ) -> QuantizedTensor<Self, D> {
        NdArrayQTensor {
            qtensor: NdArrayOps::flip(tensor.qtensor, axes),
            scheme: tensor.scheme,
            strategy: tensor.strategy,
        }
    }

    fn q_gather<const D: usize>(
        dim: usize,
        tensor: QuantizedTensor<Self, D>,
        indices: IntTensor<Self, D>,
    ) -> QuantizedTensor<Self, D> {
        NdArrayQTensor {
            qtensor: NdArrayMathOps::gather(dim, tensor.qtensor, indices),
            scheme: tensor.scheme,
            strategy: tensor.strategy,
        }
    }

    fn q_select<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
    ) -> QuantizedTensor<Self, D> {
        NdArrayQTensor {
            qtensor: NdArrayMathOps::select(tensor.qtensor, dim, indices),
            scheme: tensor.scheme,
            strategy: tensor.strategy,
        }
    }

    fn q_slice<const D1: usize, const D2: usize>(
        tensor: QuantizedTensor<Self, D1>,
        ranges: [Range<usize>; D2],
    ) -> QuantizedTensor<Self, D1> {
        NdArrayQTensor {
            qtensor: NdArrayOps::slice(tensor.qtensor, ranges),
            scheme: tensor.scheme,
            strategy: tensor.strategy,
        }
    }

    fn q_argmax<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        dim: usize,
    ) -> IntTensor<Self, D> {
        NdArrayMathOps::argmax(tensor.qtensor, dim)
    }

    fn q_argmin<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        dim: usize,
    ) -> IntTensor<Self, D> {
        NdArrayMathOps::argmin(tensor.qtensor, dim)
    }

    fn q_expand<const D1: usize, const D2: usize>(
        tensor: QuantizedTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> QuantizedTensor<Self, D2> {
        NdArrayQTensor {
            qtensor: NdArrayOps::expand(tensor.qtensor, shape),
            scheme: tensor.scheme,
            strategy: tensor.strategy,
        }
    }
}
