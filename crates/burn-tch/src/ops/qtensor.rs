use burn_tensor::{
    ops::{FloatTensor, QTensorOps, QuantizedTensor},
    quantization::{
        QTensorPrimitive, Quantization, QuantizationParametersPrimitive, QuantizationScheme,
        QuantizationStrategy, QuantizationType,
    },
    DType, Shape, TensorData,
};

use crate::{LibTorch, LibTorchDevice, QuantElement, TchElement, TchQTensor, TchShape, TchTensor};

use super::TchOps;

impl<E: TchElement, Q: QuantElement> QTensorOps<Self> for LibTorch<E, Q> {
    fn q_from_data<const D: usize>(
        data: TensorData,
        device: &LibTorchDevice,
    ) -> QuantizedTensor<Self, D> {
        let shape_tch = TchShape::<D>::from(data.shape.as_slice());
        let device = (*device).into();

        // NOTE: tch-rs doesn't have `from_blob_quantized_*` APIs
        // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/quantized/Quantizer.cpp#L322
        // So for now we have to load the dequantized values to quantize them back since the dequantization
        // methods take the values provided when quantizing.
        let (tensor, scheme) = match data.dtype {
            DType::QFloat(strategy) => match strategy {
                QuantizationStrategy::PerTensorAffineInt8(q) => {
                    let values = q.dequantize(&data.iter::<i8>().collect::<Vec<_>>());
                    let tensor = tch::Tensor::from_slice(&values).to(device);
                    let tensor = TchOps::<E>::quantize::<D, i8>(
                        TchTensor::new(tensor.reshape(shape_tch.dims)),
                        &strategy,
                    )
                    .tensor;
                    (tensor, strategy.scheme())
                }
                QuantizationStrategy::PerTensorSymmetricInt8(q) => {
                    let values = q.dequantize(&data.iter::<i8>().collect::<Vec<_>>());
                    let tensor = tch::Tensor::from_slice(&values).to(device);
                    let tensor = TchOps::<E>::quantize::<D, i8>(
                        TchTensor::new(tensor.reshape(shape_tch.dims)),
                        &strategy,
                    )
                    .tensor;
                    (tensor, strategy.scheme())
                }
            },
            _ => panic!(
                "Invalid dtype (expected DType::QFloat, got {:?})",
                data.dtype
            ),
        };
        TchQTensor {
            qtensor: TchTensor::new(tensor),
            scheme,
        }
    }

    fn quantize<const D: usize>(
        tensor: FloatTensor<Self, D>,
        scheme: &QuantizationScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self, D> {
        let mut tensor = tensor;
        // Quantize only works on Float Tensor
        if E::dtype() == DType::F16 {
            tensor.tensor = tensor.tensor.to_kind(tch::Kind::Float);
        }

        let qtensor = match scheme {
            QuantizationScheme::PerTensorAffine(dtype) => match dtype {
                QuantizationType::QInt8 => tensor.tensor.quantize_per_tensor_tensor_qparams(
                    &qparams.scale.tensor,
                    &qparams.offset.unwrap().tensor,
                    tch::Kind::QInt8,
                ),
            },
            QuantizationScheme::PerTensorSymmetric(_) => {
                tensor.tensor.quantize_per_tensor_tensor_qparams(
                    &qparams.scale.tensor,
                    &tch::Tensor::zeros_like(&qparams.scale.tensor),
                    tch::Kind::QInt8,
                )
            }
        };

        TchQTensor {
            qtensor: TchTensor::new(qtensor),
            scheme: scheme.clone(),
        }
    }

    fn dequantize<const D: usize>(tensor: QuantizedTensor<Self, D>) -> FloatTensor<Self, D> {
        TchTensor::new(tensor.qtensor.tensor.dequantize().to_kind(E::KIND))
    }

    fn q_shape<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> Shape<D> {
        tensor.qtensor.shape()
    }

    fn q_device<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> LibTorchDevice {
        tensor.qtensor.tensor.device().into()
    }

    fn q_reshape<const D1: usize, const D2: usize>(
        tensor: QuantizedTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> QuantizedTensor<Self, D2> {
        TchQTensor {
            qtensor: TchOps::reshape(tensor.qtensor, shape),
            scheme: tensor.scheme,
        }
    }

    async fn q_into_data<const D: usize>(tensor: QuantizedTensor<Self, D>) -> TensorData {
        let shape = Self::q_shape(&tensor);
        let tensor = Self::q_reshape(tensor.clone(), Shape::new([shape.num_elements()]));
        let strategy = tensor.strategy();

        // To get the integer values we have to call `int_repr()`
        let values: Result<Vec<i8>, tch::TchError> = tensor.qtensor.tensor.int_repr().try_into();

        TensorData::quantized(values.unwrap(), shape, strategy)
    }
}
