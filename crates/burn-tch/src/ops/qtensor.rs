use burn_tensor::{
    ops::{FloatTensor, QTensorOps, QuantizedTensor},
    quantization::{Quantization, QuantizationStrategy},
    DType, Shape, TensorData,
};

use crate::{LibTorch, LibTorchDevice, QTchTensor, TchElement, TchShape, TchTensor};

use super::TchOps;

impl<E: TchElement> QTensorOps<Self> for LibTorch<E> {
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
        let (tensor, strategy) = match data.dtype {
            DType::QFloat(strategy) => match strategy {
                QuantizationStrategy::PerTensorAffineInt8(q) => {
                    let values = q.dequantize(&data.iter::<i8>().collect::<Vec<_>>());
                    let tensor = tch::Tensor::from_slice(&values).to(device);
                    let tensor = TchOps::<E>::quantize::<D, i8>(
                        TchTensor::new(tensor.reshape(shape_tch.dims)),
                        &strategy,
                    )
                    .tensor;
                    (tensor, strategy)
                }
                QuantizationStrategy::PerTensorSymmetricInt8(q) => {
                    let values = q.dequantize(&data.iter::<i8>().collect::<Vec<_>>());
                    let tensor = tch::Tensor::from_slice(&values).to(device);
                    let tensor = TchOps::<E>::quantize::<D, i8>(
                        TchTensor::new(tensor.reshape(shape_tch.dims)),
                        &strategy,
                    )
                    .tensor;
                    (tensor, strategy)
                }
            },
            _ => panic!(
                "Invalid dtype (expected DType::QFloat, got {:?})",
                data.dtype
            ),
        };
        QTchTensor {
            qtensor: TchTensor::new(tensor),
            strategy,
        }
    }

    fn quantize<const D: usize>(
        tensor: FloatTensor<Self, D>,
        strategy: QuantizationStrategy,
    ) -> QuantizedTensor<Self, D> {
        let mut tensor = tensor;
        // Quantize only works on Float Tensor
        if E::dtype() == DType::F16 {
            tensor.tensor = tensor.tensor.to_kind(tch::Kind::Float);
        }

        match strategy {
            QuantizationStrategy::PerTensorAffineInt8(ref q) => QTchTensor {
                qtensor: TchTensor::new(tensor.tensor.quantize_per_tensor(
                    q.scale.into(),
                    q.offset.into(),
                    tch::Kind::QInt8,
                )),
                strategy,
            },
            QuantizationStrategy::PerTensorSymmetricInt8(ref q) => QTchTensor {
                qtensor: TchTensor::new(tensor.tensor.quantize_per_tensor(
                    q.scale.into(),
                    0,
                    tch::Kind::QInt8,
                )),
                strategy,
            },
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
        QTchTensor {
            qtensor: TchOps::reshape(tensor.qtensor, shape),
            strategy: tensor.strategy,
        }
    }

    async fn q_into_data<const D: usize>(tensor: QuantizedTensor<Self, D>) -> TensorData {
        let shape = Self::q_shape(&tensor);
        let tensor = Self::q_reshape(tensor.clone(), Shape::new([shape.num_elements()]));
        // To get the integer values we have to call `int_repr()`
        let values: Result<Vec<i8>, tch::TchError> = tensor.qtensor.tensor.int_repr().try_into();

        TensorData::quantized(values.unwrap(), shape, tensor.strategy)
    }
}
