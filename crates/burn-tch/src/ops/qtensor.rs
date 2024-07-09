use burn_tensor::{
    ops::{FloatTensor, QTensorOps, QuantizedTensor},
    DType, QuantizationStrategy, Shape, TensorData,
};

use crate::{LibTorch, LibTorchDevice, TchElement, TchTensor};

use super::TchOps;

impl<E: TchElement> QTensorOps<Self> for LibTorch<E> {
    fn quantize<const D: usize>(
        tensor: FloatTensor<Self, D>,
        strategy: &QuantizationStrategy,
    ) -> QuantizedTensor<Self, D> {
        let mut tensor = tensor;
        // Quantize only works on Float Tensor
        if E::dtype() == DType::F16 {
            tensor.tensor = tensor.tensor.to_kind(tch::Kind::Float);
        }

        match strategy {
            QuantizationStrategy::PerTensorAffineInt8(ref q) => {
                TchTensor::new(tensor.tensor.quantize_per_tensor(
                    q.scale.into(),
                    q.offset.into(),
                    tch::Kind::QInt8,
                ))
            }
            QuantizationStrategy::PerTensorSymmetricInt8(ref q) => TchTensor::new(
                tensor
                    .tensor
                    .quantize_per_tensor(q.scale.into(), 0, tch::Kind::QInt8),
            ),
        }
    }

    fn dequantize<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        _strategy: &QuantizationStrategy,
    ) -> FloatTensor<Self, D> {
        TchTensor::new(tensor.tensor.dequantize())
    }

    fn q_shape<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> Shape<D> {
        tensor.shape()
    }

    fn q_device<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> LibTorchDevice {
        tensor.tensor.device().into()
    }

    fn q_reshape<const D1: usize, const D2: usize>(
        tensor: QuantizedTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> QuantizedTensor<Self, D2> {
        TchOps::reshape(tensor, shape)
    }

    async fn q_into_data<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        strategy: QuantizationStrategy,
    ) -> TensorData {
        let shape = Self::q_shape(&tensor);
        let tensor = Self::q_reshape(tensor.clone(), Shape::new([shape.num_elements()]));
        // To get the integer values we have to call `int_repr()`
        let values: Result<Vec<i8>, tch::TchError> = tensor.tensor.int_repr().try_into();

        TensorData::quantized(values.unwrap(), shape, strategy)
    }
}
