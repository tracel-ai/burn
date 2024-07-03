use burn_tensor::{
    ops::{FloatTensor, QTensorOps, QuantizedTensor},
    QuantizationStrategy, Shape,
};

use crate::{LibTorch, LibTorchDevice, TchElement, TchTensor};

impl<E: TchElement> QTensorOps<Self> for LibTorch<E> {
    fn quantize<const D: usize>(
        tensor: FloatTensor<Self, D>,
        strategy: &QuantizationStrategy,
    ) -> QuantizedTensor<Self, D> {
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
}
