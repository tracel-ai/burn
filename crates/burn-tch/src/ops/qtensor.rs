use burn_tensor::{
    backend::Backend,
    ops::{QTensorOps, QuantizedTensor},
    QuantizationStrategy, Reader, Shape, TensorData,
};

use crate::{LibTorch, TchElement, TchTensor};

use super::TchOps;

impl<E: TchElement> QTensorOps<Self> for LibTorch<E> {
    fn quantize<const D: usize>(
        tensor: <Self as Backend>::FloatTensorPrimitive<D>,
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
    ) -> <Self as Backend>::FloatTensorPrimitive<D> {
        TchTensor::new(tensor.tensor.dequantize())
    }

    fn quantized_into_data<const D: usize>(tensor: QuantizedTensor<Self, D>) -> Reader<TensorData> {
        let shape = tensor.shape();
        let tensor = TchOps::reshape(tensor.clone(), Shape::new([shape.num_elements()]));
        // We have to call `.int_repr()` to get a CPU tensor with the underlying int data
        let values: Result<Vec<i8>, tch::TchError> =
            tensor.tensor.int_repr().shallow_clone().try_into();

        Reader::Concrete(TensorData::new(values.unwrap(), shape))
    }
}
