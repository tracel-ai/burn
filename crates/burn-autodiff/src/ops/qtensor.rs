use burn_tensor::{
    backend::Backend,
    ops::{FloatTensor, QTensorOps, QuantizedTensor},
    Device, QuantizationStrategy, Shape,
};

use crate::{checkpoint::strategy::CheckpointStrategy, Autodiff};

impl<B: Backend, C: CheckpointStrategy> QTensorOps<Self> for Autodiff<B, C> {
    fn quantize<const D: usize>(
        _tensor: FloatTensor<Self, D>,
        _strategy: &QuantizationStrategy,
    ) -> QuantizedTensor<Self, D> {
        todo!() // required for QAT
    }

    fn dequantize<const D: usize>(
        _tensor: QuantizedTensor<Self, D>,
        _strategy: &QuantizationStrategy,
    ) -> FloatTensor<Self, D> {
        todo!()
    }

    fn q_shape<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> Shape<D> {
        B::q_shape(tensor)
    }

    fn q_device<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> Device<Self> {
        B::q_device(tensor)
    }
}
