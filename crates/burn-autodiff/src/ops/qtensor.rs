use burn_tensor::{
    backend::Backend,
    ops::{FloatTensor, QTensorOps, QuantizedTensor},
    QuantizationStrategy,
};

use crate::{checkpoint::strategy::CheckpointStrategy, Autodiff};

impl<B: Backend, C: CheckpointStrategy> QTensorOps<Self> for Autodiff<B, C> {
    fn quantize<const D: usize>(
        _tensor: FloatTensor<Self, D>,
        _strategy: &QuantizationStrategy,
    ) -> QuantizedTensor<Self, D> {
        unimplemented!()
    }

    fn dequantize<const D: usize>(
        _tensor: QuantizedTensor<Self, D>,
        _strategy: &QuantizationStrategy,
    ) -> FloatTensor<Self, D> {
        unimplemented!()
    }
}
