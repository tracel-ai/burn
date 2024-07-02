use burn_tensor::{
    ops::{FloatTensor, QTensorOps, QuantizedTensor},
    QuantizationStrategy,
};

use crate::{FloatNdArrayElement, NdArray};

impl<E: FloatNdArrayElement> QTensorOps<Self> for NdArray<E> {
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
