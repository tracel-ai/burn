use burn_tensor::{
    backend::Backend,
    ops::{FloatTensor, QTensorOps, QuantizedTensor},
    QuantizationStrategy,
};

use crate::{
    element::{FloatCandleElement, IntCandleElement},
    Candle,
};

impl<F: FloatCandleElement, I: IntCandleElement> QTensorOps<Self> for Candle<F, I> {
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
