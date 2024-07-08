use burn_tensor::{
    backend::Backend,
    ops::{FloatTensor, QTensorOps, QuantizedTensor},
    Device, QuantizationStrategy, Shape,
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

    fn q_shape<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> Shape<D> {
        super::base::shape(tensor)
    }

    fn q_device<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> Device<Self> {
        super::base::device(tensor)
    }
}
