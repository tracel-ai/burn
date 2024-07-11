use burn_tensor::{
    backend::Backend,
    ops::{FloatTensor, QTensorOps, QuantizedTensor},
    DType, Device, QuantizationStrategy, Shape, TensorData,
};

use crate::{
    element::{FloatCandleElement, IntCandleElement},
    Candle,
};

impl<F: FloatCandleElement, I: IntCandleElement> QTensorOps<Self> for Candle<F, I> {
    fn q_from_data<const D: usize>(
        data: TensorData,
        device: &Device<Self>,
    ) -> QuantizedTensor<Self, D> {
        unimplemented!() // no i8 support
    }

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

    fn q_reshape<const D1: usize, const D2: usize>(
        tensor: QuantizedTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> QuantizedTensor<Self, D2> {
        super::base::reshape(tensor, shape)
    }

    async fn q_into_data<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        strategy: QuantizationStrategy,
    ) -> TensorData {
        super::base::into_data(tensor)
    }
}
