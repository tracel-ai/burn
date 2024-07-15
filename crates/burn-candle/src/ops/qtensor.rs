use burn_tensor::{
    backend::Backend,
    ops::{FloatTensor, QTensorOps, QuantizedTensor},
    quantization::QuantizationStrategy,
    DType, Device, Shape, TensorData,
};

use crate::{
    element::{FloatCandleElement, IntCandleElement},
    Candle, QCandleTensor,
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
        _strategy: QuantizationStrategy,
    ) -> QuantizedTensor<Self, D> {
        unimplemented!()
    }

    fn dequantize<const D: usize>(_tensor: QuantizedTensor<Self, D>) -> FloatTensor<Self, D> {
        unimplemented!()
    }

    fn q_shape<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> Shape<D> {
        super::base::shape(&tensor.qtensor)
    }

    fn q_device<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> Device<Self> {
        super::base::device(&tensor.qtensor)
    }

    fn q_reshape<const D1: usize, const D2: usize>(
        tensor: QuantizedTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> QuantizedTensor<Self, D2> {
        QCandleTensor {
            qtensor: super::base::reshape(tensor.qtensor, shape),
            strategy: tensor.strategy,
        }
    }

    async fn q_into_data<const D: usize>(tensor: QuantizedTensor<Self, D>) -> TensorData {
        unimplemented!()
    }
}
