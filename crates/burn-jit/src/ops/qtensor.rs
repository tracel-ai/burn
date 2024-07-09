use burn_tensor::{
    ops::{FloatTensor, QTensorOps, QuantizedTensor},
    Device, QuantizationStrategy, Shape, TensorData,
};

use crate::{FloatElement, IntElement, JitBackend, JitRuntime};

impl<R, F, I> QTensorOps<Self> for JitBackend<R, F, I>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
{
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
        tensor.shape.clone()
    }

    fn q_device<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> Device<Self> {
        tensor.device.clone()
    }

    fn q_reshape<const D1: usize, const D2: usize>(
        tensor: QuantizedTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> QuantizedTensor<Self, D2> {
        super::reshape(tensor, shape)
    }

    async fn q_into_data<const D: usize>(
        _tensor: QuantizedTensor<Self, D>,
        _strategy: QuantizationStrategy,
    ) -> TensorData {
        unimplemented!()
    }
}
