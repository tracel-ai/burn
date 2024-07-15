use burn_tensor::{
    ops::{FloatTensor, QTensorOps, QuantizedTensor},
    quantization::QuantizationStrategy,
    Device, Shape, TensorData,
};

use crate::{FloatElement, IntElement, JitBackend, JitRuntime, QJitTensor};

impl<R, F, I> QTensorOps<Self> for JitBackend<R, F, I>
where
    R: JitRuntime,
    F: FloatElement,
    I: IntElement,
{
    fn q_from_data<const D: usize>(
        _data: TensorData,
        _device: &Device<Self>,
    ) -> QuantizedTensor<Self, D> {
        todo!()
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
        tensor.qtensor.shape.clone()
    }

    fn q_device<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> Device<Self> {
        tensor.qtensor.device.clone()
    }

    fn q_reshape<const D1: usize, const D2: usize>(
        tensor: QuantizedTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> QuantizedTensor<Self, D2> {
        QJitTensor {
            qtensor: super::reshape(tensor.qtensor, shape),
            strategy: tensor.strategy,
        }
    }

    async fn q_into_data<const D: usize>(_tensor: QuantizedTensor<Self, D>) -> TensorData {
        unimplemented!()
    }
}
