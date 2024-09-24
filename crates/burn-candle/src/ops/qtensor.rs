use std::ops::Range;

use burn_tensor::{
    backend::Backend,
    ops::{FloatTensor, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{QuantizationParametersPrimitive, QuantizationScheme, QuantizationStrategy},
    DType, Device, Shape, TensorData,
};

use crate::{
    element::{FloatCandleElement, IntCandleElement},
    Candle, CandleQTensor,
};

impl<F: FloatCandleElement, I: IntCandleElement> QTensorOps<Self> for Candle<F, I> {
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        unimplemented!() // no i8 support
    }

    fn quantize(
        _tensor: FloatTensor<Self>,
        _scheme: &QuantizationScheme,
        _qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn dequantize(_tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        unimplemented!()
    }

    fn q_shape(tensor: &QuantizedTensor<Self>) -> Shape {
        super::base::shape(&tensor.qtensor)
    }

    fn q_device(tensor: &QuantizedTensor<Self>) -> Device<Self> {
        super::base::device(&tensor.qtensor)
    }

    fn q_to_device(
        _tensor: QuantizedTensor<Self>,
        _device: &Device<Self>,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        CandleQTensor {
            qtensor: super::base::reshape(tensor.qtensor, shape),
            scheme: tensor.scheme,
        }
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> TensorData {
        unimplemented!()
    }

    fn q_swap_dims(
        _tensor: QuantizedTensor<Self>,
        _dim1: usize,
        _dim2: usize,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_permute(_tensor: QuantizedTensor<Self>, _axes: &[usize]) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_flip(_tensor: QuantizedTensor<Self>, _axes: &[usize]) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_gather(
        _dim: usize,
        _tensor: QuantizedTensor<Self>,
        _indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_select(
        _tensor: QuantizedTensor<Self>,
        _dim: usize,
        _indices: IntTensor<Self>,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_slice(_tensor: QuantizedTensor<Self>, _ranges: &[Range<usize>]) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_expand(_tensor: QuantizedTensor<Self>, _shape: Shape) -> QuantizedTensor<Self> {
        unimplemented!()
    }
}
