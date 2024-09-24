use std::ops::Range;

use burn_tensor::{
    ops::{FloatTensor, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{QuantizationParametersPrimitive, QuantizationScheme},
    Device, Shape, TensorData,
};

use crate::{client::FusionClient, Fusion, FusionBackend};

impl<B: FusionBackend> QTensorOps<Self> for Fusion<B> {
    fn q_from_data(_data: TensorData, _device: &Device<Self>) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn quantize(
        _tensor: FloatTensor<Self>,
        _scheme: &QuantizationScheme,
        _qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn quantize_dynamic(
        _tensor: FloatTensor<Self>,
        _scheme: &QuantizationScheme,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn dequantize(_tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        unimplemented!()
    }

    fn q_shape(tensor: &QuantizedTensor<Self>) -> Shape {
        tensor.qtensor.shape()
    }

    fn q_device(tensor: &QuantizedTensor<Self>) -> Device<Self> {
        tensor.qtensor.client.device().clone()
    }

    fn q_to_device(
        _tensor: QuantizedTensor<Self>,
        _device: &Device<Self>,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_reshape(_tensor: QuantizedTensor<Self>, _shape: Shape) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    async fn q_into_data(_tensor: QuantizedTensor<Self>) -> TensorData {
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
