use burn_backend::{
    ExecutionError, Shape, Slice, TensorData,
    ops::QTensorOps,
    quantization::{QuantScheme, QuantizationParametersPrimitive},
    tensor::{Device, FloatTensor, IntTensor, QuantizedTensor},
};

use crate::{BackendRouter, RunnerChannel};

impl<R: RunnerChannel> QTensorOps<Self> for BackendRouter<R> {
    fn q_from_data(_data: TensorData, _device: &Device<Self>) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn quantize(
        _tensor: FloatTensor<Self>,
        _scheme: &QuantScheme,
        _qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn quantize_dynamic(
        _tensor: FloatTensor<Self>,
        _scheme: &QuantScheme,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn dequantize(_tensor: QuantizedTensor<Self>) -> FloatTensor<Self> {
        unimplemented!()
    }

    fn q_device(_tensor: &QuantizedTensor<Self>) -> Device<Self> {
        unimplemented!()
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

    async fn q_into_data(_tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
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

    fn q_slice(_tensor: QuantizedTensor<Self>, _slices: &[Slice]) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_expand(_tensor: QuantizedTensor<Self>, _shape: Shape) -> QuantizedTensor<Self> {
        unimplemented!()
    }
}
